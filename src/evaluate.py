"""
Evaluation harness that reproduces the PAPER'S protocol -- not the shortcut.

WHY THIS IS NOT just `roc_auc_score`:
  A normal AUC sweeps the decision *threshold* of one trained model over its
  predicted probabilities. The paper does something different (Section 5.2): it
  sweeps the *training-set class balance*. Each point on the ROC curve is a
  separately trained classifier -- fixed SMOTE level on the minority, and an
  increasing amount of random UNDER-sampling of the majority. More
  under-sampling pushes the operating point up-and-right (more true positives,
  more false positives). Those points trace the curve; we then take the area
  under it with the trapezoid rule (Section 5.3).

  Reproducing this protocol -- rather than calling roc_auc_score -- is the whole
  reason this counts as reproducing the paper.

THE GOLDEN RULE, enforced here:
  SMOTE and under-sampling touch the TRAINING fold ONLY. The held-out test fold
  stays a pristine sample of the real, imbalanced world. Resampling before the
  split is data leakage and inflates your score into a lie.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from smote import smote_resample


def _undersample_majority(X, y, minority_label, majority_target, rng):
    """Keep all minority rows; randomly keep `majority_target` majority rows."""
    minority_mask = (y == minority_label)
    maj_idx = np.where(~minority_mask)[0]
    min_idx = np.where(minority_mask)[0]

    if majority_target < len(maj_idx):
        keep = rng.choice(maj_idx, size=majority_target, replace=False)
    else:
        keep = maj_idx  # asked for more than we have -> keep all (no under-sampling)

    idx = np.concatenate([min_idx, keep])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _tp_fp_rates(y_true, y_pred, minority_label):
    """Return (%TP, %FP) as percentages, matching the paper's ROC axes."""
    pos = (y_true == minority_label)
    neg = ~pos
    TP = int((y_pred[pos] == minority_label).sum())
    FN = int(pos.sum()) - TP
    FP = int((y_pred[neg] == minority_label).sum())
    TN = int(neg.sum()) - FP
    tpr = 100.0 * TP / (TP + FN) if (TP + FN) else 0.0
    fpr = 100.0 * FP / (FP + TN) if (FP + TN) else 0.0
    return tpr, fpr


def _trapezoid_auc(points):
    """Area under the ROC curve (points are (%FP, %TP) in 0..100).

    We sort by %FP, anchor the curve at (0,0) and (100,100) as the paper does
    for the top-right, and integrate with the trapezoid rule. Divide by 10000
    to report on the familiar 0..1 AUC scale (paper prints x10000, e.g. 7242).
    """
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    pts = [(0.0, 0.0)] + pts + [(100.0, 100.0)]
    # collapse duplicate %FP values, keeping the max %TP (upper envelope)
    dedup = {}
    for fp, tp in pts:
        dedup[fp] = max(dedup.get(fp, 0.0), tp)
    fps = sorted(dedup)
    tps = [dedup[f] for f in fps]
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.0 renamed it
    area = trapz(tps, fps)             # area in the 100x100 box
    return area / 10000.0              # -> 0..1


def roc_sweep_auc(X, y, minority_label, smote_percent, undersample_levels,
                  k=5, cv_folds=10, random_state=42):
    """Run the full paper protocol for one (dataset, SMOTE level) and return AUC.

    smote_percent = 0  -> plain under-sampling curve ("Under" in the paper).
    smote_percent > 0  -> SMOTE + under-sampling curve ("SMOTE" in the paper).
    """
    rng = np.random.default_rng(random_state)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    roc_points = []
    for U in undersample_levels:
        fold_tp, fold_fp = [], []
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            n_minor = int((y_tr == minority_label).sum())  # base for under-sampling

            # 1) over-sample minority with SMOTE (training fold only)
            if smote_percent > 0:
                X_tr, y_tr = smote_resample(
                    X_tr, y_tr, minority_label, N=smote_percent, k=k,
                    random_state=int(rng.integers(0, 1_000_000)),
                )

            # 2) under-sample majority. Level U%: majority_target = n_minor*100/U.
            #    Higher U -> fewer majority kept -> more aggressive under-sampling.
            majority_target = int(round(n_minor * 100.0 / U))
            X_tr, y_tr = _undersample_majority(
                X_tr, y_tr, minority_label, majority_target, rng)

            # 3) train the actual model and score on the untouched test fold
            clf = DecisionTreeClassifier(random_state=random_state)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            tpr, fpr = _tp_fp_rates(y_te, y_pred, minority_label)
            fold_tp.append(tpr)
            fold_fp.append(fpr)

        roc_points.append((float(np.mean(fold_fp)), float(np.mean(fold_tp))))

    return _trapezoid_auc(roc_points), roc_points
