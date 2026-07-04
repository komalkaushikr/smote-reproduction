"""Stage 5: prove my from-scratch SMOTE is correct by comparing it to the
battle-tested imbalanced-learn implementation on identical input.

They will not be byte-identical (different RNG, different neighbour tie-breaks),
but the synthetic points must occupy the same region and a classifier trained on
each must score almost identically.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from data_loader import load_pima
from smote import smote_resample
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE as ImbSMOTE

def _cv_auc(resampler):
    """Plain 10-fold CV AUC using a given resampling function on the train fold."""
    X, y, minority_label = load_pima(config.PIMA_CSV)
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                          random_state=config.RANDOM_SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        Xtr, ytr = resampler(X[tr], y[tr], minority_label)
        clf = DecisionTreeClassifier(random_state=config.RANDOM_SEED).fit(Xtr, ytr)
        proba = clf.predict_proba(X[te])[:, list(clf.classes_).index(minority_label)]
        aucs.append(roc_auc_score(y[te] == minority_label, proba))
    return float(np.mean(aucs))

def mine(X, y, minority_label):
    # balance to 1:1 -> N chosen so minority roughly matches majority
    n_min = (y == minority_label).sum(); n_maj = (y != minority_label).sum()
    N = int(round((n_maj - n_min) / n_min * 100 / 100) * 100) or 100
    return smote_resample(X, y, minority_label, N=N, k=config.K_NEIGHBORS,
                          random_state=config.RANDOM_SEED)

def theirs(X, y, minority_label):
    Xr, yr = ImbSMOTE(k_neighbors=config.K_NEIGHBORS,
                      random_state=config.RANDOM_SEED).fit_resample(X, y)
    return Xr, yr

def main():
    a_mine = _cv_auc(mine)
    a_theirs = _cv_auc(theirs)
    print("Verification: my SMOTE vs imbalanced-learn (single-model 10-fold AUC)\n")
    print(f"  from-scratch SMOTE : {a_mine:.4f}")
    print(f"  imbalanced-learn   : {a_theirs:.4f}")
    print(f"  difference         : {abs(a_mine - a_theirs):.4f}")
    verdict = "PASS -- implementations agree" if abs(a_mine - a_theirs) < 0.03 \
              else "investigate -- larger gap than expected"
    print(f"\n  {verdict}")

if __name__ == "__main__":
    main()
