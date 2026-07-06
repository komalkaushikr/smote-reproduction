"""Stage 6 (Extend): run the SAME pipeline on Mammography -- the paper's severe
42:1 imbalance case -- with no changes to SMOTE, the model, or the evaluator.
Only the dataset changes. This is the payoff of the (X, y, minority_label)
interface: one new loader, everything else reused.

Paper AUCs for Mammography (Table 3, C4.5, /10000):
    Under = 0.9260,  400% SMOTE = 0.9304  (best is 0.9330 at 300%)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from data_loader import load_mammography, class_balance
from evaluate import roc_sweep_auc

PAPER = {"under": 0.9260, "smote_400": 0.9304}

def main():
    print("Loading Mammography (first run downloads it, ~1 min)...")
    X, y, minority_label = load_mammography()
    bal = class_balance(y)
    imb = bal[0] / bal[1]
    print(f"  shape {X.shape}, balance {bal}  (~{imb:.0f}:1 imbalance)\n")

    print("Plain under-sampling curve...")
    auc_under, _ = roc_sweep_auc(
        X, y, minority_label, smote_percent=0,
        undersample_levels=config.UNDERSAMPLE_LEVELS,
        k=config.K_NEIGHBORS, cv_folds=config.CV_FOLDS,
        random_state=config.RANDOM_SEED)
    print(f"  Under AUC = {auc_under:.4f}")

    print(f"SMOTE({config.SMOTE_PERCENT_MAMMO}%) + under-sampling curve...")
    auc_smote, _ = roc_sweep_auc(
        X, y, minority_label, smote_percent=config.SMOTE_PERCENT_MAMMO,
        undersample_levels=config.UNDERSAMPLE_LEVELS,
        k=config.K_NEIGHBORS, cv_folds=config.CV_FOLDS,
        random_state=config.RANDOM_SEED)
    print(f"  SMOTE AUC = {auc_smote:.4f}\n")

    print(f"{'method':<22}{'paper':>10}{'mine':>10}{'diff':>10}")
    print("-" * 52)
    print(f"{'plain under-sampling':<22}{PAPER['under']:>10.4f}"
          f"{auc_under:>10.4f}{auc_under-PAPER['under']:>+10.4f}")
    print(f"{'SMOTE(400%)+under':<22}{PAPER['smote_400']:>10.4f}"
          f"{auc_smote:>10.4f}{auc_smote-PAPER['smote_400']:>+10.4f}")
    print("-" * 52)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "mammography_results.json").write_text(json.dumps(
        {"dataset": "Mammography", "imbalance": f"{imb:.0f}:1",
         "under_auc": auc_under, "smote_auc": auc_smote,
         "smote_percent": config.SMOTE_PERCENT_MAMMO}, indent=2))
    print(f"\nSaved -> {config.RESULTS_DIR / 'mammography_results.json'}")

if __name__ == "__main__":
    main()
