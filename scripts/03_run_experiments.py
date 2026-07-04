"""Stage 3: run the harness. Plain under-sampling vs SMOTE+under-sampling,
using the paper's ROC-sweep protocol. Saves AUCs + ROC points for later stages."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from data_loader import load_pima
from evaluate import roc_sweep_auc

def main():
    X, y, minority_label = load_pima(config.PIMA_CSV)

    print("Running PLAIN under-sampling curve (this is the 'Under' baseline)...")
    auc_under, pts_under = roc_sweep_auc(
        X, y, minority_label, smote_percent=0,
        undersample_levels=config.UNDERSAMPLE_LEVELS,
        k=config.K_NEIGHBORS, cv_folds=config.CV_FOLDS,
        random_state=config.RANDOM_SEED)
    print(f"  Under AUC = {auc_under:.4f}")

    print(f"Running SMOTE({config.SMOTE_PERCENT_PIMA}%) + under-sampling curve...")
    auc_smote, pts_smote = roc_sweep_auc(
        X, y, minority_label, smote_percent=config.SMOTE_PERCENT_PIMA,
        undersample_levels=config.UNDERSAMPLE_LEVELS,
        k=config.K_NEIGHBORS, cv_folds=config.CV_FOLDS,
        random_state=config.RANDOM_SEED)
    print(f"  SMOTE AUC = {auc_smote:.4f}")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "dataset": "Pima",
        "under_auc": auc_under, "under_points": pts_under,
        "smote_auc": auc_smote, "smote_points": pts_smote,
        "smote_percent": config.SMOTE_PERCENT_PIMA,
    }
    (config.RESULTS_DIR / "pima_results.json").write_text(json.dumps(out, indent=2))
    print(f"Saved -> {config.RESULTS_DIR / 'pima_results.json'}")

if __name__ == "__main__":
    main()
