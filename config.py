"""Central configuration. One place to change seeds, paths, and hyperparameters
so every script behaves identically and results are reproducible."""

from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PIMA_CSV = DATA_DIR / "pima.csv"

# ---- reproducibility ----
RANDOM_SEED = 42          # fixed seed everywhere -> same synthetic points every run

# ---- SMOTE hyperparameters (paper uses k=5) ----
K_NEIGHBORS = 5

# ---- experiment protocol (from the paper, Section 5.2) ----
# For Pima the paper's ROC curve uses 100% SMOTE (Figure 9: "100 SMOTE-C4.5").
SMOTE_PERCENT_PIMA = 100

# Majority under-sampling levels swept to trace out the ROC curve (Section 5.2).
UNDERSAMPLE_LEVELS = [10, 15, 25, 50, 75, 100, 125, 150, 175,
                      200, 300, 400, 500, 600, 700, 800, 1000, 2000]

CV_FOLDS = 10             # paper averages %TP/%FP over 10-fold cross-validation
