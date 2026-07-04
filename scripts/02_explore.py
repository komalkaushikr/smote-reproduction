"""Stage 1: the 'before' picture -- confirm the imbalance SMOTE is meant to fix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from data_loader import load_pima, class_balance

def main():
    X, y, minority_label = load_pima(config.PIMA_CSV)
    bal = class_balance(y)
    total = len(y)
    print(f"Pima: {total} samples, {X.shape[1]} numeric features")
    print(f"Class balance: {bal}")
    for label, count in bal.items():
        tag = "MINORITY" if label == minority_label else "majority"
        print(f"  class {label} ({tag}): {count}  ({100*count/total:.1f}%)")
    imb = bal[1 - minority_label] / bal[minority_label]
    print(f"Imbalance ratio (majority:minority) = {imb:.2f} : 1")
    print("=> imbalanced enough to be worth resampling, but the LEAST skewed "
          "dataset in the paper -- so expect only a small SMOTE lift here.")

if __name__ == "__main__":
    main()
