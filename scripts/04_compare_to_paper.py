"""Stage 4: put my numbers next to the paper's (Table 3, C4.5 base classifier).

Paper's Pima AUCs (÷10000): Under = 0.7242, 100% SMOTE = 0.7307.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

PAPER = {"under": 0.7242, "smote_100": 0.7307}  # Table 3, Pima row

def main():
    res = json.loads((config.RESULTS_DIR / "pima_results.json").read_text())
    mine_under, mine_smote = res["under_auc"], res["smote_auc"]

    print("Pima Indian Diabetes -- AUC reproduction (base classifier: decision tree)\n")
    print(f"{'method':<22}{'paper':>10}{'mine':>10}{'diff':>10}")
    print("-" * 52)
    print(f"{'plain under-sampling':<22}{PAPER['under']:>10.4f}"
          f"{mine_under:>10.4f}{mine_under-PAPER['under']:>+10.4f}")
    print(f"{'SMOTE(100%)+under':<22}{PAPER['smote_100']:>10.4f}"
          f"{mine_smote:>10.4f}{mine_smote-PAPER['smote_100']:>+10.4f}")
    print("-" * 52)

    paper_lift = PAPER["smote_100"] - PAPER["under"]
    mine_lift = mine_smote - mine_under
    print(f"\nSMOTE lift over baseline:  paper = {paper_lift:+.4f},  "
          f"mine = {mine_lift:+.4f}")
    print("\nRead this honestly: the goal is to reproduce the TREND and the")
    print("methodology, not the exact 4th decimal. The paper used C4.5;")
    print("scikit-learn's DecisionTree is CART (different splitting & pruning),")
    print("so absolute AUCs differ. On Pima the paper itself reports SMOTE")
    print("barely helps (least-skewed dataset) -- a small/near-flat lift here")
    print("is the correct finding, not a bug.")

if __name__ == "__main__":
    main()
