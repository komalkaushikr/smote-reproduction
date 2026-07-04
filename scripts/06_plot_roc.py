import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

res = json.loads((config.RESULTS_DIR / "pima_results.json").read_text())
def curve(points):
    pts = sorted(points); pts = [(0,0)] + pts + [(100,100)]
    return [p[0] for p in pts], [p[1] for p in pts]

fu_x, fu_y = curve(res["under_points"])
fs_x, fs_y = curve(res["smote_points"])
plt.figure(figsize=(6,6))
plt.plot(fu_x, fu_y, "o-", label=f"Under (AUC={res['under_auc']:.4f})", color="#c44")
plt.plot(fs_x, fs_y, "s-", label=f"100% SMOTE+Under (AUC={res['smote_auc']:.4f})", color="#268")
plt.plot([0,100],[0,100], "--", color="gray", label="random")
plt.xlabel("% False Positive"); plt.ylabel("% True Positive")
plt.title("Pima Indian Diabetes — SMOTE reproduction\n(protocol from Chawla et al. 2002, Fig. 9)")
plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(config.RESULTS_DIR / "pima_roc.png", dpi=110)
print("saved", config.RESULTS_DIR / "pima_roc.png")
