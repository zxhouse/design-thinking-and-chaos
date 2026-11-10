"""Schematic of the simulated system (matches network_sim.py)."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
plt.rcParams.update({"font.family": "serif", "font.size": 9.5})
fig, ax = plt.subplots(figsize=(8.6, 3.3)); ax.set_xlim(0, 10); ax.set_ylim(-0.35, 4.1); ax.axis("off")
def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#334155", lw=1))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center")
def arr(a, b, ls="-", col="#334155", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=col, ls=ls, connectionstyle=f"arc3,rad={rad}"))
box(0.2, 1.2, 1.9, 1.0, "Sender\nmean load $\\lambda_0$\n(3 traffic types)", "#F1F5F9")
box(3.0, 1.2, 1.9, 1.0, "Edge switch $S_1$\n$\\mu_1,\\theta_1,\\kappa_1$\n$B_1=120$", "#DBEAFE")
box(5.8, 1.2, 1.9, 1.0, "Core switch $S_2$\n$\\mu_2,\\theta_2$\n$B_2=140$", "#DBEAFE")
box(8.3, 1.4, 1.5, 0.6, "Output", "#F1F5F9")
box(5.8, 0.0, 1.9, 0.6, "Cross-traffic, mean 15", "#F1F5F9")
box(2.6, 3.0, 5.5, 0.85, "Iterate–test controller\nsets $p\\in\\{0,1\\}^{14}$, observes one score $Y$ per 250-step test", "#EDE9FE")
arr((2.1, 1.7), (3.0, 1.7)); arr((4.9, 1.7), (5.8, 1.7)); arr((7.7, 1.7), (8.3, 1.7)); arr((6.75, 0.6), (6.75, 1.2))
arr((3.95, 1.2), (1.35, 1.2), ls="--", col="#DC2626", rad=-0.35)
arr((6.3, 1.2), (0.95, 1.2), ls="--", col="#DC2626", rad=-0.30)
ax.text(2.6, -0.25, "loss at $S_1$ and $S_2$, fed back after $\\tau=5$ steps", color="#DC2626", ha="center", fontsize=8.5)
arr((4.2, 3.0), (3.95, 2.2), col="#7C3AED"); arr((6.5, 3.0), (6.75, 2.2), col="#7C3AED")
ax.text(5.35, 2.6, "configuration $p$", color="#7C3AED", ha="center", fontsize=8.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_schema.pdf"), bbox_inches="tight", pad_inches=0.05); plt.savefig(os.path.join(OUT, "fig_schema.png"), dpi=200, bbox_inches="tight", pad_inches=0.05)
