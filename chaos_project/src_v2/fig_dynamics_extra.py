"""Figures for the extended dynamics analysis: overload attractors, indicators, long memory."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
OUT = os.path.join(HERE, "..", "..", "results")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = dict(blue="#2563EB", red="#DC2626", violet="#7C3AED", grey="#64748B", green="#059669", amber="#D97706")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------- overload attractors
o = np.load(os.path.join(DATA, "overload_orbits.npz"))
od = json.load(open(os.path.join(DATA, "overload_dynamics.json")))
lams = o["lams"]
P1, P2 = 3318, 972
fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.0))
q = o[f"cfg{P1}_q"]
ax[0, 0].plot(np.repeat(lams, q.shape[1]), q.ravel(), ",", color="k", alpha=0.35, rasterized=True)
ax[0, 0].set_xlim(60, 100)
ax[0, 0].set_xlabel("Constant offered load $\\lambda_0$")
ax[0, 0].set_ylabel("$q_1$ (last 300 steps)")
ax[0, 0].set_title(f"(a) Orbit diagram, configuration {P1}")
l = o[f"cfg{P1}_lle"]
ax[1, 0].plot(lams, l, color=C["violet"], lw=0.8)
ax[1, 0].axhline(0, color=C["grey"], lw=0.8, ls=":")
ax[1, 0].set_xlim(60, 100)
ax[1, 0].set_ylim(-0.3, 0.3)
ax[1, 0].set_xlabel("Constant offered load $\\lambda_0$")
ax[1, 0].set_ylabel("Model exponent")
ax[1, 0].set_title("(b) Model exponent along the same sweep")
s = o[f"cfg{P2}_series90"]
ax[0, 1].plot(s[:-1], s[1:], ".", ms=1.2, color=C["blue"], rasterized=True)
ax[0, 1].set_xlabel("$q_1(t)$")
ax[0, 1].set_ylabel("$q_1(t+1)$")
ax[0, 1].set_title(f"(c) Return map, configuration {P2}, $\\lambda_0=90$")
ax[1, 1].plot(s[:-5], s[5:], ".", ms=1.2, color=C["red"], rasterized=True)
ax[1, 1].set_xlabel("$q_1(t)$")
ax[1, 1].set_ylabel("$q_1(t+5)$")
ax[1, 1].set_title("(d) Same series, lag $\\tau=5$")
fig.tight_layout()
save(fig, "fig_overload")

# ---------------------------------------------------------------- indicators vs model
ic = json.load(open(os.path.join(DATA, "indicator_check.json")))
regs = [("poisson_orig", "Poisson"), ("self_similar_orig", "ON/OFF"), ("chaos_orig", "Chaotic")]
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
w = 0.2
labels = [("frac_indicator_gt_005", "Short-window indicator $>0.05$", C["grey"]),
          ("frac_kantz_gt_0", "Kantz $>0$", C["blue"]),
          ("frac_wolf_gt_0", "Wolf-type $>0$", C["amber"]),
          ("frac_model_gt_0", "Model exponent $>0$", C["red"])]
for k, (key, lab, col) in enumerate(labels):
    vals = [100 * ic[r]["sample"][key] for r, _ in regs]
    ax[0].bar(np.arange(3) + (k - 1.5) * w, vals, w, color=col, label=lab)
ax[0].set_xticks(range(3))
ax[0].set_xticklabels([n for _, n in regs])
ax[0].set_ylabel("% of test-length runs")
ax[0].set_title("(a) Positive exponents at the operating load")
ax[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
g = np.load(os.path.join(DATA, "gt_chaos_orig.npy"))
t = np.load(os.path.join(DATA, "model_lle_chaos_orig.npy"))
hb = ax[1].hexbin(t[:, 2], g[:, 2], gridsize=40, mincnt=1, cmap="viridis", bins="log")
ax[1].axhline(0.05, color=C["red"], lw=0.8, ls="--")
ax[1].set_xlabel("Permutation entropy of $q_1$ (order 4)")
ax[1].set_ylabel("Mean short-window indicator")
ax[1].set_title("(b) Chaotic forcing: indicator tracks entropy")
fig.colorbar(hb, ax=ax[1], label="configurations")
fig.tight_layout()
save(fig, "fig_indicators")

# ---------------------------------------------------------------- long memory
hc = np.load(os.path.join(DATA, "hurst_curves.npz"))
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
for key, lab, col in [("onoff_arrivals_0", "ON/OFF arrivals", C["red"]),
                      ("poisson_arrivals", "Poisson arrivals", C["grey"]),
                      ("onoff_queue_q1", "Edge queue under ON/OFF", C["blue"])]:
    s, Fs = hc[f"{key}_dfa_s"], hc[f"{key}_dfa_F"]
    ax[0].loglog(s, Fs / Fs[0], "o-", ms=2.5, lw=1, color=col, label=lab)
    j, y = hc[f"{key}_wav_j"], hc[f"{key}_wav_y"]
    ax[1].plot(j, y - y[0], "o-", ms=2.5, lw=1, color=col, label=lab)
ss = np.array([8, 512])
ax[0].loglog(ss, (ss / 8.0) ** 0.82, color="k", lw=0.8, ls="--", label="slope 0.82")
ax[0].loglog(ss, (ss / 8.0) ** 0.5, color="k", lw=0.8, ls=":", label="slope 0.5")
for a in ax:
    pass
ax[0].axvspan(868, 2 ** 15, color=C["amber"], alpha=0.12, lw=0)
ax[0].text(900, 1.3, "beyond the\nlongest period", fontsize=7, color=C["amber"])
ax[0].set_xlabel("Scale $s$ (steps)")
ax[0].set_ylabel("DFA-1 fluctuation $F(s)/F(8)$")
ax[0].set_title("(a) Detrended fluctuation analysis")
ax[0].legend(fontsize=7, frameon=False)
ax[1].axvspan(np.log2(868), 15, color=C["amber"], alpha=0.12, lw=0)
ax[1].set_xlabel("Octave $j$ (scale $2^j$ steps)")
ax[1].set_ylabel("$\\log_2$ wavelet energy (relative)")
ax[1].set_title("(b) Wavelet log-scale diagram (Daubechies-3)")
fig.tight_layout()
save(fig, "fig_hurst")
print("figures written")
