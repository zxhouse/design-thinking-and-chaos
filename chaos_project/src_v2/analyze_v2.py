"""Tables and figures for the v2 manuscript, computed only from data_v2/."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
OUT = os.path.join(HERE, "..", "..", "results")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"blue": "#2563EB", "red": "#DC2626", "violet": "#7C3AED", "grey": "#64748B", "amber": "#D97706"}


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def load(name):
    with open(os.path.join(DATA, name + ".json")) as f:
        return json.load(f)


summary = {}

# ---------------- ground truth ----------------
gt = {k: np.load(os.path.join(DATA, f"gt_{k}.npy")) for k in
      ["chaos_orig", "poisson_orig", "self_similar_orig", "chaos_zero_floor"]}
for k, g in gt.items():
    perf, pen, lle, chaos_frac, sd = g.T
    summary[f"gt_{k}"] = {
        "opt_pen": float(pen.max()), "opt_perf": float(perf.max()),
        "frac_lle_pos": float((lle > 0.05).mean()),
        "mean_lle": float(lle.mean()),
        "median_test_sd_perf": float(np.median(sd)),
        "n_within_1pct_pen": int((pen >= 0.99 * pen.max()).sum()),
        "corr_perf_lle": float(np.corrcoef(perf, lle)[0, 1]),
        "argmax_pen_lle": float(lle[pen.argmax()]),
        "argmax_perf_lle": float(lle[perf.argmax()]),
    }

# ---------------- dynamics ----------------
dyn = load("dynamics")
iq = np.array(dyn["input_vs_queue"])
summary["dyn_input_lle"] = ci(iq[:, 0])
summary["dyn_queue_lle"] = ci(iq[:, 1])
summary["dyn_corr"] = float(np.corrcoef(iq[:, 0], iq[:, 1])[0, 1])
sw = dyn["sweep"]
for inp in ["constant", "poisson"]:
    rows = [r for r in sw if r["input"] == inp]
    summary[f"sweep_{inp}_max_lle"] = max(r["lle"] for r in rows)
    summary[f"sweep_{inp}_frac_pos"] = float(np.mean([r["lle"] > 0.05 for r in rows]))
    summary[f"sweep_{inp}_max_q1_range"] = max(r["q1_range"] for r in rows)


# ---------------- controller grids ----------------
def grid(name):
    d = load(name)
    H = d["horizons"]
    res = d["results"]
    tab = {}
    for lab, runs in res.items():
        tab[lab] = {}
        for key in ["shipped", "best_found", "overest", "fav_lle", "fav_chaotic_frac"]:
            arr = np.array([r[key] for r in runs]) * (100 if key in ("shipped", "best_found", "overest") else 1)
            tab[lab][key] = [ci(arr[:, j]) for j in range(arr.shape[1])]
        tab[lab]["_raw_shipped"] = np.array([r["shipped"] for r in runs]) * 100
    return H, tab


def paired(tab, a, b, j):
    d = tab[a]["_raw_shipped"][:, j] - tab[b]["_raw_shipped"][:, j]
    return ci(d)


H, alloc = grid("allocation_v2")
_, reg = grid("regimes_v2")
_, abl = grid("ablations_v2")
_, aq = grid("aqm_v2")

for obj in ["pen", "perf"]:
    for j, h in enumerate(H):
        for n in [1, 2, 3, 4, 5]:
            m = alloc[f"{obj}|new{n}"]
            summary[f"alloc_{obj}_new{n}_h{h}"] = {
                "shipped": m["shipped"][j], "best": m["best_found"][j], "overest": m["overest"][j],
                "fav_lle": m["fav_lle"][j], "fav_chaotic": m["fav_chaotic_frac"][j],
                "diff_vs_new5": paired(alloc, f"{obj}|new{n}", f"{obj}|new5", j)}
        m = alloc[f"{obj}|random"]
        summary[f"alloc_{obj}_random_h{h}"] = {"shipped": m["shipped"][j], "overest": m["overest"][j],
                                              "fav_chaotic": m["fav_chaotic_frac"][j],
                                              "diff_new1_vs_random": paired(alloc, f"{obj}|new1", f"{obj}|random", j)}
for lab in reg:
    for j, h in enumerate(H):
        summary[f"reg_{lab}_h{h}"] = {k: reg[lab][k][j] for k in ["shipped", "best_found", "overest"]}
for mode in ["poisson", "self_similar"]:
    for j, h in enumerate(H):
        summary[f"reg_{mode}_diff_h{h}"] = paired(reg, f"{mode}|new1", f"{mode}|new5", j)
for lab in abl:
    summary[f"abl_{lab}"] = {"shipped": abl[lab]["shipped"][-1],
                             "diff_vs_base": paired(abl, lab, "baseline", -1) if lab != "baseline" else None}
for lab in aq:
    for j, h in enumerate(H):
        summary[f"aqm_{lab}_h{h}"] = {k: aq[lab][k][j] for k in ["shipped", "best_found", "overest", "fav_chaotic_frac"]}
for j, h in enumerate(H):
    summary[f"aqm_diff_h{h}"] = paired(aq, "zero_floor|new1", "zero_floor|new5", j)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    return o


with open(os.path.join(OUT, "summary_v2.json"), "w") as f:
    json.dump(_clean(summary), f, indent=1)

# ---------------- Figure: landscape ----------------
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))
g = gt["chaos_orig"]
ax[0].hist(g[:, 1] / g[:, 1].max() * 100, bins=60, color=C["red"], alpha=.75)
ax[0].set_xlabel("True utility, % of global optimum")
ax[0].set_ylabel("Configurations")
ax[0].set_title("(a) Utility of all 16,384 configurations")
ax[1].scatter(g[:, 2], g[:, 0], s=1.5, alpha=.25, color=C["grey"], edgecolors="none")
ax[1].axvline(0.05, color=C["red"], ls="--", lw=1)
ax[1].set_xlabel("Mean Lyapunov estimate of $q_1$")
ax[1].set_ylabel("Performance (no penalty)")
ax[1].set_title("(b) Performance vs. instability")
ax[2].hist(g[:, 4], bins=60, color=C["blue"], alpha=.75)
ax[2].set_xlabel("SD of one 250-step test")
ax[2].set_title("(c) Noise of a single test")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_landscape.pdf"))
plt.savefig(os.path.join(OUT, "fig_landscape.png"), dpi=200)
plt.close()

# ---------------- Figure: dynamics ----------------
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
ax[0].scatter(iq[:, 0], iq[:, 1], s=14, color=C["red"])
lim = [min(iq.min(), 0), iq.max() * 1.05]
ax[0].plot(lim, lim, color=C["grey"], lw=.8, ls=":")
ax[0].set_xlabel("Lyapunov estimate of the arrival series")
ax[0].set_ylabel("Lyapunov estimate of queue $q_1$")
ax[0].set_title("(a) Input vs. queue, 40 seeds")
for inp, col in [("constant", C["blue"]), ("poisson", C["violet"])]:
    for tau, ls in [(1, ":"), (5, "-"), (40, "--")]:
        rows = [r for r in sw if r["input"] == inp and r["tau"] == tau]
        ax[1].plot([r["load"] for r in rows], [r["lle"] for r in rows], color=col, ls=ls, lw=1.4,
                   label=f"{inp}, $\\tau$={tau}")
ax[1].axhline(0.05, color=C["red"], lw=.8, ls="--")
ax[1].set_xlabel("Offered load $\\lambda_0$ (packets/step)")
ax[1].set_ylabel("Lyapunov estimate of $q_1$")
ax[1].set_title("(b) One configuration, two inputs")
ax[1].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
osc = json.load(open(os.path.join(DATA, "round2_osc.json")))
for tau, ls in [(5, "-"), (40, "--")]:
    loads = sorted({int(k.split("_lam")[1]) for k in osc})
    ax[2].plot(loads, [100 * osc[f"tau{tau}_lam{l}"]["frac_osc"] for l in loads], ls, color=C["blue"], marker="o", ms=3, label=f"oscillating, $\\tau$={tau}")
    ax[2].plot(loads, [100 * osc[f"tau{tau}_lam{l}"]["frac_irregular"] for l in loads], ls, color=C["red"], marker="s", ms=3, label=f"irregular, $\\tau$={tau}")
ax[2].axvline(52, color=C["grey"], lw=.8, ls=":")
ax[2].text(53, 14, "operating\nload", fontsize=7.5, color=C["grey"])
ax[2].set_xlabel("Constant offered load $\\lambda_0$")
ax[2].set_ylabel("% of 200 random configurations")
ax[2].set_title("(c) Oscillation needs overload")
ax[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_dynamics.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.savefig(os.path.join(OUT, "fig_dynamics.png"), dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close()

# ---------------- Figure: allocation ----------------
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.3))
ns = [1, 2, 3, 4, 5]
for j, (h, col) in enumerate(zip(H, ["#94A3B8", "#64748B", "#334155", "#0F172A"])):
    m = [alloc[f"pen|new{n}"]["shipped"][j] for n in ns]
    ax[0].errorbar(ns, [x[0] for x in m], yerr=[x[1] for x in m], fmt="-o", color=col, capsize=3, ms=4,
                   label=f"{h} rounds")
    r = alloc["pen|random"]["shipped"][j]
    ax[0].plot([5.6], [r[0]], marker="x", color=col)
ax[0].set_xticks(ns + [5.6])
ax[0].set_xticklabels(["1", "2", "3", "4", "5", "rand."])
ax[0].set_xlabel("New configurations per round (of 5 tests)")
ax[0].set_ylabel("Shipped utility, % of optimum")
ax[0].set_title("(a) Budget allocation")
ax[0].set_ylim(87.5, 98.8)
ax[0].legend(fontsize=7, ncol=2, loc="upper center", frameon=False)
j = H.index(50)
for n, col in zip(ns, ["#0F172A", "#334155", "#64748B", "#94A3B8", C["red"]]):
    x = np.arange(len(H))
    ax[1].plot(H, [alloc[f"pen|new{n}"]["overest"][k][0] for k in range(len(H))], "-o", ms=3, color=col,
               label=f"{n} new")
ax[1].axhline(0, color=C["grey"], lw=.8)
ax[1].set_xscale("log")
ax[1].set_xticks(H)
ax[1].set_xticklabels(H)
ax[1].set_xlabel("Round")
ax[1].set_ylabel("Belief minus truth, % of optimum")
ax[1].set_title("(b) Optimizer's curse")
ax[1].legend(fontsize=7)
for n, col in zip(ns, ["#0F172A", "#334155", "#64748B", "#94A3B8", C["red"]]):
    gap = [alloc[f"pen|new{n}"]["best_found"][k][0] - alloc[f"pen|new{n}"]["shipped"][k][0] for k in range(len(H))]
    ax[2].plot(H, gap, "-o", ms=3, color=col, label=f"{n} new")
ax[2].set_xscale("log")
ax[2].set_xticks(H)
ax[2].set_xticklabels(H)
ax[2].set_xlabel("Round")
ax[2].set_ylabel("Best tested minus shipped (pp)")
ax[2].set_title("(c) Choice gap")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_allocation.pdf"))
plt.savefig(os.path.join(OUT, "fig_allocation.png"), dpi=200)
plt.close()
print(json.dumps(_clean({k: v for k, v in summary.items() if "h50" in k or k.startswith(("gt_", "dyn", "sweep", "abl"))}), indent=1))
