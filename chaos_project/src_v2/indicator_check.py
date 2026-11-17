"""Instability judged by the model itself, for every configuration.

1. Constant input, all 16,384 configurations, loads 52-90 (tau = 5): largest
   Lyapunov exponent of the deterministic map (twin trajectories).
2. Stochastic inputs, all configurations, the 16 ground-truth seeds, 400 steps:
   conditional Lyapunov exponent of the driven system, permutation entropy of q1
   (order 4) and, from the ground-truth table, the short-window indicator.
3. Kantz and Wolf-type estimates on q1 for a random sample of configurations.
4. Separation (Proposition 2) re-checked with the model-based unstable set.

Output: data_v2/model_lle_<regime>.npy (columns: mean exponent, fraction of seeds > 0,
mean permutation entropy) and data_v2/indicator_check.json.
"""
import os
import json
import itertools
import numpy as np
from scipy.stats import spearmanr
import fastnet as F
import nonlin as NL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
GT_SEEDS = list(range(900001, 900017))
ALL = np.arange(2 ** 14)
REGIMES = [("chaos", "orig"), ("poisson", "orig"), ("self_similar", "orig"), ("chaos", "zero_floor")]


def pe_rows(Q, order=4):
    """Permutation entropy of every column of Q (stable tie-breaking), vectorised."""
    n, m = Q.shape
    W = np.stack([Q[i:n - order + 1 + i] for i in range(order)], axis=2)  # (n-o+1, m, order)
    pat = np.argsort(W, axis=2, kind="stable")
    codes = (pat * (order ** np.arange(order))).sum(axis=2)
    out = np.zeros(m)
    import math
    norm = math.log(math.factorial(order))
    for j in range(m):
        _, c = np.unique(codes[:, j], return_counts=True)
        p = c / c.sum()
        out[j] = -(p * np.log(p)).sum() / norm
    return out


def neighbours():
    nb = [sum(1 << b for b in c) for d in (1, 2) for c in itertools.combinations(range(14), d)]
    return ALL[:, None] ^ np.array(nb)[None, :]


def main():
    out = {}
    # 1. constant input
    const = {}
    for lam in (52.0, 60.0, 70.0, 80.0, 90.0):
        a = np.full(8000, lam)
        x = np.full(8000, 15.0)
        lle = F.twin_lle(ALL, a, x, tau=5, burn=3000)
        const[str(int(lam))] = dict(frac_positive=float((lle > 0.01).mean()), n_positive=int((lle > 0.01).sum()),
                                    max=float(lle.max()), median=float(np.median(lle)))
        bits = (ALL[:, None] >> (13 - np.arange(14))) & 1
        k1 = 0.05 + 0.10 * (bits[:, 6] * 2 + bits[:, 7])
        pos = lle > 0.01
        const[f"{int(lam)}_slope_of_chaotic"] = {f"{s:.2f}": int((pos & np.isclose(k1, s)).sum()) for s in (0.05, 0.15, 0.25, 0.35)}
        print("constant", lam, const[str(int(lam))])
    out["constant_input_all_configs"] = const

    NB = neighbours()
    rng = np.random.default_rng(7)
    sample = np.sort(rng.choice(ALL, 400, replace=False))
    for mode, aqm in REGIMES:
        key = f"{mode}_{aqm}"
        L = np.zeros((len(GT_SEEDS), len(ALL)))
        PE = np.zeros_like(L)
        samp = {"kantz": [], "wolf": [], "model": [], "indicator": []}
        for k, s in enumerate(GT_SEEDS):
            a, x = F.inputs(mode, s, 400)
            L[k] = F.twin_lle(ALL, a, x, tau=5, zero_floor=(aqm == "zero_floor"), burn=50, seed=s)
            Q1, _ = F.simulate(ALL, a, x, tau=5, zero_floor=(aqm == "zero_floor"))
            PE[k] = pe_rows(Q1[50:])
            if k < 4:
                for j in sample:
                    q = Q1[50:, j]
                    if np.ptp(q) < 1e-9:
                        continue
                    samp["kantz"].append(NL.kantz(q, m=3, tau=2, theiler=5, kmax=8, kfit=(1, 4))[0])
                    samp["wolf"].append(NL.wolf(q, m=3, tau=2, evolve=2, theiler=5))
                    samp["model"].append(L[k, j])
                    samp["indicator"].append(F.NetworkSystem._estimate_lyapunov(None, q))
        tab = np.column_stack([L.mean(0), (L > 0).mean(0), PE.mean(0)])
        np.save(os.path.join(DATA, f"model_lle_{key}.npy"), tab)
        g = np.load(os.path.join(DATA, f"gt_{key}.npy"))
        flagged = g[:, 3] > 0.5
        unstable = tab[:, 1] > 0.5
        r = dict(model_lle_mean=float(L.mean()), model_lle_q99=float(np.quantile(L.mean(0), 0.99)),
                 model_lle_max=float(L.mean(0).max()), n_model_unstable=int(unstable.sum()),
                 frac_seed_runs_positive=float((L > 0).mean()),
                 n_flagged=int(flagged.sum()),
                 spearman_indicator_vs_model=float(spearmanr(g[:, 2], tab[:, 0])[0]),
                 spearman_indicator_vs_PE=float(spearmanr(g[:, 2], tab[:, 2])[0]),
                 PE_median=float(np.median(tab[:, 2])), PE_q05=float(np.quantile(tab[:, 2], 0.05)),
                 PE_median_flagged=float(np.median(tab[flagged, 2])),
                 PE_median_unflagged=float(np.median(tab[~flagged, 2])) if (~flagged).any() else None)
        sv = {kk: np.array(v, float) for kk, v in samp.items()}
        ok = np.isfinite(sv["kantz"]) & np.isfinite(sv["wolf"])
        r["sample"] = dict(n=int(ok.sum()),
                           frac_indicator_gt_005=float((sv["indicator"][ok] > 0.05).mean()),
                           frac_kantz_gt_0=float((sv["kantz"][ok] > 0).mean()),
                           frac_wolf_gt_0=float((sv["wolf"][ok] > 0).mean()),
                           frac_model_gt_0=float((sv["model"][ok] > 0).mean()),
                           median_kantz=float(np.median(sv["kantz"][ok])), median_wolf=float(np.median(sv["wolf"][ok])),
                           median_model=float(np.median(sv["model"][ok])),
                           spearman_kantz_indicator=float(spearmanr(sv["kantz"][ok], sv["indicator"][ok])[0]),
                           spearman_wolf_indicator=float(spearmanr(sv["wolf"][ok], sv["indicator"][ok])[0]))
        for col, name in [(1, "pen"), (0, "perf")]:
            U = g[:, col]
            locopt = U >= U[NB].max(1)
            better_stable = ((U[NB] > U[:, None]) & (~unstable[NB])).any(1)
            r[f"{name}_local_opt"] = int(locopt.sum())
            r[f"{name}_local_opt_model_unstable"] = int((locopt & unstable).sum())
            r[f"{name}_local_opt_model_lle"] = [float(v) for v in tab[locopt, 0]]
            r[f"{name}_sep_violations_model"] = int((unstable & ~better_stable).sum())
        out[key] = r
        print(key, json.dumps({k: v for k, v in r.items() if k != "sample"}), r["sample"])
    json.dump(out, open(os.path.join(DATA, "indicator_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
