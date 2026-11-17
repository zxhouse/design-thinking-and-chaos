"""Theorem 2: pathwise upper bound on the model exponent for any input, checked.

Norm: ||(q1, q2, L)|| = max(|q1|/w1, |q2|/w2, max_k rho^k |L_k|), L_k = loss of age k.
Idle step (both queues empty before and after, with slack): one-step Lipschitz
constant rho * max(1, c1 w1 + c2 w2 + s sigma). Any other step: K(a_t, x_t).
Bound: Lambda <= (1/T) sum_t ln k_t, minimised over a grid of (rho, w1, w2).

Output: data_v2/stability_bound.json, data_v2/stability_bound_<regime>.npy
(columns: mean bound, fraction of seeds certified, mean idle fraction, mean model exponent)
"""
import os
import json
import itertools
import numpy as np
import fastnet as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
ALL = np.arange(2 ** 14)
TAU = 5
GT_SEEDS = list(range(900001, 900017))
GRID = [(r, w1, w1 * c) for r in (0.45, 0.55, 0.65, 0.75, 0.85) for w1 in (5.0, 10.0, 20.0, 40.0, 80.0) for c in (1.0, 2.0, 4.0)]


def sig(q, th, sl):
    return 1.0 / (1.0 + np.exp(-np.clip(sl * (q - th), -30.0, 30.0)))


def constants(par):
    mu1, th1, sl1, mu2, th2, sl2 = par
    p1, p2 = sig(0.0, th1, sl1), sig(0.0, th2, sl2)
    c1 = sl1 * p1 * (1 - p1) + p2
    c2 = sl2 * p2 * (1 - p2)
    s = 4.5 * (p1 + (1 - p1) * p2)
    return p1, p2, c1, c2, s


def idle_flags(a, x, par):
    """(steps, n) boolean: both queues empty before the step and strictly below capacity."""
    mu1, th1, sl1, mu2, th2, sl2 = par
    p1, p2 = sig(0.0, th1, sl1), sig(0.0, th2, sl2)
    b = F.Batch(ALL, TAU)
    out = np.zeros((len(a), len(ALL)), bool)
    for t in range(len(a)):
        a1 = a[t] * np.maximum(0.4, 1.0 - 1.8 * b.L.mean(axis=1))
        dep1 = a1 * (1 - p1)
        out[t] = (b.q1 == 0) & (b.q2 == 0) & (dep1 < mu1 - 1e-9) & ((dep1 + x[t]) * (1 - p2) < mu2 - 1e-9)
        b.step(a[t], x[t])
    return out


def bound(a, x, idle, par, burn):
    mu1, th1, sl1, mu2, th2, sl2 = par
    _, _, c1, c2, s = constants(par)
    a, x, idle = a[burn:], x[burn:], idle[burn:]
    nidle = idle.sum(axis=0)
    T = len(a)
    best = np.full(len(ALL), np.inf)
    M = np.maximum(1.0, 0.4 * a + x)[:, None]
    aa = a[:, None]
    al1 = np.maximum(1.0, aa * sl1[None, :] / 4 - 1)
    be1 = np.maximum(1.0, aa * sl1[None, :] / 4)
    a2 = mu1[None, :] + x[:, None]
    al2 = np.maximum(1.0, a2 * sl2[None, :] / 4 - 1)
    be2 = np.maximum(1.0, a2 * sl2[None, :] / 4)
    for rho, w1, w2 in GRID:
        sigma = np.mean(rho ** -np.arange(1, TAU + 1))
        kid = rho * np.maximum(1.0, c1 * w1 + c2 * w2 + s * sigma)
        g = 1.8 * aa * sigma
        K = np.maximum.reduce([al1 + g / w1, al2 + (al1 * w1 + g) / w2,
                               np.full_like(al1, rho), rho * (be1 * w1 + be2 * w2 + g) / M])
        lnK = np.where(idle, 0.0, np.log(K)).sum(axis=0)
        val = (nidle * np.log(kid) + lnK) / T
        best = np.minimum(best, val)
    return best, nidle / T


def main():
    par = F.params(ALL)
    out = {"grid": len(GRID)}
    # constant input at the operating load and above
    for lam in (52.0, 60.0):
        a = np.full(2000, lam); x = np.full(2000, 15.0)
        idle = idle_flags(a, x, par)
        bd, fi = bound(a, x, idle, par, 500)
        lle = np.load(os.path.join(DATA, f"stability_fixed_{int(lam)}.npy"))[:, 6]
        out[f"constant_{int(lam)}"] = dict(n_certified=int((bd < 0).sum()), n_always_idle=int((fi == 1).sum()),
                                           violations=int((lle > bd + 1e-6).sum()),
                                           median_bound=float(np.median(bd)), median_lle=float(np.median(lle)))
        print("constant", lam, out[f"constant_{int(lam)}"])
    for mode in ("poisson", "self_similar", "chaos"):
        BD = np.zeros((len(GT_SEEDS), len(ALL))); FI = np.zeros_like(BD); L = np.zeros_like(BD)
        for k, sd in enumerate(GT_SEEDS):
            a, x = F.inputs(mode, sd, 400)
            idle = idle_flags(a, x, par)
            BD[k], FI[k] = bound(a, x, idle, par, 50)
            L[k] = F.twin_lle(ALL, a, x, tau=TAU, burn=50, seed=sd)
        tab = np.column_stack([BD.mean(0), (BD < 0).mean(0), FI.mean(0), L.mean(0)])
        np.save(os.path.join(DATA, f"stability_bound_{mode}_orig.npy"), tab)
        cert = BD < 0
        sl1 = par[2]
        r = dict(violations=int((L > BD + 1e-6).sum()), runs=int(BD.size),
                 frac_runs_certified=float(cert.mean()),
                 n_configs_certified_all_seeds=int(cert.all(0).sum()),
                 n_configs_certified_majority=int((cert.mean(0) > 0.5).sum()),
                 frac_runs_lle_negative=float((L < 0).mean()),
                 median_idle_fraction=float(np.median(FI)),
                 idle_fraction_certified_min=float(FI[cert].min()) if cert.any() else None,
                 idle_fraction_uncertified_max=float(FI[~cert].max()) if (~cert).any() else None,
                 certified_by_slope={f"{s:.2f}": float(cert[:, np.isclose(sl1, s)].mean()) for s in (0.05, 0.15, 0.25, 0.35)},
                 median_bound=float(np.median(BD)), median_lle=float(np.median(L)))
        out[f"{mode}_orig"] = r
        print(mode, json.dumps(r))
    json.dump(out, open(os.path.join(DATA, "stability_bound.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
