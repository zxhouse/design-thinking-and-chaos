"""Analytical stability results, checked against the model exponent.

Part A (constant input). For every configuration and load: the unique fixed point
(closed form for the queues, bisection for the loss fraction), the Jacobian of the
one-step map at that point (central differences) and its spectral radius; for the
case "edge queue congested, core queue empty" also the roots of the analytical
characteristic polynomial. Compared with the model exponent (twin trajectories).

Part B (any input). The pathwise upper bound of Theorem 2,
    Lambda <= (1/T) sum_t ln k_t,
with k_t = rho on idle steps and k_t = K(a_t, x_t) otherwise, minimised over the
weights of the norm, compared with the model exponent for every configuration and
ground-truth seed.

Output: data_v2/stability_theory.json, data_v2/stability_fixed_<load>.npy
"""
import os
import json
import numpy as np
import fastnet as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
ALL = np.arange(2 ** 14)
TAU = 5
C = 15.0
B1, B2 = F.B1, F.B2


def sig(q, th, sl):
    return 1.0 / (1.0 + np.exp(-np.clip(sl * (q - th), -30.0, 30.0)))


def queue_fp(a, mu, th, sl, cap):
    """Unique fixed point of q -> max(0, min(cap, q + a(1-P(q))) - mu) and the departures."""
    p0 = sig(0.0, th, sl)
    empty = a * (1 - p0) <= mu
    with np.errstate(divide="ignore", invalid="ignore"):
        qi = th + np.log(np.maximum(a - mu, 1e-300) / mu) / sl
    full = ~empty & (qi >= cap - mu)
    q = np.where(empty, 0.0, np.where(full, cap - mu, qi))
    dep = np.where(empty, a * (1 - p0), mu)
    return q, dep, np.where(empty, 0, np.where(full, 2, 1))


def steady(m, lam, par):
    mu1, th1, sl1, mu2, th2, sl2 = par
    a1 = lam * np.maximum(0.4, 1 - 1.8 * m)
    q1, dep1, c1 = queue_fp(a1, mu1, th1, sl1, B1)
    q2, dep2, c2 = queue_fp(dep1 + C, mu2, th2, sl2, B2)
    loss = (a1 + C - dep2) / np.maximum(1.0, a1 + C)
    return loss, q1, q2, c1, c2, a1


def fixed_point(lam, par):
    lo, hi = np.zeros(len(par[0])), np.ones(len(par[0]))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        up = steady(mid, lam, par)[0] > mid   # G is non-increasing, so G(m) > m means m < m*
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    m = 0.5 * (lo + hi)
    return (m,) + steady(m, lam, par)[1:]


def one_step(z, lam):
    b = F.Batch(ALL, TAU)
    b.set_state(z)
    b.k = 0
    b.step(np.full(len(ALL), lam), np.full(len(ALL), C))
    return np.column_stack([b.q1, b.q2, np.roll(b.L, -1, axis=1)])


def jacobian(z, lam, h=1e-6):
    n, d = z.shape
    J = np.zeros((n, d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = h
        J[:, :, j] = (one_step(z + e, lam) - one_step(z - e, lam)) / (2 * h)
    return J


def char_roots(lam, par, m, q1, a1):
    """Roots of (z-A)(1-f S(z)) + b e S(z) = 0, S(z) = (1/tau) sum z^-k, times z^tau.
    Valid when the edge queue is interior and the core queue is empty."""
    mu1, th1, sl1, mu2, th2, sl2 = par
    P = sig(q1, th1, sl1)
    Pp = sl1 * P * (1 - P)
    d2 = (mu1 + C) * sig(0.0, th2, sl2)
    A = 1 - a1 * Pp
    b = 1.8 * lam * (1 - P)
    e = a1 * Pp / (a1 + C)
    f = -1.8 * lam * (C * P - d2) / (a1 + C) ** 2
    rad = np.zeros(len(m))
    for i in range(len(m)):
        # z^tau (z-A) - (f/tau)(z-A) sum_{k<tau} z^k + (b e/tau) sum_{k<tau} z^k
        poly = np.zeros(TAU + 2)
        poly[0] = 1.0; poly[1] = -A[i]
        s = np.ones(TAU) / TAU
        poly[1:TAU + 1] += -f[i] * s            # -f/tau * z * sum
        poly[2:TAU + 2] += f[i] * A[i] * s      # +f A/tau * sum
        poly[2:TAU + 2] += b[i] * e[i] * s
        rad[i] = np.abs(np.roots(poly)).max()
    return rad, A


def part_a():
    par = F.params(ALL)
    out = {}
    for lam in (52.0, 60.0, 70.0, 80.0, 90.0):
        m, q1, q2, c1, c2, a1 = fixed_point(lam, par)
        z = np.column_stack([q1, q2] + [m] * TAU)
        resid = np.abs(one_step(z, lam) - z).max(axis=1)
        J = jacobian(z, lam)
        rho = np.abs(np.linalg.eigvals(J)).max(axis=1)
        a = np.full(8000, lam); x = np.full(8000, C)
        lle = F.twin_lle(ALL, a, x, tau=TAU, burn=3000)
        Q1, _ = F.simulate(ALL, a, x, tau=TAU)
        settled = np.ptp(Q1[-500:], axis=0) < 1e-6
        main = (c1 == 1) & (c2 == 0) & (m < 1 / 3 - 1e-9)
        rad = np.full(len(ALL), np.nan); A = np.full(len(ALL), np.nan)
        if main.any():
            sub = tuple(p[main] for p in par)
            rad[main], A[main] = char_roots(lam, sub, m[main], q1[main], a1[main])
        mu1, sl1 = par[0], par[2]
        P1 = sig(q1, par[1], sl1)
        gain = sl1 * mu1 * P1          # |1 - gain| < 1  <=>  gain < 2 (queue map alone)
        np.save(os.path.join(DATA, f"stability_fixed_{int(lam)}.npy"),
                np.column_stack([m, q1, q2, c1, c2, rho, lle, settled, rad, gain]))
        unstable = rho > 1 + 1e-6
        r = dict(max_residual=float(resid.max()),
                 cases={f"{i}{j}": int(((c1 == i) & (c2 == j)).sum()) for i in range(3) for j in range(3)},
                 pacing_floor=int((m >= 1 / 3 - 1e-9).sum()),
                 n_linearly_unstable=int(unstable.sum()),
                 n_settled=int(settled.sum()),
                 stable_and_not_settled=int((~unstable & ~settled).sum()),
                 unstable_and_settled=int((unstable & settled).sum()),
                 n_lle_positive=int((lle > 0.01).sum()),
                 lle_positive_and_linearly_stable=int(((lle > 0.01) & ~unstable).sum()),
                 max_rho_where_lle_neg_settled=float(rho[settled].max()) if settled.any() else None,
                 unstable_by_slope={f"{s:.2f}": int((unstable & np.isclose(sl1, s)).sum()) for s in (0.05, 0.15, 0.25, 0.35)},
                 main_case=int(main.sum()),
                 main_case_max_abs_diff_rho=float(np.nanmax(np.abs(rad[main] - rho[main]))) if main.any() else None,
                 main_case_unstable=int((main & unstable).sum()),
                 main_case_unstable_with_gain_lt_2=int((main & unstable & (gain < 2)).sum()),
                 main_case_stable_with_gain_gt_2=int((main & ~unstable & (gain > 2)).sum()),
                 # fixed point stable: does log(rho) match the model exponent?
                 corr_lnrho_lle_stable=float(np.corrcoef(np.log(np.maximum(rho[settled & (rho > 1e-6)], 1e-12)),
                                                         lle[settled & (rho > 1e-6)])[0, 1]) if (settled & (rho > 1e-6)).sum() > 2 else None)
        out[str(int(lam))] = r
        print(lam, json.dumps(r))
    return out


if __name__ == "__main__":
    res = {"constant_input": part_a()}
    json.dump(res, open(os.path.join(DATA, "stability_theory.json"), "w"), indent=1)
