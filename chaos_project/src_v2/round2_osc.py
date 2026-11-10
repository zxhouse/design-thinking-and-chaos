"""Endogenous-oscillation sweep: where do endogenous oscillations occur (constant input, 200 random configs)?"""
import os, json, numpy as np
from multiprocessing import Pool
import experiments_v2 as E
LOADS = (30., 40., 50., 52., 60., 70., 80., 90.)
def job(i):
    rng = np.random.default_rng(800000 + i); cfg = tuple(int(b) for b in rng.integers(0, 2, 14))
    out = []
    for tau in (5, 40):
        for lam in LOADS:
            net = E.make_net(cfg, "poisson", 1, mean_rate=lam, tau=tau)
            net.traffic_gen.step = (lambda L=lam: L); net.cross_traffic_gen.step = (lambda: 15.)
            q = np.array(net.simulate(steps=1200)["q1_trajectory"][600:])
            late = float(np.ptp(q[-300:]))
            # distinct values visited at the end (rounded): 1 = fixed point, small = periodic, many = irregular
            distinct = len(set(np.round(q[-200:], 2)))
            # Lyapunov estimate on the deterministic series (same estimator)
            lle = E.lle_series(q)
            out.append([tau, lam, late, distinct, lle])
    return out
if __name__ == "__main__":
    with Pool(10) as p: a = np.array([r for rs in p.map(job, range(200)) for r in rs])
    res = {}
    for tau in (5, 40):
        for lam in LOADS:
            m = (a[:, 0] == tau) & (a[:, 1] == lam)
            osc = a[m, 2] > 1.0
            irregular = osc & (a[m, 3] > 20)
            res[f"tau{tau}_lam{int(lam)}"] = {"frac_osc": float(osc.mean()), "frac_irregular": float(irregular.mean()),
                "median_lle_irregular": float(np.median(a[m][irregular, 4])) if irregular.any() else None}
    json.dump(res, open(os.path.join(E.DATA, "round2_osc.json"), "w"), indent=1)
    for k, v in res.items(): print(k, v)
