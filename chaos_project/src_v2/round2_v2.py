"""Supplementary experiments:
  (a) endogenous-oscillation sweep over 200 random configurations (constant input);
  (b) balanced policy (3 new / 2 re-tests) in the other regimes.
"""
import os, json, numpy as np
from multiprocessing import Pool
import experiments_v2 as E

def _cfg_sweep(i):
    rng = np.random.default_rng(800000 + i)
    cfg = tuple(int(b) for b in rng.integers(0, 2, E.N_FEAT))
    rows = []
    for tau in (5, 40):
        for lam in (30.0, 50.0, 70.0, 90.0):
            net = E.make_net(cfg, "poisson", 1, mean_rate=lam, tau=tau)
            net.traffic_gen.step = (lambda L=lam: L)
            net.cross_traffic_gen.step = (lambda: 15.0)
            res = net.simulate(steps=600)
            q1 = np.array(res["q1_trajectory"][200:]); q2 = np.array(res["q2_trajectory"][200:])
            rows.append([tau, lam, float(np.ptp(q1)), float(np.ptp(q2)), res["lyapunov"]])
    return rows

if __name__ == "__main__":
    with Pool(10) as p:
        rows = [r for rs in p.map(_cfg_sweep, range(200)) for r in rs]
    a = np.array(rows)
    out = {"n_cfg": 200, "max_q1_range": float(a[:, 2].max()), "max_q2_range": float(a[:, 3].max()),
           "frac_q_range_gt_5": float(((a[:, 2] > 5) | (a[:, 3] > 5)).mean()),
           "frac_lle_gt_005": float((a[:, 4] > 0.05).mean()), "max_lle": float(a[:, 4].max())}
    json.dump(out, open(os.path.join(E.DATA, "round2_sweep.json"), "w"), indent=1)
    print(out)
    v = {}
    for key, kw in [("self_similar|orig", dict(mode="self_similar")), ("poisson|orig", dict(mode="poisson")),
                    ("chaos|zero_floor", dict(aqm="zero_floor"))]:
        for n in (3,):
            v[f"{key.split('|')[0]}_{key.split('|')[1]}|new{n}"] = (key, dict(objective="pen", n_new=n, **kw))
    E.run_grid("round2_balanced", v, reps=100, rounds=100)
