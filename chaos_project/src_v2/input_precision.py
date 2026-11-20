"""Does the precision of the logistic-map traffic generator change the test results?

The chaotic input is regenerated with the logistic map iterated in single
precision, double precision and 50-digit arithmetic (same initial value and the
same random kicks). For the 16 ground-truth seeds we record when the arrival
series part, and their summary statistics. For 300 random configurations and the
best one we then recompute the ground-truth scores (16 seeds x 400 steps) with each
version of the input and compare them.

Output: data_v2/input_precision.json
"""
import os
import json
import numpy as np
import mpmath as mp
from scipy.stats import spearmanr
import fastnet as F
import nonlin as NL
import experiments_v2 as E

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
mp.mp.dps = 50
SEEDS = list(range(900001, 900017))
STEPS = 400


def series(kind, seed, mean_rate):
    g = F.TrafficGenerator(mode="chaos", mean_rate=mean_rate, seed=seed)
    c = {"f32": np.float32, "f64": float, "mp": mp.mpf}[kind]
    lit = (lambda s: mp.mpf(s)) if kind == "mp" else (lambda s: c(float(s)))
    x, r = c(g.chaos_x), lit("3.89")
    out = []
    for _ in range(STEPS):
        x = r * x * (lit("1") - x)
        if g.rng.random() < 0.01:
            x = (x + lit("0.01")) % lit("1")
        out.append(int(np.round(float(lit(repr(mean_rate)) * (lit("0.3") + lit("1.4") * x)))))
    return out


def run(idx, seed, a, x):
    net = E.make_net(E.idx_to_cfg(int(idx)), "chaos", seed)
    ia, ix = iter(a), iter(x)
    net.traffic_gen.step = lambda: next(ia)
    net.cross_traffic_gen.step = lambda: next(ix)
    return E.components(net.simulate(steps=STEPS))


def main():
    gt = np.load(os.path.join(DATA, "gt_chaos_orig.npy"))
    rng = np.random.default_rng(11)
    cfgs = np.unique(np.append(rng.choice(2 ** 14, 300, replace=False), gt[:, 1].argmax()))
    inp = {k: {s: (series(k, s, 52.0), series(k, s + 100, 15.0)) for s in SEEDS} for k in ("f32", "f64", "mp")}
    # the double-precision copy must be the original generator
    a0, x0 = F.inputs("chaos", SEEDS[0], STEPS)
    assert list(a0.astype(int)) == inp["f64"][SEEDS[0]][0] and list(x0.astype(int)) == inp["f64"][SEEDS[0]][1]
    out = {"n_configs": int(len(cfgs)), "seeds": len(SEEDS), "steps": STEPS, "series": {}, "scores": {}}
    for k in ("f32", "f64"):
        first = [next((t for t in range(STEPS) if inp[k][s][0][t] != inp["mp"][s][0][t]), STEPS) for s in SEEDS]
        same = [np.mean(np.array(inp[k][s][0]) == np.array(inp["mp"][s][0])) for s in SEEDS]
        out["series"][f"{k}_first_difference_min"] = int(min(first))
        out["series"][f"{k}_first_difference_median"] = float(np.median(first))
        out["series"][f"{k}_first_difference_max"] = int(max(first))
        out["series"][f"{k}_share_equal_values"] = float(np.mean(same))
    for k in ("f32", "f64", "mp"):
        A = np.array([inp[k][s][0] for s in SEEDS], float)
        out["series"][k] = dict(mean=float(A.mean()), sd=float(A.std()),
                                PE4=float(np.mean([NL.permutation_entropy(r, 4) for r in A])),
                                lag1=float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in A])))
    res = {}
    for k in ("f32", "f64", "mp"):
        R = np.array([[run(i, s, *inp[k][s]) for s in SEEDS] for i in cfgs])   # (cfg, seed, 3)
        res[k] = R
    # double precision reproduces the ground-truth table
    out["f64_matches_ground_truth"] = float(np.abs(res["f64"][:, :, 1].mean(1) - gt[cfgs, 1]).max())
    top = gt[:, 1].max()
    for k in ("f32", "f64"):
        for j, name in [(0, "performance"), (1, "penalised")]:
            m, ref = res[k][:, :, j].mean(1), res["mp"][:, :, j].mean(1)
            se = np.sqrt(res[k][:, :, j].var(1, ddof=1) / len(SEEDS) + res["mp"][:, :, j].var(1, ddof=1) / len(SEEDS))
            d = m - ref
            z = d[se > 0] / se[se > 0]
            out["scores"][f"{k}_{name}"] = dict(mean_diff_pct_of_optimum=float(100 * d.mean() / top),
                                                ci95_pct=float(100 * 1.96 * d.std(ddof=1) / np.sqrt(len(d)) / top),
                                                mean_abs_diff_pct=float(100 * np.abs(d).mean() / top),
                                                max_abs_diff_pct=float(100 * np.abs(d).max() / top),
                                                share_within_2se=float((np.abs(z) < 2).mean()),
                                                spearman=float(spearmanr(m, ref)[0]))
        fl, flr = (res[k][:, :, 2] > 0.05).mean(), (res["mp"][:, :, 2] > 0.05).mean()
        out["scores"][f"{k}_flag_rate"] = [float(fl), float(flr)]
    best = int(np.where(cfgs == gt[:, 1].argmax())[0][0])
    out["best_config_rank"] = {k: int((res[k][:, :, 1].mean(1) > res[k][best, :, 1].mean()).sum()) + 1 for k in res}
    json.dump(out, open(os.path.join(DATA, "input_precision.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
