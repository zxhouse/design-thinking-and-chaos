"""
Experiments for the manuscript "Test Twice, Ship Once".

The network model is in ../src/network_sim.py (unchanged). This file adds:
  * exhaustive ground truth for all 2^14 configurations (16 seeds x 400 steps),
    so "% of optimum" refers to a real global optimum;
  * two objectives: 'pen' (original, with the in-test Lyapunov penalty) and
    'perf' (throughput/delay/loss only; Lyapunov exponent reported, not optimised);
  * test-once policy fixed so that no configuration is tested twice in a round;
  * paired design: every allocation uses the same seeds (same initial
    configuration and same pre-screen/test seed stream where consumption allows);
  * 100 replications, 100 rounds, results at several horizons;
  * random-search baseline under the same test budget;
  * dynamics checks: arrival-series LLE vs queue LLE, constant/Poisson input
    sweeps over load and feedback delay (is the chaos endogenous?);
  * robustness: AQM with zero drop probability at an empty queue.

Usage:  python3 experiments_v2.py [ground_truth|dynamics|allocation|aqm|all]
Paths are relative to this file.
"""

import os
import sys
import json
import time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from network_sim import NetworkSystem, TrafficGenerator, NetworkNode  # noqa: E402

DATA = os.path.join(HERE, "..", "data_v2")
os.makedirs(DATA, exist_ok=True)

N_FEAT = 14
GT_SEEDS = list(range(900001, 900017))  # 16 seeds, disjoint from controller seeds
GT_STEPS = 400
TEST_STEPS = 250
PRESCREEN_STEPS = 60


# --------------------------------------------------------------------------
# Utility components
# --------------------------------------------------------------------------
def components(res):
    norm_t = np.clip(res["throughput"] / 65.0, 0.0, 1.0)
    norm_d = np.clip(res["delay"] / 80.0, 0.0, 1.0)
    norm_l = np.clip(res["drop_rate"] / 0.15, 0.0, 1.0)
    perf = 0.50 * norm_t + 0.25 * (1.0 - norm_d) + 0.25 * (1.0 - norm_l)
    pen = float(np.clip(perf - (0.15 if res["lyapunov"] > 0.05 else 0.0), 0.0, 1.0))
    return float(np.clip(perf, 0.0, 1.0)), pen, float(res["lyapunov"])


class ZeroFloorNode(NetworkNode):
    """AQM whose drop probability is exactly zero at an empty queue."""
    def drop_probability(self):
        raw = NetworkNode.drop_probability(self)
        e0 = np.clip(self.aqm_slope * (0.0 - self.aqm_threshold), -30.0, 30.0)
        p0 = 1.0 / (1.0 + np.exp(-e0))
        return float(max(0.0, (raw - p0) / (1.0 - p0)))


def make_net(cfg, mode, seed, aqm="orig", mean_rate=52.0, tau=5, hurst=0.82):
    net = NetworkSystem(cfg, traffic_mode=mode, mean_rate=mean_rate, hurst=hurst, seed=seed)
    if aqm == "zero_floor":
        for name in ("s1", "s2"):
            old = getattr(net, name)
            new = ZeroFloorNode(old.node_id, old.capacity, old.service_rate, old.aqm_threshold, old.aqm_slope)
            setattr(net, name, new)
    if tau != 5:
        net.feedback_delay = tau
        net.recent_loss = [0.0] * tau
    return net


def run_net(cfg, mode, seed, steps, aqm="orig"):
    return components(make_net(cfg, mode, seed, aqm).simulate(steps=steps))


def idx_to_cfg(i):
    return tuple((i >> (N_FEAT - 1 - b)) & 1 for b in range(N_FEAT))


def cfg_to_idx(cfg):
    i = 0
    for b in cfg:
        i = (i << 1) | int(b)
    return i


# --------------------------------------------------------------------------
# 1. Exhaustive ground truth
# --------------------------------------------------------------------------
def _gt_chunk(args):
    mode, aqm, lo, hi = args
    out = np.zeros((hi - lo, 5))
    for j, i in enumerate(range(lo, hi)):
        cfg = idx_to_cfg(i)
        vals = np.array([run_net(cfg, mode, s, GT_STEPS, aqm) for s in GT_SEEDS])
        perf, pen, lle = vals[:, 0], vals[:, 1], vals[:, 2]
        out[j] = [perf.mean(), pen.mean(), lle.mean(), (lle > 0.05).mean(), perf.std(ddof=1)]
    return lo, out


def ground_truth(mode="chaos", aqm="orig", workers=10):
    fn = os.path.join(DATA, f"gt_{mode}_{aqm}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    n = 2 ** N_FEAT
    step = 256
    tasks = [(mode, aqm, lo, min(n, lo + step)) for lo in range(0, n, step)]
    table = np.zeros((n, 5))
    t0 = time.time()
    with Pool(workers) as pool:
        for lo, out in pool.imap_unordered(_gt_chunk, tasks):
            table[lo:lo + len(out)] = out
    np.save(fn, table)
    print(f"ground truth {mode}/{aqm}: {time.time()-t0:.0f}s")
    return table


# --------------------------------------------------------------------------
# 2. Controller (revised)
# --------------------------------------------------------------------------
class Controller:
    def __init__(self, gt, mode="chaos", objective="pen", aqm="orig", policy="iterate",
                 n_new=1, budget=5, ideas=5, radius=2, prescreen=True, seed=0):
        self.gt, self.mode, self.obj, self.aqm = gt, mode, objective, aqm
        self.policy, self.n_new, self.budget = policy, n_new, budget
        self.ideas, self.radius, self.prescreen = ideas, radius, prescreen
        self.rng = np.random.default_rng(seed)
        self.col = 0 if objective == "perf" else 1
        self.opt = gt[:, self.col].max()
        self.stats = {}  # cfg -> [sum, count]
        self.pool = []
        init = tuple(int(b) for b in self.rng.integers(0, 2, N_FEAT))
        self._test(init)
        self.pool.append(init)
        self.fav = init
        self.hist = []

    def _obs(self, cfg, steps):
        perf, pen, _ = run_net(cfg, self.mode, int(self.rng.integers(1, 10**7)), steps, self.aqm)
        return perf if self.obj == "perf" else pen

    def _test(self, cfg):
        y = self._obs(cfg, TEST_STEPS)
        s = self.stats.setdefault(cfg, [0.0, 0])
        s[0] += y
        s[1] += 1

    def belief(self, cfg):
        s = self.stats[cfg]
        return s[0] / s[1]

    def _variants(self, base):
        out = []
        b = np.array(base)
        for _ in range(self.ideas):
            d = int(self.rng.integers(1, self.radius + 1))
            flip = self.rng.choice(N_FEAT, size=d, replace=False)
            v = b.copy()
            v[flip] = 1 - v[flip]
            out.append(tuple(int(x) for x in v))
        return out

    def step(self):
        if self.policy == "random":
            for _ in range(self.budget):
                self._test(tuple(int(b) for b in self.rng.integers(0, 2, N_FEAT)))
        else:
            base = self.pool[int(self.rng.integers(len(self.pool)))]
            ideas = list(dict.fromkeys(self._variants(base)))  # unique, order kept
            if self.prescreen:
                scores = [self._obs(i, PRESCREEN_STEPS) for i in ideas]
                order = list(np.argsort(scores)[::-1])
            else:
                order = list(self.rng.permutation(len(ideas)))
            n_new = min(self.n_new, self.budget, len(ideas))
            for k in order[:n_new]:
                self._test(ideas[k])
                if ideas[k] not in self.pool:
                    self.pool.append(ideas[k])
            n_re = self.budget - n_new
            if n_re > 0:
                ranked = sorted(self.stats, key=self.belief, reverse=True)
                for r in range(n_re):
                    self._test(ranked[r % len(ranked)])
        self.fav = max(self.stats, key=self.belief)
        g = self.gt
        fi = cfg_to_idx(self.fav)
        best = max(g[cfg_to_idx(c), self.col] for c in self.stats)
        self.hist.append({
            "shipped": g[fi, self.col] / self.opt,
            "best_found": best / self.opt,
            "overest": (self.belief(self.fav) - g[fi, self.col]) / self.opt,
            "fav_lle": g[fi, 2],
            "fav_chaotic_frac": g[fi, 3],
            "n_tested": len(self.stats),
        })

    def run(self, rounds):
        for _ in range(rounds):
            self.step()
        return self.hist


HORIZONS = [10, 25, 50, 100]


def _ctrl_job(args):
    gt_key, kw, seed, rounds = args
    gt = _GT[gt_key]
    h = Controller(gt, seed=seed, **kw).run(rounds)
    return {k: [h[t - 1][k] for t in HORIZONS if t <= rounds] for k in h[0]}


_GT = {}


def _init_worker(keys):
    for k in keys:
        mode, aqm = k.split("|")
        _GT[k] = np.load(os.path.join(DATA, f"gt_{mode}_{aqm}.npy"))


def run_grid(name, variants, reps, rounds=100, workers=10):
    keys = sorted({v[0] for v in variants.values()})
    tasks, labels = [], []
    for label, (gt_key, kw) in variants.items():
        for r in range(reps):
            tasks.append((gt_key, kw, 70000 + r * 13, rounds))
            labels.append(label)
    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(keys,)) as pool:
        res = pool.map(_ctrl_job, tasks, chunksize=4)
    out = {lab: [] for lab in variants}
    for lab, r in zip(labels, res):
        out[lab].append(r)
    with open(os.path.join(DATA, f"{name}.json"), "w") as f:
        json.dump({"horizons": HORIZONS, "results": out}, f)
    print(f"{name}: {len(tasks)} runs in {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------
# 3. Dynamics checks
# --------------------------------------------------------------------------
def lle_series(x):
    """Same estimator as network_sim (delay 2, dim 3, +4 steps, +-5 exclusion)."""
    return NetworkSystem._estimate_lyapunov(None, np.asarray(x, dtype=float))


def dynamics():
    cfg = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]  # configuration used in Fig. 3
    out = {"input_vs_queue": [], "sweep": []}
    # (a) LLE of the chaotic arrival series itself vs LLE of the queue it drives
    for s in range(40):
        g = TrafficGenerator(mode="chaos", mean_rate=55.0, seed=5000 + s)
        arr = [g.step() for _ in range(600)]
        net = make_net(cfg, "chaos", 5000 + s, mean_rate=55.0)
        res = net.simulate(steps=600)
        out["input_vs_queue"].append([lle_series(arr[50:]), res["lyapunov"]])
    # (b) non-chaotic input, sweep load and feedback delay: is there endogenous chaos?
    for mode in ["constant", "poisson"]:
        for tau in [1, 5, 10, 20, 40]:
            for lam in np.linspace(30, 90, 13):
                lles, amps = [], []
                for s in range(6):
                    m = "poisson" if mode == "poisson" else "poisson"
                    net = make_net(cfg, m, 6000 + s, mean_rate=float(lam), tau=tau)
                    if mode == "constant":
                        net.traffic_gen.step = (lambda L=float(lam): L)
                        net.cross_traffic_gen.step = (lambda: 15.0)
                    res = net.simulate(steps=600)
                    q = np.array(res["q1_trajectory"][200:])
                    lles.append(res["lyapunov"])
                    amps.append(float(q.max() - q.min()))
                out["sweep"].append({"input": mode, "tau": tau, "load": float(lam),
                                     "lle": float(np.mean(lles)), "q1_range": float(np.mean(amps))})
    with open(os.path.join(DATA, "dynamics.json"), "w") as f:
        json.dump(out, f)
    print("dynamics done")


# --------------------------------------------------------------------------
def allocation():
    v = {}
    for obj in ["pen", "perf"]:
        for n in [1, 2, 3, 4, 5]:
            v[f"{obj}|new{n}"] = ("chaos|orig", dict(objective=obj, n_new=n))
        v[f"{obj}|random"] = ("chaos|orig", dict(objective=obj, policy="random"))
    run_grid("allocation_v2", v, reps=100, rounds=100)
    v = {
        "poisson|new1": ("poisson|orig", dict(mode="poisson", objective="pen", n_new=1)),
        "poisson|new5": ("poisson|orig", dict(mode="poisson", objective="pen", n_new=5)),
        "self_similar|new1": ("self_similar|orig", dict(mode="self_similar", objective="pen", n_new=1)),
        "self_similar|new5": ("self_similar|orig", dict(mode="self_similar", objective="pen", n_new=5)),
    }
    run_grid("regimes_v2", v, reps=100, rounds=100)


def ablations():
    base = dict(objective="pen", n_new=1)
    v = {
        "baseline": base,
        "radius1": {**base, "radius": 1},
        "radius14": {**base, "radius": 14},
        "no_prescreen": {**base, "prescreen": False},
        "ideas1": {**base, "ideas": 1},
        "ideas15": {**base, "ideas": 15},
    }
    run_grid("ablations_v2", {k: ("chaos|orig", kw) for k, kw in v.items()}, reps=100, rounds=50)


def aqm():
    v = {f"zero_floor|new{n}": ("chaos|zero_floor", dict(aqm="zero_floor", objective="pen", n_new=n)) for n in [1, 5]}
    v["zero_floor|random"] = ("chaos|zero_floor", dict(aqm="zero_floor", objective="pen", policy="random"))
    run_grid("aqm_v2", v, reps=100, rounds=100)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("ground_truth", "all"):
        ground_truth("chaos", "orig")
        ground_truth("poisson", "orig")
        ground_truth("self_similar", "orig")
        ground_truth("chaos", "zero_floor")
    if what in ("dynamics", "all"):
        dynamics()
    if what in ("allocation", "all"):
        allocation()
    if what in ("ablations", "all"):
        ablations()
    if what in ("aqm", "all"):
        aqm()
