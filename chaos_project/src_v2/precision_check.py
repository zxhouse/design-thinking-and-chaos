"""Do the results depend on floating-point precision?

The model is re-implemented as a scalar map that runs in single precision
(float32), double precision (float64) and 50-digit arithmetic (mpmath). For the
three chaotic configurations at load 90 we compare trajectories (time until they
part by more than one packet), the model exponent, and statistics of the
attractor. For stable cases (the same configurations at load 52, and the
configuration of Fig. 2 driven by logistic-map traffic) we compare trajectories
directly. All precisions are fed the same input series.

Output: data_v2/precision_check.json
"""
import os
import json
import math
import numpy as np
import mpmath as mp
from scipy.stats import ks_2samp
import fastnet as F
import nonlin as NL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
mp.mp.dps = 50
TAU = 5


class Ops:
    def __init__(self, kind):
        self.kind = kind
        if kind == "f32":
            self.c = np.float32; self.exp = lambda v: np.exp(v, dtype=np.float32)
        elif kind == "f64":
            self.c = float; self.exp = math.exp
        else:
            self.c = mp.mpf; self.exp = mp.exp


def node(o, q, a, mu, th, sl, cap):
    c = o.c
    z = sl * (q - th)
    z = max(c(-30), min(c(30), z))
    p = c(1) / (c(1) + o.exp(-z))
    acc = a * (c(1) - p)
    drop = a * p
    avail = cap - q
    if acc > avail:
        drop = drop + (acc - avail)
        acc = max(c(0), avail)
    q = q + acc
    dep = min(q, mu)
    return q - dep, dep, drop


def step(o, st, a, x, par):
    c = o.c
    q1, q2, L = st
    mu1, th1, sl1, mu2, th2, sl2 = par
    m = sum(L, c(0)) / c(TAU)
    a1 = a * max(c("0.4") if o.kind == "mp" else c(0.4), c(1) - (c("1.8") if o.kind == "mp" else c(1.8)) * m)
    q1, dep1, d1 = node(o, q1, a1, mu1, th1, sl1, c(F.B1))
    q2, dep2, d2 = node(o, q2, dep1 + x, mu2, th2, sl2, c(F.B2))
    loss = (d1 + d2) / max(c(1), a1 + x)
    return q1, q2, L[1:] + [loss]


def params(o, idx):
    raw = [float(v[0]) for v in F.params(np.array([idx]))]
    if o.kind == "mp":   # exact decimal parameters
        return [mp.mpf(repr(round(v, 6))) for v in raw]
    return [o.c(v) for v in raw]


def run(kind, idx, a, x, st0=None):
    o = Ops(kind)
    par = params(o, idx)
    st = (o.c(0), o.c(0), [o.c(0)] * TAU) if st0 is None else st0
    out = np.zeros(len(a))
    for t in range(len(a)):
        st = step(o, st, o.c(float(a[t])), o.c(float(x[t])), par)
        out[t] = float(st[0])
    return out, st


def twin(kind, idx, a, x, burn, delta):
    """Model exponent by twin trajectories in the given arithmetic."""
    o = Ops(kind)
    c = o.c
    par = params(o, idx)
    st = (c(0), c(0), [c(0)] * TAU)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(TAU + 2); u /= np.linalg.norm(u)
    acc, cnt, tw = c(0), 0, None
    d = c(delta)
    for t in range(len(a)):
        if t == burn:
            tw = (st[0] + d * c(float(u[0])), st[1] + d * c(float(u[1])), [st[2][k] + d * c(float(u[2 + k])) for k in range(TAU)])
        at, xt = c(float(a[t])), c(float(x[t]))
        st = step(o, st, at, xt, par)
        if tw is not None:
            tw = step(o, tw, at, xt, par)
            diff = [tw[0] - st[0], tw[1] - st[1]] + [tw[2][k] - st[2][k] for k in range(TAU)]
            nd = (mp.sqrt if kind == "mp" else math.sqrt)(sum(v * v for v in diff))
            if nd == 0:
                return float("-inf")
            acc += (mp.log if kind == "mp" else math.log)(nd / d)
            cnt += 1
            s = d / nd
            tw = (st[0] + diff[0] * s, st[1] + diff[1] * s, [st[2][k] + diff[2 + k] * s for k in range(TAU)])
    return float(acc / cnt)


def first_apart(u, v, tol=1.0):
    w = np.where(np.abs(u - v) > tol)[0]
    return int(w[0]) if len(w) else None


def main():
    od = json.load(open(os.path.join(DATA, "overload_dynamics.json")))
    picks = od["picked_configs"]
    out = {"digits_reference": mp.mp.dps, "chaotic": {}, "stable": {}}
    N, BURN = 6000, 1000
    for p in picks:
        a = np.full(N, 90.0); x = np.full(N, 15.0)
        tr = {k: run(k, p, a, x)[0] for k in ("f32", "f64", "mp")}
        lam64 = twin("f64", p, a, x, BURN, 1e-7)
        lam_mp = twin("mp", p, a, x, BURN, 1e-7)
        lam_mp_small = twin("mp", p, a, x, BURN, 1e-25)
        ref = tr["mp"][BURN:]
        r = dict(model_exponent_f64=lam64, model_exponent_mp=lam_mp, model_exponent_mp_delta1e25=lam_mp_small,
                 steps_until_apart_f64=first_apart(tr["f64"], tr["mp"]), steps_until_apart_f32=first_apart(tr["f32"], tr["mp"]),
                 predicted_f64=math.log(1.0 / 2.2e-16) / lam_mp, predicted_f32=math.log(1.0 / 1.2e-7) / lam_mp)
        for k in ("f32", "f64", "mp"):
            s = tr[k][BURN:]
            r[k] = dict(mean=float(s.mean()), sd=float(s.std()), q05=float(np.quantile(s, 0.05)), q95=float(np.quantile(s, 0.95)),
                        PE4=NL.permutation_entropy(s, 4), ks_vs_mp=float(ks_2samp(s, ref)[0]), ks_p=float(ks_2samp(s, ref)[1]))
        out["chaotic"][str(p)] = r
        print(p, json.dumps(r))
    # stable cases
    cases = [(f"cfg{p}_const52", p, np.full(2000, 52.0), np.full(2000, 15.0)) for p in picks]
    cfg = int("00101000010010", 2)
    a, x = F.inputs("chaos", 5000, 2000, mean_rate=55.0)
    cases.append((f"cfg{cfg}_logistic55", cfg, a, x))
    for mode in ("chaos", "poisson", "self_similar"):
        gt = np.load(os.path.join(DATA, f"gt_{mode}_orig.npy"))
        best = int(gt[:, 1].argmax())
        a, x = F.inputs(mode, 900001, 2000)
        cases.append((f"best{best}_{mode}", best, a, x))
    for name, p, a, x in cases:
        tr = {k: run(k, p, a, x)[0] for k in ("f32", "f64", "mp")}
        r = dict(max_abs_diff_f64=float(np.abs(tr["f64"] - tr["mp"]).max()), max_abs_diff_f32=float(np.abs(tr["f32"] - tr["mp"]).max()),
                 model_exponent_f64=twin("f64", p, a, x, 50, 1e-7), model_exponent_mp=twin("mp", p, a, x, 50, 1e-7),
                 q1_mean=float(tr["mp"].mean()), q1_max=float(tr["mp"].max()))
        out["stable"][name] = r
        print(name, json.dumps(r))
    # the vectorised implementation against the scalar one in double precision
    a = np.full(500, 90.0); x = np.full(500, 15.0)
    Q1, _ = F.simulate(np.array(picks[:1]), a, x)
    out["scalar_vs_vectorised_f64"] = float(np.abs(Q1[:, 0] - run("f64", picks[0], a, x)[0])[:100].max())
    json.dump(out, open(os.path.join(DATA, "precision_check.json"), "w"), indent=1)
    print(out["scalar_vs_vectorised_f64"])


if __name__ == "__main__":
    main()
