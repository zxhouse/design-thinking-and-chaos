"""Long-range dependence of the heavy-tailed ON/OFF input, checked on long series.

For long arrival series (2^18 steps) from the ON/OFF generator, and for controls
(Poisson, logistic-map input) and for the edge queue driven by ON/OFF traffic:
stationarity (KPSS, ADF, block means), distribution of values and increments
(kurtosis, Hill tail index), and the Hurst index from four estimators (DFA-1,
DFA-2, Abry-Veitch wavelet, log-periodogram) in two ranges of scales: 8-512 steps
(the scales a 250-step test sees) and >= 1024 steps (beyond the ~870-step cap of
the generator's Pareto periods). Also the tail index of the ON/OFF periods.

Output: data_v2/hurst_analysis.json, data_v2/hurst_curves.npz
"""
import os
import json
import math
import numpy as np
from scipy import stats
import fastnet as F
import nonlin as NL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
N = 2 ** 18
SHORT, LONG = (8, 512), (1024, N // 16)


def gph_band(x, smin, smax):
    """Log-periodogram slope over periods smin..smax steps; H = d + 1/2."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    jlo, jhi = max(1, int(n / smax)), int(n / smin)
    I = np.abs(np.fft.rfft(x)[jlo:jhi + 1]) ** 2 / (2 * math.pi * n)
    lam = 2 * math.pi * np.arange(jlo, jhi + 1) / n
    # average the periodogram in log-spaced bins to reduce variance
    edges = np.unique(np.logspace(np.log10(lam[0]), np.log10(lam[-1]), 40))
    k = np.digitize(lam, edges)
    xs, ys = [], []
    for b in np.unique(k):
        m = k == b
        xs.append(np.mean(-np.log(4 * np.sin(lam[m] / 2) ** 2)))
        ys.append(np.log(np.mean(I[m])))
    d = np.polyfit(xs, ys, 1)[0]
    return float(d + 0.5)


def estimates(x, rng_scales):
    lo, hi = rng_scales
    j1, j2 = int(round(np.log2(lo))), int(round(np.log2(hi)))
    return dict(dfa1=NL.hurst_dfa(x, 1, smin=max(lo, 8), smax=hi),
                dfa2=NL.hurst_dfa(x, 2, smin=max(lo, 16), smax=hi),
                wavelet=float(NL.hurst_wavelet(x, j1=max(j1, 1), j2=min(j2, int(np.log2(len(x))) - 3))[0]),
                periodogram=gph_band(x, lo, hi))


def describe(x):
    dx = np.diff(x)
    blocks = x[:len(x) // 8 * 8].reshape(8, -1)
    return dict(mean=float(x.mean()), var=float(x.var()),
                kurtosis_excess=float(stats.kurtosis(x)), kurtosis_excess_increments=float(stats.kurtosis(dx)),
                hill_increments=NL.hill(dx, 0.01), max_abs_increment=float(np.abs(dx).max()),
                block_means=[float(v) for v in blocks.mean(1)], block_vars=[float(v) for v in blocks.var(1)],
                kpss=NL.kpss(x), adf=NL.adf(x, p=20))


def onoff(seed):
    g = F.TrafficGenerator(mode="self_similar", mean_rate=52.0, hurst=0.82, seed=seed)
    a = np.empty(N)
    act = np.empty(N)
    for t in range(N):
        a[t] = g.step()
        act[t] = g.source_states.sum()
    return a, act


def main():
    out, curves = {}, {}
    # ON/OFF periods: tail index and cap
    g = F.TrafficGenerator(mode="self_similar", hurst=0.82, seed=1)
    dur = g._sample_pareto(2_000_000)
    out["periods"] = dict(alpha_nominal=g.alpha, max=int(dur.max()), mean=float(dur.mean()),
                          hill_top1pct=NL.hill(dur[dur < dur.max()], 0.01),
                          hill_top10pct=NL.hill(dur[dur < dur.max()], 0.10))
    series = {}
    for s in range(4):
        a, act = onoff(7000 + s)
        series[f"onoff_arrivals_{s}"] = a
        if s == 0:
            series["onoff_active_sources"] = act
    for mode, key in [("poisson", "poisson_arrivals"), ("chaos", "logistic_arrivals")]:
        gg = F.TrafficGenerator(mode=mode, mean_rate=52.0, seed=7100)
        series[key] = np.array([gg.step() for _ in range(N)], float)
    # edge queue driven by ON/OFF traffic, configuration that is optimal in the ground truth
    gt = np.load(os.path.join(DATA, "gt_self_similar_orig.npy"))
    best = int(gt[:, 0].argmax())
    a, x = F.inputs("self_similar", 7200, N // 2)
    Q1, _ = F.simulate(np.array([best]), a, x)
    series["onoff_queue_q1"] = Q1[1000:, 0]
    out["queue_config"] = best
    for k, v in series.items():
        out[k] = dict(describe(v), H_short=estimates(v, SHORT), H_long=estimates(v, (LONG[0], len(v) // 16)))
        print(k, json.dumps({kk: out[k][kk] for kk in ("H_short", "H_long", "kpss", "adf", "hill_increments")}))
        if k in ("onoff_arrivals_0", "poisson_arrivals", "onoff_queue_q1"):
            sc, Fs = NL.dfa(v, 1, smin=8, smax=len(v) // 8)
            curves[f"{k}_dfa_s"], curves[f"{k}_dfa_F"] = sc, Fs
            _, js, y = NL.hurst_wavelet(v, j1=1, j2=int(np.log2(len(v))) - 3)
            curves[f"{k}_wav_j"], curves[f"{k}_wav_y"] = js, y
    on = [out[f"onoff_arrivals_{s}"] for s in range(4)]
    out["onoff_arrivals_summary"] = {
        rng: {est: [float(np.mean([o[rng][est] for o in on])), float(np.std([o[rng][est] for o in on], ddof=1))]
              for est in ("dfa1", "dfa2", "wavelet", "periodogram")} for rng in ("H_short", "H_long")}
    print(json.dumps(out["onoff_arrivals_summary"], indent=1))
    json.dump(out, open(os.path.join(DATA, "hurst_analysis.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DATA, "hurst_curves.npz"), **curves)


if __name__ == "__main__":
    main()
