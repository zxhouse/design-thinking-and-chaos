"""Is the queue's irregularity related to that of the chaotic input, linearly or not?

Same set-up as experiments_v2.dynamics() part (a) (configuration of Fig. 2, load 55,
600 steps) but with 400 seeds instead of 40. For each seed: the short-window
indicator of the arrival series and of q1, the model's conditional exponent and the
permutation entropy of both series. Dependence is measured by Pearson, Spearman and
Kendall coefficients, distance correlation and mutual information (k-NN), each with a
permutation p-value, and by a quadratic regression.

Output: data_v2/input_queue_dependence.json
"""
import os
import json
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import fastnet as F
import nonlin as NL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
CFG = int("00101000010010", 2)
N_SEEDS = 400
EST = F.NetworkSystem._estimate_lyapunov


def dcor(x, y):
    def centred(v):
        D = np.abs(v[:, None] - v[None, :])
        return D - D.mean(0) - D.mean(1)[:, None] + D.mean()
    A, B = centred(x), centred(y)
    v = (A * B).mean()
    return float(np.sqrt(max(v, 0) / np.sqrt((A * A).mean() * (B * B).mean())))


def perm_p(f, x, y, n=999, seed=0):
    rng = np.random.default_rng(seed)
    obs = f(x, y)
    null = np.array([f(x, rng.permutation(y)) for _ in range(n)])
    return obs, float((1 + (null >= obs).sum()) / (n + 1))


def mi(x, y):
    return float(mutual_info_regression(x[:, None], y, n_neighbors=5, random_state=0)[0])


def main():
    rows = []
    for s in range(N_SEEDS):
        a, x = F.inputs("chaos", 5000 + s, 600, mean_rate=55.0)
        Q1, _ = F.simulate(np.array([CFG]), a, x)
        q = Q1[:, 0]
        model = float(F.twin_lle(np.array([CFG]), a, x, burn=50, seed=s)[0])
        rows.append([EST(None, a[50:]), EST(None, q[50:]), model,
                     NL.permutation_entropy(a[50:], 4), NL.permutation_entropy(q[50:], 4)])
    R = np.array(rows)
    xin, xq = R[:, 0], R[:, 1]
    out = {"n": N_SEEDS, "input_mean": float(xin.mean()), "input_sd": float(xin.std(ddof=1)),
           "queue_mean": float(xq.mean()), "queue_sd": float(xq.std(ddof=1)),
           "first40_pearson": float(np.corrcoef(xin[:40], xq[:40])[0, 1])}
    r, p = stats.pearsonr(xin, xq); out["pearson"] = [float(r), float(p)]
    r, p = stats.spearmanr(xin, xq); out["spearman"] = [float(r), float(p)]
    r, p = stats.kendalltau(xin, xq); out["kendall"] = [float(r), float(p)]
    out["dcor"] = list(perm_p(dcor, xin, xq))
    out["mutual_info"] = list(perm_p(mi, xin, xq, n=199))
    X = np.column_stack([np.ones_like(xin), xin, xin ** 2])
    beta, *_ = np.linalg.lstsq(X, xq, rcond=None)
    rss1 = ((xq - X @ beta) ** 2).sum()
    rss0 = ((xq - xq.mean()) ** 2).sum()
    Fst = ((rss0 - rss1) / 2) / (rss1 / (len(xq) - 3))
    out["quadratic"] = dict(R2=float(1 - rss1 / rss0), F=float(Fst), p=float(stats.f.sf(Fst, 2, len(xq) - 3)))
    out["model_lle"] = dict(mean=float(R[:, 2].mean()), max=float(R[:, 2].max()), frac_positive=float((R[:, 2] > 0).mean()))
    out["PE_input"] = dict(mean=float(R[:, 3].mean()), sd=float(R[:, 3].std(ddof=1)))
    out["PE_queue"] = dict(mean=float(R[:, 4].mean()), sd=float(R[:, 4].std(ddof=1)))
    r, p = stats.spearmanr(R[:, 3], R[:, 4]); out["PE_spearman"] = [float(r), float(p)]
    out["pairs"] = R.tolist()
    json.dump(out, open(os.path.join(DATA, "input_queue_dependence.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
