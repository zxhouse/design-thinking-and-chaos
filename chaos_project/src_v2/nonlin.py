"""Time-series tools used in the dynamics and long-memory analyses.

Lyapunov exponents from a series: Rosenstein (divergence curve), Kantz, Wolf-type
(fixed evolution time with replacement). Correlation dimension (Grassberger-Procaccia).
Permutation entropy (Bandt-Pompe). 0-1 test for chaos (Gottwald-Melbourne).
Hurst index: DFA, log-periodogram (GPH), Abry-Veitch wavelet (Daubechies-3),
aggregated variance. Stationarity: KPSS (level) and ADF (constant). Tail index: Hill.
Synthetic fGn (Davies-Harte) for validation. `selftest()` checks every estimator on
systems with known values.
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, zeta


# ----------------------------------------------------------------- embedding
def embed(x, m, tau):
    x = np.asarray(x, float)
    n = len(x) - (m - 1) * tau
    return np.column_stack([x[i * tau:i * tau + n] for i in range(m)])


def _fit_slope(k, y):
    k, y = np.asarray(k, float), np.asarray(y, float)
    ok = np.isfinite(y)
    return float(np.polyfit(k[ok], y[ok], 1)[0]) if ok.sum() >= 2 else float("nan")


# ----------------------------------------------------------------- Lyapunov
def rosenstein(x, m=3, tau=2, theiler=10, kmax=20, kfit=(1, 8), n_ref=None, seed=0):
    """Mean log divergence of nearest neighbours vs k; slope over kfit (per step)."""
    X = embed(x, m, tau)
    N = len(X) - kmax
    if N < 20:
        return float("nan"), None
    tree = cKDTree(X[:N])
    refs = np.arange(N) if n_ref is None or n_ref >= N else \
        np.sort(np.random.default_rng(seed).choice(N, n_ref, replace=False))
    kq = min(N, 2 * theiler + 4)
    d, j = tree.query(X[refs], k=kq)
    curve_sum, curve_n = np.zeros(kmax + 1), np.zeros(kmax + 1)
    for r, i in enumerate(refs):
        ok = (np.abs(j[r] - i) > theiler) & (d[r] > 1e-12)
        if not ok.any():
            continue
        nn = j[r][np.argmax(ok)]
        dk = np.linalg.norm(X[i:i + kmax + 1] - X[nn:nn + kmax + 1], axis=1)
        good = dk > 1e-12
        curve_sum[good] += np.log(dk[good])
        curve_n[good] += 1
    curve = curve_sum / np.maximum(curve_n, 1)
    ks = np.arange(kfit[0], kfit[1] + 1)
    return _fit_slope(ks, curve[ks]), curve


def kantz(x, m=3, tau=2, eps=None, theiler=10, kmax=20, kfit=(1, 8), n_ref=2000, seed=0):
    """Kantz (1994): S(k) = < ln mean_{j in U_eps(i)} |X_{i+k}-X_{j+k}| >; slope over kfit."""
    X = embed(x, m, tau)
    N = len(X) - kmax
    if N < 20:
        return float("nan"), None
    if eps is None:
        eps = 0.05 * np.std(X[:, 0]) * math.sqrt(m) + 1e-12
    tree = cKDTree(X[:N])
    rng = np.random.default_rng(seed)
    refs = np.arange(N) if N <= n_ref else rng.choice(N, n_ref, replace=False)
    S, cnt = np.zeros(kmax + 1), 0
    for i in refs:
        nb = np.array(tree.query_ball_point(X[i], eps), int)
        nb = nb[np.abs(nb - i) > theiler]
        if len(nb) == 0:
            continue
        dk = np.linalg.norm(X[nb[:, None] + np.arange(kmax + 1)] - X[i + np.arange(kmax + 1)][None], axis=2)
        mean_d = dk.mean(axis=0)
        if np.any(mean_d <= 0):
            continue
        S += np.log(mean_d)
        cnt += 1
    if cnt == 0:
        return float("nan"), None
    S /= cnt
    ks = np.arange(kfit[0], kfit[1] + 1)
    return _fit_slope(ks, S[ks]), S


def wolf(x, m=3, tau=2, evolve=3, theiler=10, dmin_frac=1e-3, dmax_frac=0.1, k_cand=40):
    """Wolf et al. (1985)-type estimate: follow one neighbour for `evolve` steps,
    accumulate the log stretch, replace it by the candidate closest in angle."""
    X = embed(x, m, tau)
    N = len(X) - evolve - 1
    scale = np.std(X) * math.sqrt(m)
    dmin, dmax = dmin_frac * scale, dmax_frac * scale
    tree = cKDTree(X[:N])

    def candidates(i):
        d, j = tree.query(X[i], k=min(k_cand, N))
        ok = (np.abs(j - i) > theiler) & (d > dmin)
        return d[ok], j[ok]

    i = 0
    d, j = candidates(i)
    if len(j) == 0:
        return float("nan")
    jn, total, steps = j[0], 0.0, 0
    while i + evolve < N and jn + evolve < N:
        d0 = np.linalg.norm(X[jn] - X[i])
        d1 = np.linalg.norm(X[jn + evolve] - X[i + evolve])
        if d0 > 0 and d1 > 0:
            total += math.log(d1 / d0)
            steps += evolve
        old_dir = X[jn + evolve] - X[i + evolve]
        i += evolve
        if i >= N:
            break
        if d1 > dmax or d1 <= dmin or jn + evolve >= N:
            dd, jj = candidates(i)
            jj = jj[(dd < dmax) & (jj + evolve < N)] if (dd < dmax).any() else jj[jj + evolve < N]
            if len(jj) == 0:
                break
            v = X[jj] - X[i]
            cosang = (v @ old_dir) / (np.linalg.norm(v, axis=1) * (np.linalg.norm(old_dir) + 1e-300) + 1e-300)
            jn = int(jj[np.argmax(cosang)])
        else:
            jn = jn + evolve
    return total / steps if steps else float("nan")


# ----------------------------------------------------------------- dimension
def corr_sum(x, m, tau, radii, theiler=10, n_pts=3000, seed=0):
    X = embed(x, m, tau)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), min(n_pts, len(X)), replace=False))
    P, counts, total = X[idx], np.zeros(len(radii)), 0
    for a in range(len(idx)):
        dt = np.abs(idx[a + 1:] - idx[a]) > theiler
        if not dt.any():
            continue
        d = np.linalg.norm(P[a + 1:][dt] - P[a], axis=1)
        counts += np.searchsorted(np.sort(d), radii, side="right")
        total += dt.sum()
    return counts / max(total, 1)


def correlation_dimension(x, ms=range(1, 9), tau=1, theiler=10, n_pts=3000, cfit=(1e-3, 3e-2)):
    """Grassberger-Procaccia: slope of log C(r) vs log r where cfit[0] < C < cfit[1]."""
    s = np.std(x)
    radii = s * np.logspace(-3, 0.5, 60)
    out = {}
    for m in ms:
        C = corr_sum(x, m, tau, radii, theiler, n_pts)
        ok = (C > cfit[0]) & (C < cfit[1])
        out[m] = float(np.polyfit(np.log(radii[ok]), np.log(C[ok]), 1)[0]) if ok.sum() >= 4 else float("nan")
    return out


# ----------------------------------------------------------------- entropy, 0-1 test
def permutation_entropy(x, order=4, delay=1):
    """Normalised Bandt-Pompe entropy in [0, 1]; ties ordered by time (stable sort)."""
    X = embed(x, order, delay)
    pat = np.argsort(X, axis=1, kind="stable")
    codes = (pat * (order ** np.arange(order))).sum(axis=1)
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(order)))


def zero_one_test(x, n_c=100, seed=0):
    """Gottwald-Melbourne 0-1 test (modified, correlation method): K near 1 chaos, near 0 regular."""
    x = np.asarray(x, float)
    x = x - x.mean()
    N = len(x)
    ncut = N // 10
    rng = np.random.default_rng(seed)
    j = np.arange(1, N + 1)
    ns = np.arange(1, ncut + 1)
    Ks = []
    for c in rng.uniform(math.pi / 5, 4 * math.pi / 5, n_c):
        p, q = np.cumsum(x * np.cos(j * c)), np.cumsum(x * np.sin(j * c))
        M = np.array([np.mean((p[n:] - p[:-n]) ** 2 + (q[n:] - q[:-n]) ** 2) for n in ns])
        Vosc = (x.mean() ** 2) * (1 - np.cos(ns * c)) / (1 - np.cos(c))
        D = M - Vosc
        Ks.append(np.corrcoef(ns, D)[0, 1])
    return float(np.nanmedian(Ks))


# ----------------------------------------------------------------- Hurst
def dfa(x, order=1, smin=16, smax=None, n_scales=20):
    x = np.asarray(x, float)
    Y = np.cumsum(x - x.mean())
    N = len(Y)
    smax = smax or N // 4
    scales = np.unique(np.logspace(np.log10(smin), np.log10(smax), n_scales).astype(int))
    F = []
    for s in scales:
        nseg = N // s
        seg = Y[:nseg * s].reshape(nseg, s)
        t = np.arange(s)
        V = np.vander(t, order + 1)
        coef, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
        res = seg.T - V @ coef
        F.append(np.sqrt(np.mean(res ** 2)))
    F = np.array(F)
    return scales, F


def hurst_dfa(x, order=1, smin=16, smax=None):
    s, F = dfa(x, order, smin, smax)
    return _fit_slope(np.log(s), np.log(F))


def hurst_gph(x, power=0.5):
    """Geweke-Porter-Hudak log-periodogram regression; H = d + 1/2."""
    x = np.asarray(x, float) - np.mean(x)
    N = len(x)
    m = int(N ** power)
    I = np.abs(np.fft.rfft(x)[1:m + 1]) ** 2 / (2 * math.pi * N)
    lam = 2 * math.pi * np.arange(1, m + 1) / N
    reg = -np.log(4 * np.sin(lam / 2) ** 2)
    d = _fit_slope(reg, np.log(I))
    return d + 0.5


_DB3 = np.array([0.3326705529500826, 0.8068915093110925, 0.4598775021184915,
                 -0.1350110200102546, -0.0854412738820267, 0.0352262918857095])


def _dwt_details(x, levels):
    h = _DB3
    g = np.array([(-1) ** k * h[len(h) - 1 - k] for k in range(len(h))])
    a = np.asarray(x, float)
    out = []
    for _ in range(levels):
        n = len(a) - len(a) % 2
        a = a[:n]
        idx = (np.arange(0, n, 2)[:, None] + np.arange(len(h))[None]) % n
        out.append(a[idx] @ g)
        a = a[idx] @ h
    return out


def hurst_wavelet(x, j1=3, j2=None):
    """Abry-Veitch (1998) estimator with Daubechies-3 wavelet and bias-corrected weighted fit."""
    x = np.asarray(x, float)
    J = int(np.log2(len(x))) - 3
    j2 = j2 or J
    d = _dwt_details(x, j2)
    js, y, w = [], [], []
    for j in range(j1, j2 + 1):
        dj = d[j - 1]
        nj = len(dj)
        mu = np.mean(dj ** 2)
        gj = digamma(nj / 2) / math.log(2) - math.log2(nj / 2)
        var = zeta(2, nj / 2) / math.log(2) ** 2
        js.append(j); y.append(math.log2(mu) - gj); w.append(1 / var)
    js, y, w = np.array(js, float), np.array(y), np.array(w)
    S, Sx, Sy = w.sum(), (w * js).sum(), (w * y).sum()
    Sxx, Sxy = (w * js * js).sum(), (w * js * y).sum()
    slope = (S * Sxy - Sx * Sy) / (S * Sxx - Sx ** 2)
    return (slope + 1) / 2, js, y


def hurst_aggvar(x, mmin=4, mmax=None, n=20):
    x = np.asarray(x, float)
    mmax = mmax or len(x) // 16
    ms = np.unique(np.logspace(np.log10(mmin), np.log10(mmax), n).astype(int))
    v = [np.var(x[:len(x) // m * m].reshape(-1, m).mean(axis=1), ddof=1) for m in ms]
    return _fit_slope(np.log(ms), np.log(v)) / 2 + 1


def fgn(n, H, seed=0):
    """Fractional Gaussian noise by Davies-Harte circulant embedding."""
    k = np.arange(n + 1)
    r = 0.5 * ((k + 1.0) ** (2 * H) - 2 * k ** (2.0 * H) + np.abs(k - 1.0) ** (2 * H))
    c = np.concatenate([r, r[-2:0:-1]])
    lam = np.real(np.fft.fft(c))
    lam = np.maximum(lam, 0)
    rng = np.random.default_rng(seed)
    M = len(c)
    z = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    y = np.fft.fft(np.sqrt(lam / M) * z)
    return np.real(y[:n])


# ----------------------------------------------------------------- stationarity, tails
def kpss(x, lags=None):
    """KPSS level-stationarity statistic (5% critical value 0.463)."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    S = np.cumsum(e)
    lags = lags if lags is not None else int(np.ceil(12 * (n / 100) ** 0.25))
    s2 = np.dot(e, e) / n
    for l in range(1, lags + 1):
        s2 += 2 * (1 - l / (lags + 1)) * np.dot(e[l:], e[:-l]) / n
    return float(np.dot(S, S) / (n ** 2 * s2))


def adf(x, p=20):
    """ADF t-statistic with constant and p lagged differences (5% critical value -2.86)."""
    x = np.asarray(x, float)
    dx = np.diff(x)
    y = dx[p:]
    cols = [np.ones_like(y), x[p:-1]] + [dx[p - i:-i] for i in range(1, p + 1)]
    Z = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    res = y - Z @ beta
    s2 = res @ res / (len(y) - Z.shape[1])
    cov = s2 * np.linalg.inv(Z.T @ Z)
    return float(beta[1] / math.sqrt(cov[1, 1]))


def hill(x, frac=0.01):
    """Hill estimate of the tail index of |x| from the top `frac` order statistics."""
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    a = a[a > 0]
    k = max(10, int(frac * len(a)))
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


# ----------------------------------------------------------------- self-test
def selftest():
    out = {}
    # logistic map r=4: LLE ln 2
    x = np.empty(20000); x[0] = 0.3
    for t in range(1, len(x)):
        x[t] = 4 * x[t - 1] * (1 - x[t - 1])
    # Henon: LLE 0.419, D2 ~ 1.21
    h = np.empty((30000, 2)); h[0] = (0.1, 0.1)
    for t in range(1, len(h)):
        h[t] = (1 - 1.4 * h[t - 1, 0] ** 2 + h[t - 1, 1], 0.3 * h[t - 1, 0])
    hx = h[1000:, 0]
    out["logistic_rosenstein"] = rosenstein(x, m=1, tau=1, kfit=(0, 4), n_ref=3000)[0]
    out["logistic_kantz"] = kantz(x, m=1, tau=1, kfit=(0, 4))[0]
    out["logistic_wolf"] = wolf(x, m=1, tau=1, evolve=1)
    out["henon_rosenstein"] = rosenstein(hx, m=2, tau=1, kfit=(0, 6), n_ref=3000)[0]
    out["henon_kantz"] = kantz(hx, m=2, tau=1, kfit=(0, 6))[0]
    out["henon_wolf"] = wolf(hx, m=2, tau=1, evolve=1)
    out["henon_D2"] = correlation_dimension(hx, ms=[2, 3, 4], tau=1)
    rng = np.random.default_rng(0)
    w = rng.standard_normal(20000)
    out["white_PE5"] = permutation_entropy(w, 5)
    out["logistic_PE5"] = permutation_entropy(x, 5)
    out["white_01"] = zero_one_test(w[:3000])
    out["logistic_01"] = zero_one_test(x[:3000])
    for H in (0.5, 0.7, 0.85):
        f = fgn(2 ** 16, H, seed=1)
        out[f"fgn{H}"] = dict(dfa1=hurst_dfa(f, 1), dfa2=hurst_dfa(f, 2), gph=hurst_gph(f),
                              wav=hurst_wavelet(f)[0], agg=hurst_aggvar(f))
    out["kpss_white"] = kpss(w)
    out["adf_white"] = adf(w)
    out["adf_randomwalk"] = adf(np.cumsum(w))
    out["hill_pareto1.5"] = hill(rng.pareto(1.5, 50000) + 1)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=1))
