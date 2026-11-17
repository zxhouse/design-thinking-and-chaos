"""Vectorised re-implementation of the two-switch model in ../src/network_sim.py.

One call advances many networks at once: one row per configuration (or per load
value), all driven by the same or by row-specific arrival series. Arrival series
do not depend on the network state, so they can be generated in advance with the
original TrafficGenerator. `check()` confirms that trajectories match
network_sim.NetworkSystem to floating-point precision.

The module also computes the largest Lyapunov exponent of the model itself
(not from a time series) by the twin-trajectory (Benettin) method: a copy of every
network is started at a distance delta in the full state space (q1, q2 and the
tau stored loss values), driven by the same input, and renormalised after every
step. For a stochastic input this is the conditional (random-dynamical-system)
exponent: positive means that two copies of the same network fed the same traffic
diverge, negative means that they synchronise.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from network_sim import TrafficGenerator, NetworkSystem  # noqa: E402

B1, B2 = 120.0, 140.0


def params(idx):
    """Model parameters for configuration indices (array of ints, bit 0 = most significant)."""
    idx = np.asarray(idx)
    bits = (idx[:, None] >> (13 - np.arange(14))) & 1
    p = bits.astype(float)
    mu1 = 50.0 + 3.5 * (p[:, 0] * 4 + p[:, 1] * 2 + p[:, 2])
    th1 = 40.0 + 5.5 * (p[:, 3] * 4 + p[:, 4] * 2 + p[:, 5])
    sl1 = 0.05 + 0.10 * (p[:, 6] * 2 + p[:, 7])
    mu2 = 55.0 + 3.5 * (p[:, 8] * 4 + p[:, 9] * 2 + p[:, 10])
    th2 = 45.0 + 5.5 * (p[:, 11] * 4 + p[:, 12] * 2 + p[:, 13])
    sl2 = np.full_like(mu1, 0.12)
    return mu1, th1, sl1, mu2, th2, sl2


def _drop(q, th, sl, zero_floor):
    p = 1.0 / (1.0 + np.exp(-np.clip(sl * (q - th), -30.0, 30.0)))
    if zero_floor:
        p0 = 1.0 / (1.0 + np.exp(-np.clip(sl * (0.0 - th), -30.0, 30.0)))
        p = np.maximum(0.0, (p - p0) / (1.0 - p0))
    return p


def _node(q, a, mu, th, sl, cap, zero_floor):
    p = _drop(q, th, sl, zero_floor)
    acc = a * (1.0 - p)
    drop = a * p
    avail = cap - q
    over = acc > avail
    drop = np.where(over, drop + (acc - avail), drop)
    acc = np.where(over, np.maximum(0.0, avail), acc)
    q = q + acc
    dep = np.minimum(q, mu)
    return q - dep, dep, drop


class Batch:
    """State of many networks: q1, q2 (n,) and loss buffer L (n, tau)."""

    def __init__(self, idx, tau=5, zero_floor=False):
        self.par = params(idx)
        n = len(np.asarray(idx))
        self.tau, self.zf = tau, zero_floor
        self.q1, self.q2 = np.zeros(n), np.zeros(n)
        self.L = np.zeros((n, tau))
        self.k = 0  # position of the oldest loss value

    def state(self):
        return np.column_stack([self.q1, self.q2, self.L])

    def set_state(self, x):
        self.q1, self.q2, self.L = x[:, 0].copy(), x[:, 1].copy(), x[:, 2:].copy()

    def step(self, a_raw, cross):
        mu1, th1, sl1, mu2, th2, sl2 = self.par
        pacing = np.maximum(0.4, 1.0 - 1.8 * self.L.mean(axis=1))
        a1 = a_raw * pacing
        self.q1, dep1, d1 = _node(self.q1, a1, mu1, th1, sl1, B1, self.zf)
        a2 = dep1 + cross
        self.q2, dep2, d2 = _node(self.q2, a2, mu2, th2, sl2, B2, self.zf)
        loss = (d1 + d2) / np.maximum(1.0, a1 + cross)
        # the original pops the oldest value and appends the newest; the mean is order-free
        self.L[:, self.k] = loss
        self.k = (self.k + 1) % self.tau
        return dep2, d1 + d2, a1 + cross


def inputs(mode, seed, steps, mean_rate=52.0, hurst=0.82):
    """Arrival and cross-traffic series exactly as NetworkSystem would draw them."""
    g = TrafficGenerator(mode=mode, mean_rate=mean_rate, hurst=hurst, seed=seed)
    c = TrafficGenerator(mode=mode, mean_rate=15.0, hurst=hurst, seed=seed + 100)
    a = np.array([g.step() for _ in range(steps)], dtype=float)
    x = np.array([c.step() for _ in range(steps)], dtype=float)
    return a, x


def simulate(idx, a, x, tau=5, zero_floor=False, record=True):
    """Run all configurations on the same input. a, x: (steps,) or (steps, n)."""
    b = Batch(idx, tau, zero_floor)
    steps = a.shape[0]
    Q1 = np.zeros((steps, len(b.q1))) if record else None
    for t in range(steps):
        b.step(a[t], x[t])
        if record:
            Q1[t] = b.q1
    return Q1, b


def twin_lle(idx, a, x, tau=5, zero_floor=False, burn=50, delta=1e-7, seed=0, floor=-18.0):
    """Largest Lyapunov exponent of the model (twin trajectories, renormalised every step).

    Per-step log-stretch is floored at `floor` (a perturbation can be annihilated
    exactly when a queue hits zero or the buffer limit; the exponent is then -inf
    and the floor keeps averages finite). Returns the mean log-stretch per step after
    `burn` steps, one value per row.
    """
    rng = np.random.default_rng(seed)
    A, Bt = Batch(idx, tau, zero_floor), Batch(idx, tau, zero_floor)
    n = len(A.q1)
    acc = np.zeros(n)
    cnt = 0
    for t in range(a.shape[0]):
        if t == burn:
            u = rng.standard_normal((n, tau + 2))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            Bt.set_state(A.state() + delta * u)
            Bt.k = A.k
        A.step(a[t], x[t])
        Bt.step(a[t], x[t])
        if t >= burn:
            d = Bt.state() - A.state()
            nd = np.linalg.norm(d, axis=1)
            dead = nd < 1e-300
            acc += np.where(dead, floor, np.maximum(floor, np.log(np.maximum(nd, 1e-300) / delta)))
            cnt += 1
            # renormalise; re-seed a random direction where the perturbation vanished
            if dead.any():
                r = rng.standard_normal((int(dead.sum()), tau + 2))
                d[dead] = r
                nd[dead] = np.linalg.norm(r, axis=1)
            Bt.set_state(A.state() + delta * d / nd[:, None])
    return acc / max(cnt, 1)


def check(n=40, steps=300):
    """Compare with network_sim for random configurations, regimes, delays and AQM variants."""
    sys.path.insert(0, HERE)
    import experiments_v2 as E
    rng = np.random.default_rng(1)
    worst = 0.0
    for k in range(n):
        i = int(rng.integers(0, 2 ** 14))
        mode = ["poisson", "chaos", "self_similar"][k % 3]
        tau = [1, 5, 40][k % 3]
        zf = bool(k % 2)
        seed = 1000 + k
        net = E.make_net(E.idx_to_cfg(i), mode, seed, aqm="zero_floor" if zf else "orig", tau=tau)
        ref = np.array(net.simulate(steps=steps)["q1_trajectory"])
        a, x = inputs(mode, seed, steps)
        Q1, _ = simulate(np.array([i]), a, x, tau, zf)
        worst = max(worst, float(np.max(np.abs(Q1[:, 0] - ref))))
    return worst


if __name__ == "__main__":
    print("max |q1 difference| vs network_sim:", check())
