"""What are the oscillations of the queue-TCP loop under overload?

With a constant input the model is a deterministic map in 2 + tau dimensions
(q1, q2 and the stored loss values). For the same 200 random configurations as
round2_osc.py we classify the long-run behaviour (fixed point, periodic orbit of
period P, aperiodic) and compute the largest Lyapunov exponent of the map itself
(twin trajectories). For the configurations that are chaotic we draw orbit
(bifurcation) diagrams over the load, return maps, and compare series-based
estimators (Rosenstein, Kantz, Wolf-type), the correlation dimension, permutation
entropy and the 0-1 test with the model exponent.

Output: data_v2/overload_dynamics.json, data_v2/overload_orbits.npz
"""
import os
import json
import numpy as np
import fastnet as F
import nonlin as NL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
LOADS = (30., 40., 50., 52., 60., 70., 80., 90.)
STEPS, KEEP = 8000, 2000
TOL = 1e-6


def configs():
    out = []
    for i in range(200):
        rng = np.random.default_rng(800000 + i)
        bits = rng.integers(0, 2, 14)
        out.append(int("".join(str(int(b)) for b in bits), 2))
    return np.array(out)


def period(Q, pmax=400):
    """Smallest P <= pmax with |q(t+P) - q(t)| < TOL over the kept window, else 0 (aperiodic)."""
    n = Q.shape[1]
    P = np.zeros(n, int)
    for p in range(1, pmax + 1):
        d = np.max(np.abs(Q[p:] - Q[:-p]), axis=0)
        new = (P == 0) & (d < TOL)
        P[new] = p
    return P


def classify(idx, lam, tau):
    a = np.full(STEPS, lam)
    x = np.full(STEPS, 15.0)
    Q1, _ = F.simulate(idx, a, x, tau=tau)
    Q = Q1[-KEEP:]
    osc = np.ptp(Q[-300:], axis=0) > 1.0
    P = period(Q)
    lle = F.twin_lle(idx, a, x, tau=tau, burn=STEPS - 5000)
    return Q, osc, P, lle


def main():
    idx = configs()
    res, cls = {}, {}
    for tau in (5, 40):
        for lam in LOADS:
            Q, osc, P, lle = classify(idx, lam, tau)
            fixed = P == 1
            periodic = osc & (P >= 2)
            aper = osc & (P == 0)
            chaotic = aper & (lle > 0.01)
            key = f"tau{tau}_lam{int(lam)}"
            res[key] = dict(n=len(idx), osc=int(osc.sum()), fixed=int(fixed.sum()),
                            small_cycle=int((~osc & (P != 1)).sum()),
                            periodic=int(periodic.sum()), periods=sorted(set(P[periodic].tolist())),
                            aperiodic=int(aper.sum()), chaotic=int(chaotic.sum()),
                            lle_chaotic=[float(v) for v in np.sort(lle[chaotic])],
                            lle_periodic_max=float(lle[periodic].max()) if periodic.any() else None,
                            lle_fixed_max=float(lle[fixed].max()) if fixed.any() else None)
            cls[key] = dict(P=P.tolist(), lle=lle.tolist(), osc=osc.tolist())
            print(key, {k: v for k, v in res[key].items() if k != "lle_chaotic"})
    # pick the most chaotic configurations at load 90, tau 5
    c = cls["tau5_lam90"]
    lle90 = np.array(c["lle"]); P90 = np.array(c["P"]); osc90 = np.array(c["osc"])
    cand = np.where(osc90 & (P90 == 0) & (lle90 > 0.01))[0]
    order = cand[np.argsort(-lle90[cand])]
    picks = [int(idx[k]) for k in order[:3]]
    res["picked_configs"] = picks
    # orbit diagrams over load for the picked configurations
    lams = np.round(np.arange(40.0, 100.01, 0.1), 2)
    orbits = {}
    for p in picks:
        rows = np.full(len(lams), p)
        a = np.tile(lams, (STEPS, 1))
        x = np.full((STEPS, len(lams)), 15.0)
        Q1, _ = F.simulate(rows, a, x, tau=5)
        lle = F.twin_lle(rows, a, x, tau=5, burn=STEPS - 5000)
        orbits[f"cfg{p}_q"] = Q1[-300:].T.astype(np.float32)
        orbits[f"cfg{p}_lle"] = lle
        Pq = period(Q1[-KEEP:])
        # first load at which period doubling (P=2) and aperiodicity appear
        res[f"cfg{p}_bif"] = dict(first_P2=float(lams[np.argmax(Pq == 2)]) if (Pq == 2).any() else None,
                                  first_P4=float(lams[np.argmax(Pq == 4)]) if (Pq == 4).any() else None,
                                  first_aperiodic=float(lams[np.argmax(Pq == 0)]) if (Pq == 0).any() else None,
                                  periods_seen=sorted(set(Pq.tolist())),
                                  frac_chaotic=float(((Pq == 0) & (lle > 0.01)).mean()))
    orbits["lams"] = lams
    # series diagnostics for the most chaotic configuration at load 90
    diag = {}
    for p in picks:
        a = np.full(40000, 90.0)
        x = np.full(40000, 15.0)
        Q1, _ = F.simulate(np.array([p]), a, x, tau=5)
        q = Q1[5000:, 0]
        true = float(F.twin_lle(np.array([p]), a, x, tau=5, burn=5000)[0])
        d = dict(model_lle=true,
                 rosenstein=NL.rosenstein(q, m=7, tau=1, kfit=(0, 5), n_ref=4000)[0],
                 kantz=NL.kantz(q, m=7, tau=1, kfit=(0, 5))[0],
                 wolf=NL.wolf(q, m=7, tau=1, evolve=2),
                 short_indicator=float(F.NetworkSystem._estimate_lyapunov(None, q[:200])),
                 D2=NL.correlation_dimension(q, ms=range(1, 11), tau=1),
                 PE={o: NL.permutation_entropy(q, o) for o in (3, 4, 5, 6)},
                 PE_shuffled={o: NL.permutation_entropy(np.random.default_rng(0).permutation(q), o) for o in (3, 4, 5, 6)},
                 K01=NL.zero_one_test(q[:5000]))
        diag[str(p)] = d
        orbits[f"cfg{p}_series90"] = q[:5000].astype(np.float32)
        print(p, json.dumps(d))
    res["series_diagnostics"] = diag
    json.dump(res, open(os.path.join(DATA, "overload_dynamics.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DATA, "overload_orbits.npz"), **orbits)


if __name__ == "__main__":
    main()
