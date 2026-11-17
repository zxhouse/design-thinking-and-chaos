# Test Twice, Ship Once: code and data

This repository contains the code and data needed to reproduce every number and figure in
"Test Twice, Ship Once: The Optimizer's Curse in Iterative Tuning of Queue Configurations
under Chaotic and Heavy-Tailed Traffic" (Kuraś, Strzałka, Misiura).

Code is released under the MIT licence (see `LICENSE`); the data in `chaos_project/data_v2` may be reused with attribution to the article.

Tested with Python 3.11, NumPy 2.4, SciPy 1.17, scikit-learn 1.9 and Matplotlib 3.11 on macOS.

## Layout

```
chaos_project/src/network_sim.py   two-switch network model (Section 3)
chaos_project/src_v2/              experiments, analysis and figures
    fastnet.py                     vectorised copy of the model + model Lyapunov exponent (twin trajectories)
    nonlin.py                      Rosenstein, Kantz, Wolf, correlation dimension, permutation entropy,
                                   0-1 test, DFA, wavelet and periodogram Hurst estimators, KPSS, ADF, Hill
chaos_project/data_v2/             raw results (JSON) and ground-truth tables (NumPy)
results/                           summary_v2.json, separation_v2.json and figures (PDF)
```

Ground-truth tables `gt_{traffic}_{aqm}.npy` have one row per configuration (16,384 rows,
row index = integer value of the 14-bit configuration) and five columns: mean performance,
mean penalised utility, mean Lyapunov estimate, fraction of runs with estimate > 0.05, and
standard deviation of performance (16 seeds x 400 steps each).

## Quick check (about a minute, uses the shipped raw results)

```bash
pip install -r requirements.txt
cd chaos_project/src_v2
python3 analyze_v2.py        # summary_v2.json, Figures 2-4
python3 separation_v2.py     # separation_v2.json (Table 2)
python3 fig_schema_v2.py     # Figure 1
python3 fig_dynamics_extra.py # Figures 3-5
python3 fastnet.py           # checks the vectorised model against the original simulator
python3 nonlin.py            # checks every estimator on systems with known values
```

## Full re-run (about 2 hours on 10 cores)

This overwrites `data_v2/`. Seeds are fixed in the scripts, so the results should match
the shipped files up to floating-point differences between platforms.

```bash
cd chaos_project/src_v2
python3 experiments_v2.py all   # ground truth, dynamics, allocation, regimes, ablations, AQM
python3 round2_v2.py            # balanced policy in the other regimes
python3 round2_osc.py           # endogenous-oscillation sweep (Figure 2c)
python3 overload_dynamics.py    # periods, model exponents, orbit diagrams under overload (~1 min)
python3 indicator_check.py      # model exponent for all configurations, Kantz/Wolf, separation (~5 min)
python3 input_queue_dependence.py  # input vs queue indicator, 400 seeds
python3 hurst_analysis.py       # stationarity, tails and Hurst index of long series (~2 min)
python3 stability_theory.py     # fixed points, Jacobians and characteristic roots, Propositions 1-2 (~1 min)
python3 stability_bound.py      # upper bound on the model exponent, Proposition 3 (~6 min; run after stability_theory.py)
python3 empty_queue.py
```

then run the quick check above.

## Where each result comes from

| Manuscript | Script | Data |
|---|---|---|
| Section 5.1, Figs. 2-3 | `experiments_v2.py dynamics`, `round2_osc.py`, `overload_dynamics.py`, `indicator_check.py` | `dynamics.json`, `round2_osc.json`, `overload_dynamics.json`, `overload_orbits.npz`, `indicator_check.json` |
| Section 3 (Propositions 1-3), Appendix A, Sections 5.1-5.2 | `stability_theory.py`, `stability_bound.py` | `stability_theory.json`, `stability_fixed_*.npy`, `stability_bound.json`, `stability_bound_*.npy` |
| Section 5.2, Fig. 4 | `indicator_check.py`, `input_queue_dependence.py`, `empty_queue.py` | `indicator_check.json`, `model_lle_*.npy`, `input_queue_dependence.json`, `empty_queue.json` |
| Section 5.3, Table 2, Fig. 5 | `hurst_analysis.py` | `hurst_analysis.json`, `hurst_curves.npz` |
| Section 5.4, Table 3, Fig. 6 | `experiments_v2.py ground_truth`, `separation_v2.py`, `indicator_check.py` | `gt_*.npy`, `model_lle_*.npy` |
| Sections 5.5-5.6, Table 4, Fig. 7 | `experiments_v2.py allocation` | `allocation_v2.json` |
| Section 5.7, Table 5 | `experiments_v2.py allocation`/`aqm`, `round2_v2.py` | `regimes_v2.json`, `aqm_v2.json`, `round2_balanced.json` |
| Section 5.8 | `experiments_v2.py ablations` | `ablations_v2.json` |
