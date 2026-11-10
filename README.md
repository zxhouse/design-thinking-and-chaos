# Test Twice, Ship Once: code and data

This repository contains the code and data needed to reproduce every number and figure in
"Test Twice, Ship Once: The Optimizer's Curse in Iterative Tuning of Queue Configurations
under Chaotic and Heavy-Tailed Traffic" (Kuraś, Strzałka, Misiura).

Code is released under the MIT licence (see `LICENSE`); the data in `chaos_project/data_v2` may be reused with attribution to the article.

Tested with Python 3.11, NumPy 2.4 and Matplotlib 3.11 on macOS.

## Layout

```
chaos_project/src/network_sim.py   two-switch network model (Section 3)
chaos_project/src_v2/              experiments, analysis and figures
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
```

## Full re-run (about 2 hours on 10 cores)

This overwrites `data_v2/`. Seeds are fixed in the scripts, so the results should match
the shipped files up to floating-point differences between platforms.

```bash
cd chaos_project/src_v2
python3 experiments_v2.py all   # ground truth, dynamics, allocation, regimes, ablations, AQM
python3 round2_v2.py            # balanced policy in the other regimes
python3 round2_osc.py           # endogenous-oscillation sweep (Figure 2c)
```

then run the quick check above.

## Where each result comes from

| Manuscript | Script | Data |
|---|---|---|
| Section 5.1, Fig. 2 | `experiments_v2.py dynamics`, `round2_osc.py` | `dynamics.json`, `round2_osc.json` |
| Section 5.2, Table 2, Fig. 3 | `experiments_v2.py ground_truth`, `separation_v2.py` | `gt_*.npy` |
| Sections 5.3-5.4, Table 3, Fig. 4 | `experiments_v2.py allocation` | `allocation_v2.json` |
| Section 5.5, Table 4 | `experiments_v2.py allocation`/`aqm`, `round2_v2.py` | `regimes_v2.json`, `aqm_v2.json`, `round2_balanced.json` |
| Section 5.6 | `experiments_v2.py ablations` | `ablations_v2.json` |
