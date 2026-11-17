"""Share of steps with an empty edge queue, flagged vs. unflagged configurations (one ground-truth seed).

Output: data_v2/empty_queue.json
"""
import os
import json
import numpy as np
import fastnet as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_v2")
ALL = np.arange(2 ** 14)
out = {}
for mode in ("chaos", "poisson", "self_similar"):
    g = np.load(os.path.join(DATA, f"gt_{mode}_orig.npy"))
    flagged = g[:, 3] > 0.5
    a, x = F.inputs(mode, 900001, 400)
    Q1, _ = F.simulate(ALL, a, x)
    empty = (Q1[50:] < 1e-9).mean(axis=0)
    out[mode] = dict(median_empty_flagged=float(np.median(empty[flagged])),
                     median_empty_unflagged=float(np.median(empty[~flagged])))
json.dump(out, open(os.path.join(DATA, "empty_queue.json"), "w"), indent=1)
print(out)
