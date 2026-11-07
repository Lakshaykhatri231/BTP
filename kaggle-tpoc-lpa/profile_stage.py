"""
Function-level profiling: the per-stage timestamps in tpoc_lpa's `info` dict (and
time_complexity.py) say WHICH pipeline stage is slow but not WHICH FORMULA inside
it is responsible. This uses Python's built-in cProfile -- no changes to tpoc_lpa.py
needed -- to go one level deeper on a chosen graph.

Usage:
    python3 profile_stage.py snap_amazon        # Stage-1-dominant case (see Section 6.7)
    python3 profile_stage.py coauthor_physics   # Stage-2-dominant case
"""
import cProfile
import pstats
import sys

import load_snap
from phase4_ablation import load_npz_graph
from tpoc_lpa import tpoc_lpa

PHASE5 = dict(sim="tri_overlap", guard=True, refine="granularity", merge=True, merge_min_gain=0.03)

which = sys.argv[1]
if which == "snap_amazon":
    G, truth, _ = load_snap.load_snap_network("amazon")
elif which == "coauthor_physics":
    G, truth = load_npz_graph("coauthor_physics")
else:
    raise SystemExit(f"unknown graph {which}")

print(f"{which}: n={G.number_of_nodes()} m={G.number_of_edges()}")

pr = cProfile.Profile()
pr.enable()
tpoc_lpa(G, alpha=1.0, beta=0.8, **PHASE5)
pr.disable()

stats = pstats.Stats(pr)
stats.sort_stats("cumulative")
print(f"\n--- top 15 by CUMULATIVE time ({which}) ---")
stats.print_stats(15)

stats.sort_stats("tottime")
print(f"\n--- top 15 by TOTAL (self) time ({which}) ---")
stats.print_stats(15)
