"""
Time-complexity analysis: how long each Stage 1/2/3 step (and each opt-in change)
actually takes, and how that scales with graph size.

Usage:
    python3 time_complexity.py stages     # Part A: per-stage timing, Phase 5 config, across all graphs
    python3 time_complexity.py configs    # Part B: base vs Phase 4 vs Phase 5 total time, dev real graphs
    python3 time_complexity.py lfr        # Part C: clean n-scaling sweep on LFR graphs (fixed mu)
    (no args -> runs all three)

Uses tpoc_lpa's `info=` dict (see its docstring) to read back wall-clock seconds
per pipeline stage. Passing `info` has zero effect on the returned labels --
verified by hash-comparing outputs against this session's pre-instrumentation runs.
"""
import sys
import time

import datasets
from load_snap import load_snap_network
from lfr_sweep import generate_lfr
from phase4_ablation import load_npz_graph
from tpoc_lpa import tpoc_lpa

PHASE5 = dict(sim="tri_overlap", guard=True, refine="granularity", merge=True, merge_min_gain=0.03)
PHASE5_SNAP = dict(sim="tri_overlap", guard=True, refine="granularity", merge=True, merge_min_gain=0.01)
PHASE4 = dict(sim="tri_overlap", guard=True, refine="granularity")
BASE = dict()

STAGE_KEYS = ["t_node_importance", "t_stage1", "t_stage2", "t_refine", "t_merge", "t_stage3"]


def run_once(G, kw):
    info = {}
    t0 = time.perf_counter()
    tpoc_lpa(G, alpha=1.0, beta=0.8, info=info, **kw)
    wall = time.perf_counter() - t0
    return info, wall


REAL_GRAPHS = [
    ("karate", datasets.karate),
    ("dolphin", datasets.dolphin),
    ("polbooks", datasets.polbooks),
    ("football", datasets.football),
    ("email-Eu", datasets.email_eu),
    ("polblogs", datasets.polblogs),
    ("citeseer", datasets.citeseer),
    ("cora", datasets.cora),
]
HELDOUT_GRAPHS = [
    ("amazon_photo", lambda: load_npz_graph("amazon_photo")),
    ("amazon_computers", lambda: load_npz_graph("amazon_computers")),
    ("coauthor_cs", lambda: load_npz_graph("coauthor_cs")),
    ("coauthor_physics", lambda: load_npz_graph("coauthor_physics")),
]
SNAP_GRAPHS = [
    ("SNAP amazon", lambda: _snap("amazon")),
    ("SNAP youtube", lambda: _snap("youtube")),
    ("SNAP dblp", lambda: _snap("dblp")),
]


def _snap(name):
    G, truth, _ = load_snap_network(name)
    return G, {n: str(v) for n, v in truth.items()}


def part_a_stages():
    print("PART A: per-stage wall-clock time (seconds), Phase 5 config (motif+guard+granularity+merge)")
    print(f"{'graph':<18}{'n':>8}{'m':>9}" + "".join(f"{k.replace('t_', ''):>10}" for k in STAGE_KEYS) + f"{'total':>10}")
    rows = []
    for label, loader in REAL_GRAPHS + HELDOUT_GRAPHS + SNAP_GRAPHS:
        G, _ = loader()
        kw = PHASE5_SNAP if label.startswith("SNAP") else PHASE5
        info, wall = run_once(G, kw)
        rows.append((label, G.number_of_nodes(), G.number_of_edges(), info, wall))
        print(f"{label:<18}{G.number_of_nodes():>8}{G.number_of_edges():>9}" +
              "".join(f"{info[k]:>10.4f}" for k in STAGE_KEYS) + f"{wall:>10.4f}", flush=True)
    return rows


def part_b_configs():
    print("\nPART B: total wall-clock time (seconds) by config, dev real graphs (single run each)")
    print(f"{'graph':<12}{'n':>8}{'m':>9}{'base':>10}{'phase4':>10}{'phase5':>10}"
          f"{'phase4/base':>13}{'phase5/base':>13}")
    for label, loader in REAL_GRAPHS:
        G, _ = loader()
        _, t_base = run_once(G, BASE)
        _, t_p4 = run_once(G, PHASE4)
        _, t_p5 = run_once(G, PHASE5)
        print(f"{label:<12}{G.number_of_nodes():>8}{G.number_of_edges():>9}"
              f"{t_base:>10.4f}{t_p4:>10.4f}{t_p5:>10.4f}"
              f"{t_p4 / t_base:>12.2f}x{t_p5 / t_base:>12.2f}x", flush=True)


def part_c_lfr():
    print("\nPART C: LFR size sweep at fixed mu=0.3, Phase 5 config, seed 1")
    print(f"{'n':>8}{'m':>9}" + "".join(f"{k.replace('t_', ''):>10}" for k in STAGE_KEYS) + f"{'total':>10}")
    for n in (150, 1000, 5000, 10000):
        G, _ = generate_lfr(n=n, mu=0.3, seed=1)
        info, wall = run_once(G, PHASE5)
        print(f"{n:>8}{G.number_of_edges():>9}" +
              "".join(f"{info[k]:>10.4f}" for k in STAGE_KEYS) + f"{wall:>10.4f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("stages", "all"):
        part_a_stages()
    if mode in ("configs", "all"):
        part_b_configs()
    if mode in ("lfr", "all"):
        part_c_lfr()
