"""
Total wall-clock time per project phase, on the same nine graphs used in the
report's Phase-by-Phase NMI table (Section 6.6), so time and accuracy can be
read side by side.

    python3 time_by_phase.py

Phases: 1 = base (paper) | 3 = motif+guard | 4 = +granularity selector |
        5 = +merge (eps=0.01 on SNAP graphs, eps=0.03 elsewhere, matching
        the rest of the project's convention).
"""
import statistics as st
import time

import datasets
from load_snap import load_snap_network
from lfr_sweep import generate_lfr
from phase4_ablation import load_npz_graph
from tpoc_lpa import tpoc_lpa

PHASE1 = dict()
PHASE3 = dict(sim="tri_overlap", guard=True)
PHASE4 = dict(sim="tri_overlap", guard=True, refine="granularity")
PHASE5 = dict(sim="tri_overlap", guard=True, refine="granularity", merge=True, merge_min_gain=0.03)
PHASE5_SNAP = dict(sim="tri_overlap", guard=True, refine="granularity", merge=True, merge_min_gain=0.01)


def timed(G, kw):
    t0 = time.perf_counter()
    tpoc_lpa(G, alpha=1.0, beta=0.8, **kw)
    return time.perf_counter() - t0


def snap(name):
    G, truth, _ = load_snap_network(name)
    return G, {n: str(v) for n, v in truth.items()}


GRAPHS = [
    ("Karate", datasets.karate),
    ("Football", datasets.football),
    ("email-Eu-core", datasets.email_eu),
    ("Cora", datasets.cora),
    ("Amazon (SNAP)", lambda: snap("amazon")),
    ("DBLP (SNAP)", lambda: snap("dblp")),
    ("Youtube (SNAP)", lambda: snap("youtube")),
    ("Coauthor-Physics", lambda: load_npz_graph("coauthor_physics")),
]

print(f"{'graph':<18}{'n':>8}{'m':>9}{'Phase 1':>10}{'Phase 3':>10}{'Phase 4':>10}{'Phase 5':>10}")
for label, loader in GRAPHS:
    G, _ = loader()
    p5 = PHASE5_SNAP if "SNAP" in label else PHASE5
    t1 = timed(G, PHASE1)
    t3 = timed(G, PHASE3)
    t4 = timed(G, PHASE4)
    t5 = timed(G, p5)
    print(f"{label:<18}{G.number_of_nodes():>8}{G.number_of_edges():>9}"
          f"{t1:>10.4f}{t3:>10.4f}{t4:>10.4f}{t5:>10.4f}", flush=True)

# LFR mu=0.7, 5 dev seeds (1-5), mean -- matches the NMI table's row exactly
rows = {1: [], 3: [], 4: [], 5: []}
n_nodes = n_edges = 0
for seed in (1, 2, 3, 4, 5):
    G, _ = generate_lfr(n=1000, mu=0.7, seed=seed)
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    rows[1].append(timed(G, PHASE1))
    rows[3].append(timed(G, PHASE3))
    rows[4].append(timed(G, PHASE4))
    rows[5].append(timed(G, PHASE5))
print(f"{'LFR mu=0.7 (5 seeds)':<18}{n_nodes:>8}{n_edges:>9}"
      f"{st.mean(rows[1]):>10.4f}{st.mean(rows[3]):>10.4f}{st.mean(rows[4]):>10.4f}{st.mean(rows[5]):>10.4f}")
