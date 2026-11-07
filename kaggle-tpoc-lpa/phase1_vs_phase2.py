"""
Has Phase 2 improved the pipeline on the PHASE 1 benchmarks?

Compares, on every Phase 1 benchmark that has ground truth:
  * Phase 1 pipeline (TPOC-LPA, Stages 1-3)                       -> baseline
  * Phase 2a: deterministic tie-breaking                          -> same or not?
  * Phase 2b: Stage 4 with a REAL LLM                             -> only where nodes carry text
  * LLM-free re-decision (WAS argmax, the b=0 control found in Phase 2) -> applicable to every graph

Stage 4 with a real LLM needs semantic node text. Of the Phase 1 benchmarks only Football has it
(Karate/LFR/SNAP nodes are integers; Dolphin nodes are dolphin names the model cannot know).
"""
import random
import sys
import time

import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity

from tpoc_lpa import tpoc_lpa, node_importance
from stage4 import was_distribution
from dolphin_ground_truth import dolphin_truth_by_label


def to_partition(label, nodes):
    g = {}
    for n in nodes:
        g.setdefault(label[n], set()).add(n)
    return list(g.values())


def score(G, truth, label):
    nodes = list(G.nodes())
    return (normalized_mutual_info_score([truth[n] for n in nodes], [label[n] for n in nodes]),
            modularity(G, to_partition(label, nodes)))


def redecide(G, NI, base):
    """One extra pass: each boundary node takes the neighbouring community with the
    highest weighted affiliation WAS (Eq. 14), computed from the initial labels."""
    new = dict(base)
    for i in G.nodes():
        dist = was_distribution(G, NI, base, i)
        if len(dist) >= 2:
            new[i] = max(dist, key=lambda c: (dist[c], str(c)))
    return new


def run(name, G, truth):
    t0 = time.time()
    NI = node_importance(G)
    base, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8)
    b_nmi, b_q = score(G, truth, base)

    # Phase 2a: random tie-breaking (the paper's behaviour) vs our deterministic default
    rnd = []
    for s in range(5):
        lab, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8, rng=random.Random(100 + s))
        rnd.append(score(G, truth, lab)[0])
    det_same = all(abs(x - b_nmi) < 1e-12 for x in rnd)

    r = redecide(G, NI, base)
    r_nmi, r_q = score(G, truth, r)
    n_changed = sum(1 for n in G.nodes() if r[n] != base[n])
    print(f"{name:<22} n={G.number_of_nodes():<6} | Phase 1: NMI {b_nmi:.4f} Q {b_q:.4f} | "
          f"det-ties identical to random-ties: {det_same} | "
          f"WAS re-decision: NMI {r_nmi:.4f} ({r_nmi-b_nmi:+.4f}) Q {r_q:.4f} ({r_q-b_q:+.4f}), "
          f"{n_changed} nodes moved | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    print("PHASE 1 benchmarks: Phase 1 pipeline vs Phase 2 components\n", flush=True)

    G = nx.karate_club_graph()
    run("Karate", G, {n: (0 if G.nodes[n]['club'] == 'Mr. Hi' else 1) for n in G.nodes()})

    G = nx.read_gml("data/dolphins/dolphins.gml")
    tbl = dolphin_truth_by_label()
    run("Dolphin", G, {n: tbl[n] for n in G.nodes()})

    G = nx.read_gml("data/football/football.gml", label="id")
    run("Football", G, {n: G.nodes[n]["value"] for n in G.nodes()})

    from lfr_sweep import generate_lfr
    for n_, mu in [(1000, 0.1), (1000, 0.6), (10000, 0.1), (10000, 0.6)]:
        G, truth = generate_lfr(n=n_, mu=mu)
        run(f"LFR n={n_} mu={mu}", G, truth)

    if "--big" in sys.argv:
        from load_snap import load_snap_network
        for name in ("amazon", "youtube", "dblp"):
            G, truth, _ = load_snap_network(name)
            run(f"SNAP {name}", G, {n: str(truth[n]) for n in G.nodes()})
