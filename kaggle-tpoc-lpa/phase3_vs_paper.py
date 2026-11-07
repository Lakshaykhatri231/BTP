"""One benchmark, four configurations -- for the Phase 1 (base paper) vs Phase 3 comparison.

    python3 phase3_vs_paper.py lfr <n> <mu> <seed>
    python3 phase3_vs_paper.py snap <amazon|dblp|youtube>

Uses the SAME loaders as Phase 1 (SNAP graphs are the full induced subgraph,
NOT reduced to the largest component), so numbers are directly comparable to
the Phase 1 results table.
"""
import sys
import time
from collections import defaultdict

from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity

from tpoc_lpa import tpoc_lpa

CONFIGS = [
    ("base", dict(sim="jaccard", agg="sum", guard=False)),
    ("motif", dict(sim="tri_overlap", agg="sum", guard=False)),
    ("guard", dict(sim="jaccard", agg="sum", guard=True)),
    ("both", dict(sim="tri_overlap", agg="sum", guard=True)),
]


def score(G, truth, **kw):
    nodes = list(G.nodes())
    lab, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8, **kw)
    g = defaultdict(set)
    for n in nodes:
        g[lab[n]].add(n)
    nmi = normalized_mutual_info_score([truth[n] for n in nodes], [lab[n] for n in nodes])
    return nmi, modularity(G, list(g.values())), len(g)


def main():
    kind = sys.argv[1]
    if kind == "lfr":
        from lfr_sweep import generate_lfr
        n, mu, seed = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
        G, truth = generate_lfr(n=n, mu=mu, seed=seed)
        tag = f"lfr n={n} mu={mu} seed={seed}"
    else:
        from load_snap import load_snap_network
        G, truth, _ = load_snap_network(sys.argv[2])
        truth = {n: str(v) for n, v in truth.items()}
        tag = f"snap {sys.argv[2]}"
    out = []
    for name, kw in CONFIGS:
        t0 = time.time()
        nmi, q, nc = score(G, truth, **kw)
        out.append(f"{name}={nmi:.4f}/{q:.4f}/c{nc}")
    print(f"RESULT | {tag} | nodes={G.number_of_nodes()} | " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
