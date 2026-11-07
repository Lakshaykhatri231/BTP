"""Phase 4 vs Phase 3 vs Phase 1 vs the paper, on the paper's own benchmarks.

    python3 phase4_vs_paper.py snap <amazon|dblp|youtube>
    python3 phase4_vs_paper.py real <karate|dolphin|football|...>
    python3 phase4_vs_paper.py lfr <n> <mu> <seed>

Configs: base (paper) | phase3 (motif+guard) | refine (motif+guard+refine=granularity) |
         refine+merge (adds the gated merge pass, eps=0.01)
Prints RESULT | tag | nodes | name=NMI/Q/#comm/refined ...
"""
import sys
from collections import defaultdict

from sklearn.metrics import normalized_mutual_info_score as nmi_score
from networkx.algorithms.community.quality import modularity

from tpoc_lpa import tpoc_lpa

CONFIGS = [
    ("base", dict()),
    ("phase3", dict(sim="tri_overlap", guard=True)),
    ("refine", dict(sim="tri_overlap", guard=True, refine="granularity")),
    ("refine+merge", dict(sim="tri_overlap", guard=True, refine="granularity",
                          merge=True, merge_min_gain=0.01)),
]


def score(G, truth, **kw):
    info = {}
    nodes = list(G.nodes())
    lab, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8, info=info, **kw)
    g = defaultdict(set)
    for n in nodes:
        g[lab[n]].add(n)
    return (nmi_score([truth[n] for n in nodes], [lab[n] for n in nodes]),
            modularity(G, list(g.values())), len(g), info.get("refined", False), info.get("merges", 0))


def main():
    kind = sys.argv[1]
    if kind == "lfr":
        from lfr_sweep import generate_lfr
        n, mu, seed = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
        G, truth = generate_lfr(n=n, mu=mu, seed=seed)
        tag = f"lfr n={n} mu={mu} seed={seed}"
    elif kind == "snap":
        from load_snap import load_snap_network
        G, truth, _ = load_snap_network(sys.argv[2])
        truth = {n: str(v) for n, v in truth.items()}
        tag = f"snap {sys.argv[2]}"
    else:
        import datasets
        G, truth = datasets.REAL[sys.argv[2]]()
        tag = f"real {sys.argv[2]}"
    cells = []
    for name, kw in CONFIGS:
        a, q, c, r, mg = score(G, truth, **kw)
        cells.append(f"{name}={a:.4f}/{q:.4f}/c{c}{'/R' if r else ''}{f'/m{mg}' if mg else ''}")
    print(f"RESULT | {tag} | nodes={G.number_of_nodes()} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
