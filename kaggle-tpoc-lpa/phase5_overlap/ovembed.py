"""Embedded-link Stage 3: node v also joins community c (c != its own label) iff it has >= e links to members u of c
that are embedded (u is adjacent to another neighbour of v that is in c, i.e. the link closes a triangle inside c)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/Users/lakshaykhatri/Downloads/BTP/kaggle-tpoc-lpa")
from collections import defaultdict

def overlap_embedded(G, L, e=3):
    nb = {v: set(G[v]) for v in G}
    out = {}
    for v in G:
        byc = defaultdict(set)
        for u in nb[v]: byc[L[u]].add(u)
        out[v] = {L[v]} | {c for c, S in byc.items() if c != L[v] and len(S) >= e and sum(1 for u in S if nb[u] & S) >= e}
    return out

if __name__ == "__main__":
    from lfrov_gen import gen
    from lfrov_run import run, score
    from tpoc_lpa import community_label_filtering, node_importance
    seed = int(sys.argv[1]); rs = sys.argv[2].split(",")
    for r in rs:
        for om in range(2, 9):
            G, truth = gen(r, om, seed); L, ov, _ = run(r, om, seed, "best")
            res = [("WAS", ov), ("E2", overlap_embedded(G, L, 2)), ("E3", overlap_embedded(G, L, 3))]
            print(r, f"om{om}", " ".join(f"{n} {x[0]:.4f} {x[1]:.4f} {x[3]:.3f}" for n, x in ((n, score(G, truth, a)) for n, a in res)), flush=True)
