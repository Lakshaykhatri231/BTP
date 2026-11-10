"""Per-round trace of TPOC-LPA Stage 2 on email-Eu (used in the guard figure). Run from kaggle-tpoc-lpa/: python3 ../paper/figures/email_rounds.py"""
import sys, time, json; sys.path.insert(0, ".")
import datasets as D, tpoc_lpa as T
from sklearn.metrics import normalized_mutual_info_score as nmi
from networkx.algorithms.community.quality import modularity
G, truth = D.email_eu(); n = list(G.nodes())
NI = T.node_importance(G); L1, _ = T.community_preselection(G, NI, 1.0)
res = {}
for sim in ("jaccard", "tri_overlap"):
    rows = []
    for t in range(0, 30):
        lab, it = (dict(L1), 0) if t == 0 else T.community_evolution(G, NI, dict(L1), max_iter=t, sim=sim)
        q = modularity(G, T._partition_of(lab, n), weight=None)
        rows.append((t, len(set(lab.values())), round(q, 4), round(nmi([truth[x] for x in n], [lab[x] for x in n]), 4)))
        print(sim, rows[-1], flush=True)
        if t > 0 and it < t: break
    res[sim] = rows
json.dump(res, open("/Users/lakshaykhatri/Downloads/BTP/paper/figures/email_rounds.json", "w"))
