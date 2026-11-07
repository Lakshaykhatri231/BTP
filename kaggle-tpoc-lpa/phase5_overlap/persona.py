"""Full ego-splitting (Epasto, Lattanzi & Paes Leme, KDD 2017): split every node into personas = connected
components of its ego-net minus itself (components smaller than s are folded into the largest one),
run disjoint TPOC-LPA (Stages 1-2) on the persona graph, map persona communities back to nodes."""
import sys, os
from collections import defaultdict
sys.path.insert(0, "/Users/lakshaykhatri/Downloads/BTP/kaggle-tpoc-lpa")
import networkx as nx
from tpoc_lpa import tpoc_lpa

def persona_graph(G, s=2, L0=None):
    nb = {u: set(G.neighbors(u)) for u in G}
    pid = {}          # (i, neighbour j) -> persona id of i facing j
    owner = []
    for i in G:
        comps, seen = [], set()
        for st in nb[i]:
            if st in seen: continue
            comp, stack = [], [st]; seen.add(st)
            while stack:
                u = stack.pop(); comp.append(u)
                for v in nb[u] & nb[i]:
                    if v not in seen: seen.add(v); stack.append(v)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        big = [c for c in comps if len(c) >= s] or comps[:1]
        if L0 is not None and big:   # label-aware: one persona per distinct majority label of i's big components
            from collections import Counter
            grp = {}
            for c in big:
                grp.setdefault(Counter(L0[u] for u in c).most_common(1)[0][0], []).extend(c)
            main_lab = Counter(L0[u] for u in big[0]).most_common(1)[0][0]
            big = [grp[main_lab]] + [g for k, g in grp.items() if k != main_lab]
            comps = [big[0]] + [c for c in comps if len(c) < s]
        small = [u for c in comps if len(c) < s and c is not comps[0] for u in c] if comps else []
        ids = []
        for c in big:
            ids.append(len(owner)); owner.append(i)
            for u in c: pid[(i, u)] = ids[-1]
        for u in small:
            if (i, u) not in pid: pid[(i, u)] = ids[0]
        if not comps:
            owner.append(i)
    P = nx.Graph(); P.add_nodes_from(range(len(owner)))
    for a, b in G.edges():
        P.add_edge(pid[(a, b)], pid[(b, a)])
    return P, owner

def persona_cover(G, s=2, L0=None, **kw):
    P, owner = persona_graph(G, s, L0)
    L = tpoc_lpa(P, beta=0.8, **kw)[0]
    out = defaultdict(set); main = {}
    for p, c in L.items():
        out[owner[p]].add(c)
        if owner[p] not in main or P.degree(p) > P.degree(main[owner[p]][0]): main[owner[p]] = (p, c)
    persona_cover.main = {v: pc[1] for v, pc in main.items()}
    for v in G:
        out.setdefault(v, {("iso", v)})
    return dict(out), P
