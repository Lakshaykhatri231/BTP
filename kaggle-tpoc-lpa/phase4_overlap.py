"""
W3 -- is Stage 3 (label filtering) actually finding overlapping communities?

Every Phase 1-3 number is a DISJOINT NMI against single-label ground truth, yet
TPOC-LPA's headline is *overlapping* detection. Here Stage 3 is scored against
genuinely overlapping ground truth (SNAP com-* graphs keep every community a node
belongs to; Phase 1's loader kept only the first).

    python3 phase4_overlap.py selftest            # metric sanity checks
    python3 phase4_overlap.py sweep amazon        # beta sweep, Stage 3 variants
    python3 phase4_overlap.py sweep dblp

Metrics (Stage 3 output vs overlapping truth):
  ONMI_max  overlapping NMI, Lancichinetti-Fortunato-Kertesz conditional entropies with
            McDaid's max normalisation (the paper's "NMI_max")
  F1avg     average best-match F1, both directions (Yang & Leskovec's SNAP metric)
  EQ        extended modularity (paper Eq. 21, eq_metric.py) -- label-free
  ov%       share of nodes assigned to >1 community (truth's share printed for reference)
"""
import gzip
import math
import os
import sys
from collections import defaultdict

import networkx as nx

from eq_metric import eq_score
from tpoc_lpa import (node_importance, community_preselection, community_evolution,
                      community_label_filtering, edge_attractiveness, SIMILARITIES)

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(HERE, "data", "snap")


# ---------------------------------------------------------------- metrics ---
def _h(p):
    return -p * math.log2(p) if p > 0 else 0.0


def _cond_entropies(A, B, n):
    """For each community X_k in cover A: H(X_k) and H(X_k | B) per LFK, where the
    conditional is minimised over communities of B that satisfy LFK's constraint
    h(p11)+h(p00) > h(p01)+h(p10); if none does, H(X_k|B) = H(X_k)."""
    member = defaultdict(list)          # node -> indices of B communities
    for l, Y in enumerate(B):
        for v in Y:
            member[v].append(l)
    out = []
    for X in A:
        px = len(X) / n
        hx = _h(px) + _h(1 - px)
        inter = defaultdict(int)
        for v in X:
            for l in member.get(v, ()):
                inter[l] += 1
        best = hx
        for l, c in inter.items():
            ny = len(B[l])
            p11 = c / n
            p10 = (len(X) - c) / n
            p01 = (ny - c) / n
            p00 = 1 - p11 - p10 - p01
            if _h(p11) + _h(p00) > _h(p01) + _h(p10):
                hy = _h(ny / n) + _h(1 - ny / n)
                joint = _h(p11) + _h(p10) + _h(p01) + _h(p00)
                best = min(best, joint - hy)
        out.append((hx, best))
    return out


def onmi(A, B, n, norm="max"):
    """Overlapping NMI between covers A, B (lists of node sets) over n nodes."""
    A = [set(x) for x in A if x]
    B = [set(x) for x in B if x]
    if not A or not B:
        return 0.0
    ea, eb = _cond_entropies(A, B, n), _cond_entropies(B, A, n)
    if norm == "lfk":   # mean of per-community normalised conditional entropies
        hxy = sum(c / h for h, c in ea if h > 0) / len(ea)
        hyx = sum(c / h for h, c in eb if h > 0) / len(eb)
        return 1 - 0.5 * (hxy + hyx)
    HX, HY = sum(h for h, _ in ea), sum(h for h, _ in eb)
    HXY, HYX = sum(c for _, c in ea), sum(c for _, c in eb)
    mi = 0.5 * (HX + HY - HXY - HYX)
    return mi / max(HX, HY) if max(HX, HY) > 0 else 0.0


def f1_avg(A, B):
    """Average best-match F1 (Yang & Leskovec), symmetrised."""
    def one_way(P, T):
        member = defaultdict(list)
        for l, Y in enumerate(T):
            for v in Y:
                member[v].append(l)
        tot = 0.0
        for X in P:
            inter = defaultdict(int)
            for v in X:
                for l in member.get(v, ()):
                    inter[l] += 1
            best = 0.0
            for l, c in inter.items():
                p, r = c / len(X), c / len(T[l])
                best = max(best, 2 * p * r / (p + r))
            tot += best
        return tot / len(P)
    A = [set(x) for x in A if x]
    B = [set(x) for x in B if x]
    return 0.5 * (one_way(A, B) + one_way(B, A)) if A and B else 0.0


def as_cover(assign):
    """dict node -> set(labels)  ->  list of member sets."""
    comm = defaultdict(set)
    for v, cs in assign.items():
        for c in cs:
            comm[c].add(v)
    return list(comm.values())


def selftest():
    n = 100
    A = [set(range(0, 50)), set(range(50, 100))]
    assert abs(onmi(A, A, n) - 1) < 1e-9 and abs(f1_avg(A, A) - 1) < 1e-9
    assert onmi(A, [set(range(0, 100))], n) < 0.05                 # one giant blob vs 2 comms
    rnd = [set(range(i, 100, 2)) for i in range(2)]                # orthogonal split
    assert onmi(A, rnd, n) < 0.05, onmi(A, rnd, n)
    B = [set(range(0, 50)) | {60}, set(range(50, 100))]            # a node overlapped
    assert 0.8 < onmi(A, B, n) < 1.0 and 0.9 < f1_avg(A, B) < 1.0
    # disjoint covers: max-normalised ONMI must agree with sklearn NMI(max) closely
    from sklearn.metrics import normalized_mutual_info_score as nmi
    y = [0] * 40 + [1] * 35 + [2] * 25
    p = [0] * 35 + [1] * 5 + [1] * 30 + [2] * 5 + [2] * 25
    Ac = [set(i for i, v in enumerate(y) if v == k) for k in range(3)]
    Bc = [set(i for i, v in enumerate(p) if v == k) for k in range(3)]
    a, b = onmi(Ac, Bc, 100), nmi(y, p, average_method="max")
    assert abs(a - b) < 0.08, (a, b)
    print(f"selftest OK  (disjoint ONMI_max {a:.3f} vs sklearn NMI_max {b:.3f})")


# ------------------------------------------------------------------ data ----
def load_snap_overlap(name):
    """Induced subgraph on the union of top-5000 communities (as Phase 1), but keeping
    EVERY community of each node."""
    comms = []
    with gzip.open(os.path.join(SNAP_DIR, f"com-{name}.top5000.cmty.txt.gz"), "rt") as f:
        for line in f:
            ns = [int(x) for x in line.split()]
            if ns:
                comms.append(set(ns))
    keep = set().union(*comms)
    G = nx.Graph()
    G.add_nodes_from(keep)
    with gzip.open(os.path.join(SNAP_DIR, f"com-{name}.ungraph.txt.gz"), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            a, b = map(int, line.split())
            if a in keep and b in keep:
                G.add_edge(a, b)
    G.remove_nodes_from(list(nx.isolates(G)))
    nodes = set(G.nodes())
    truth = [c & nodes for c in comms]
    return G, [c for c in truth if len(c) >= 2]


# ------------------------------------------------------------ Stage 3 variants
def stage3_was(G, NI, label, beta, weight=None):
    """Paper Stage 3 (WAS over NI) when weight=None; else WAS over edge weights
    weight[(i,j)] (e.g. motif-aware attractiveness) -- the 'motif inside WAS' variant."""
    if weight is None:
        return community_label_filtering(G, NI, label, beta)
    out = {i: {label[i]} for i in G.nodes()}
    for i in G.nodes():
        neigh = list(G.neighbors(i))
        CL = {label[j] for j in neigh}
        if len(CL) > 1:
            denom = sum(weight[(i, k)] for k in neigh)
            was = {c: sum(weight[(i, j)] for j in neigh if label[j] == c) / denom for c in CL}
            avg = sum(was.values()) / len(CL)
            out[i] |= {c for c in CL if was[c] >= beta * avg}
    return out


def stage3_supported(G, NI, label, beta, support="triangle"):
    """Paper's WAS rule, but a node joins a NON-own community c only if it has structural
    support there: >= 2 neighbours in c ("nbr2"), or two neighbours in c that are adjacent,
    i.e. a triangle through the node inside c ("triangle"). Rationale: a single edge into c
    is chance; genuine overlap membership sits inside dense structure (cf. clique
    percolation), the same motif signal that helped Stage 2. Own label is always kept."""
    out = {i: {label[i]} for i in G.nodes()}
    for i in G.nodes():
        neigh = list(G.neighbors(i))
        CL = {label[j] for j in neigh}
        if len(CL) <= 1:
            continue
        denom = sum(NI[k] for k in neigh)
        was = {c: sum(NI[j] for j in neigh if label[j] == c) / denom for c in CL}
        avg = sum(was.values()) / len(CL)
        for c in CL:
            if c == label[i] or was[c] < beta * avg:
                continue
            inc = [j for j in neigh if label[j] == c]
            if len(inc) < 2:
                continue
            if support == "triangle":
                s = set(inc)
                # a triangle through i inside c: two of i's neighbours in c are adjacent
                if not any(x in s for a in inc for x in G.neighbors(a)):
                    continue
            out[i].add(c)
    return out


def sweep(name):
    G, truth = load_snap_overlap(name)
    n = G.number_of_nodes()
    tcount = defaultdict(int)
    for c in truth:
        for v in c:
            tcount[v] += 1
    tov = sum(1 for v in G.nodes() if tcount[v] > 1) / n
    print(f"{name}: n={n} m={G.number_of_edges()} truth communities={len(truth)}  truth overlap {tov:.1%}")
    NI = node_importance(G)
    for cfg, kw in (("paper (jaccard, no guard)", dict(sim="jaccard")),
                    ("phase3 (motif + guard)", dict(sim="tri_overlap", guard=True))):
        lab, _ = community_preselection(G, NI, 1.0)
        lab, _ = community_evolution(G, NI, lab, **kw)
        w = edge_attractiveness(G, NI, SIMILARITIES[kw["sim"]])
        print(f"\n  Stage 2 = {cfg}: {len(set(lab.values()))} communities")
        print(f"  {'Stage 3 variant':<28}{'beta':>6}{'ov%':>7}{'ONMI_max':>10}{'F1avg':>8}{'EQ':>8}")
        rows = [("disjoint (no Stage 3)", None, None)]
        rows += [("WAS over NI (paper)", b, None) for b in (0.5, 0.7, 0.8, 0.9, 0.95)]
        rows += [("WAS over A(i,j) (motif)", b, w) for b in (0.7, 0.8, 0.9)]
        rows += [("WAS + 2-neighbour support", b, "nbr2") for b in (0.5, 0.8)]
        rows += [("WAS + triangle support", b, "triangle") for b in (0.3, 0.5, 0.8)]
        for lbl, beta, wt in rows:
            if beta is None:
                assign = {v: {lab[v]} for v in G.nodes()}
            elif isinstance(wt, str):
                assign = stage3_supported(G, NI, lab, beta, wt)
            else:
                assign = stage3_was(G, NI, lab, beta, wt)
            cover = as_cover(assign)
            ov = sum(1 for v in G.nodes() if len(assign[v]) > 1) / n
            print(f"  {lbl:<28}{'' if beta is None else beta:>6}{ov:>7.1%}"
                  f"{onmi(cover, truth, n):>10.4f}{f1_avg(cover, truth):>8.4f}{eq_score(G, assign):>8.4f}",
                  flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) > 2 and sys.argv[1] == "sweep":
        sweep(sys.argv[2])
    else:
        print(__doc__)
