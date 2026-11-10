"""Helpers for the illustration figures: build small example graphs, run the real SN-LPA code
(kaggle-tpoc-lpa/tpoc_lpa.py) step by step, and record every intermediate partition."""
import os, sys
from collections import defaultdict
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "kaggle-tpoc-lpa"))
import networkx as nx
import tpoc_lpa as T
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import normalized_mutual_info_score as _nmi

FINAL = dict(sim="tri_overlap", guard=True, refine="granularity", merge=True, merge_criterion="mapeq", moves="synwalk")


def planted(sizes, pin, pout, seed):
    """Planted-partition graph with nodes numbered 1..n (community blocks are consecutive)."""
    G0 = nx.random_partition_graph(list(sizes), pin, pout, seed=seed)
    G = nx.Graph()
    G.add_nodes_from(v + 1 for v in sorted(G0))
    G.add_edges_from((u + 1, v + 1) for u, v in G0.edges())
    truth = {v + 1: G0.nodes[v]["block"] for v in G0}
    return G, truth


def k(lab): return len(set(lab.values()))
def Q(G, lab): return modularity(G, T._partition_of(lab, list(G.nodes())), weight=None)
def NMI(G, lab, truth):
    n = list(G.nodes()); return _nmi([truth[x] for x in n], [lab[x] for x in n])


def rounds(G, NI, L1, sim="jaccard", gamma=0.0, max_t=50):
    """Partition after every Stage-2 round (round 0 = Stage 1), using the library itself:
    the code is deterministic, so running max_iter=t gives the state after t rounds."""
    out = [dict(L1)]
    for t in range(1, max_t + 1):
        if gamma > 0:
            lab, it = T.community_evolution_null(G, NI, dict(L1), T.SIMILARITIES[sim], gamma, max_iter=t)
        else:
            lab, it = T.community_evolution(G, NI, dict(L1), max_iter=t, sim=sim)
        if lab == out[-1]:
            break
        out.append(lab)
    return out


def stage2_steps(G, NI, L1, sim="jaccard", gamma=0.0, max_iter=100):
    """Re-runs Stage 2 node by node and records each decision with its scores. Mirrors
    community_evolution (gamma=0) and community_evolution_null (gamma>0); the result is checked
    against the library below, so the recorded scores are the ones the code uses."""
    sim_fn = T.SIMILARITIES[sim]
    w = T.edge_attractiveness(G, NI, sim_fn)
    order = T._order_by_ni(G.nodes(), NI)
    lab = dict(L1); steps = []
    s = {i: sum(w[(i, j)] for j in G.neighbors(i)) for i in G}; W = sum(s.values())
    S = defaultdict(float)
    for i in G: S[lab[i]] += s[i]
    for it in range(1, max_iter + 1):
        changed = False
        for i in order:
            cur = lab[i]
            if gamma == 0:
                RC = {lab[j] for j in G.neighbors(i)}
                if len(RC) == 1:
                    new, sc = next(iter(RC)), None
                else:
                    sc = {c: sum(NI[i] * NI[j] / (2 - sim_fn(G, i, j)) ** 2 for j in G.neighbors(i) if lab[j] == c) for c in RC}
                    b = max(sc.values()); new = min([c for c, v in sc.items() if v == b], key=str)
            else:
                kk = defaultdict(float)
                for j in G.neighbors(i): kk[lab[j]] += w[(i, j)]
                sc = {}
                for c in set(kk) | {cur}:
                    tot = S[c] - (s[i] if c == cur else 0.0)
                    sc[c] = (kk.get(c, 0.0), gamma * s[i] * tot / W, kk.get(c, 0.0) - gamma * s[i] * tot / W)
                b = max(v[2] for v in sc.values()); tied = [c for c, v in sc.items() if v[2] >= b - 1e-12]
                new = cur if cur in tied else min(tied, key=str)
                if new != cur: S[cur] -= s[i]; S[new] += s[i]
            steps.append(dict(round=it, node=i, old=cur, new=new, scores=sc))
            if new != cur: lab[i] = new; changed = True
        if not changed: break
    ref = (T.community_evolution_null(G, NI, dict(L1), sim_fn, gamma)[0] if gamma > 0
           else T.community_evolution(G, NI, dict(L1), sim=sim)[0])
    assert lab == ref, "step trace disagrees with the library"
    return steps, lab


def pipeline(G, truth=None):
    """Every intermediate partition of SN-LPA (final configuration) and of the baseline."""
    NI = T.node_importance(G); L1, _ = T.community_preselection(G, NI, 1.0)
    r = dict(NI=NI, L1=dict(L1))
    r["base"] = T.community_evolution(G, NI, dict(L1))[0]                         # TPOC-LPA Stage 2
    r["base_ov"] = T.community_label_filtering(G, NI, r["base"], 0.8)
    r["g0"] = T.community_evolution(G, NI, dict(L1), sim="tri_overlap", guard=True)[0]   # M1+M2, gamma=0
    r["g1"] = T.community_evolution_null(G, NI, dict(L1), T.tri_overlap, 1.0)[0]        # M1+M3, gamma=1
    r["refined"] = k(r["g1"]) <= 1.5 * k(r["g0"])                                        # M4
    r["S2"] = dict(r["g1"] if r["refined"] else r["g0"])
    r["M5"], _, r["merges"] = T.community_merge(G, dict(r["S2"]), 0.0, "mapeq")         # M5
    r["M6"], r["moves"] = T.boundary_moves(G, NI, dict(r["M5"]), criterion="synwalk")   # M6
    r["ov"] = T.community_label_filtering(G, NI, r["M6"], 0.8)                          # Stage 3
    full = T.tpoc_lpa(G, **FINAL)
    assert full[0] == r["M6"] and full[1] == r["ov"], "pipeline trace disagrees with tpoc_lpa()"
    base = T.tpoc_lpa(G)
    assert base[0] == r["base"] and base[1] == r["base_ov"]
    return r


def stage1_steps(G, NI, alpha=1.0):
    """Re-runs Stage 1 (community_preselection) and records, for every core, the neighbours it
    tried in NI order with their CTS value and whether they joined. Checked against the library."""
    U = set(G.nodes()); label = {}; steps = []
    while U:
        u = T._argmax_by_ni(U, NI); U.remove(u); s = {u}; label[u] = u; prev = 0.0; tried = []
        for v in T._order_by_ni((v for v in G.neighbors(u) if v in U), NI):
            if v not in U: continue
            cts = T.cts_of_community(G, s | {v}); ok = cts >= alpha * prev
            tried.append((v, cts, ok))
            if ok: s.add(v); label[v] = u; U.discard(v); prev = cts
        steps.append(dict(core=u, members=sorted(s), tried=tried, label=dict(label)))
    assert label == T.community_preselection(G, NI, alpha)[0], "Stage 1 trace disagrees with the library"
    return steps


def stage3_detail(G, NI, label, beta=0.8):
    """WAS of every boundary node (Eq. 14) and the labels it keeps; mirrors community_label_filtering."""
    out = {}
    for i in G:
        neigh = list(G.neighbors(i)); CL = {label[j] for j in neigh}
        if len(CL) > 1:
            den = sum(NI[k] for k in neigh)
            was = {c: sum(NI[j] for j in neigh if label[j] == c) / den for c in CL}
            mean = sum(was.values()) / len(was)
            out[i] = dict(was=was, mean=mean, keep=sorted(c for c in CL if was[c] >= beta * mean))
    return out


def moves_trace(G, NI, label):
    """Single-node Synwalk moves in the order boundary_moves makes them: list of (sweep, node, from, to).
    Mirrors the criterion="synwalk" path of boundary_moves (before its gate); checked against the library."""
    from collections import defaultdict
    m2 = 2.0 * G.number_of_edges(); lab = dict(label)
    vol, cut = defaultdict(float), defaultdict(float)
    for n in G: vol[lab[n]] += G.degree(n)
    for u, v in G.edges():
        if lab[u] != lab[v]: cut[lab[u]] += 1; cut[lab[v]] += 1
    log = []
    for sw in range(20):
        ch = 0
        for i in T._order_by_ni(G.nodes(), NI):
            a, ki = lab[i], G.degree(i); kc = defaultdict(int)
            for j in G.neighbors(i): kc[lab[j]] += 1
            kia = kc.get(a, 0); ca2 = cut[a] - (ki - kia) + kia
            ja = T._synwalk_term(vol[a] - ki, ca2, m2) - T._synwalk_term(vol[a], cut[a], m2)
            best, bd, bs = a, 0.0, None
            for b, kib in kc.items():
                if b == a: continue
                cb2 = cut[b] + (ki - kib) - kib
                d = -(ja + T._synwalk_term(vol[b] + ki, cb2, m2) - T._synwalk_term(vol[b], cut[b], m2))
                if d < bd - 1e-15: best, bd, bs = b, d, cb2
            if best != a:
                cut[a], cut[best] = ca2, bs; vol[a] -= ki; vol[best] += ki; lab[i] = best; ch += 1; log.append((sw, i, a, best))
        if not ch: break
    ref, n = T.boundary_moves(G, NI, dict(label), criterion="synwalk")
    assert n == len(log) and ref == lab, "move trace disagrees with the library"
    return log
