"""Phase 6: the baselines of the TPOC-LPA paper (Sec. 4.4) that have a standard public implementation,
plus the metrics the paper reports (NMI, Q, LFK-NMI, NMI_max, Omega, EQ).

Disjoint baselines (igraph C core): Fastgreedy (Clauset-Newman-Moore), LPA (Raghavan), Louvain (Blondel),
Walktrap (Pons-Latapy, steps=4). Extra, NOT in the paper: Infomap (the map equation our Phase 5 changes borrow from).
Overlapping baseline: SLPA (Xie et al. 2011), own implementation.
NOT runnable (no public code found; the paper's published numbers are quoted instead): NSA, SimCMR, WLC, UMSTMO, GREESE, MDPA.
"""
import os, sys, random, multiprocessing as mp
from collections import Counter, defaultdict
import numpy as np
import igraph
from sklearn.metrics import normalized_mutual_info_score as _nmi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase5_overlap"))

# ---------------------------------------------------------------- helpers
def to_igraph(G):
    nodes = list(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    g = igraph.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    return g, nodes

def modularity_of(G, label):
    from networkx.algorithms.community.quality import modularity
    g = defaultdict(set)
    for v, l in label.items(): g[l].add(v)
    return modularity(G, list(g.values()), weight=None)

def nmi_of(G, truth, label):
    nodes = list(G.nodes())
    return _nmi([str(truth[v]) for v in nodes], [str(label[v]) for v in nodes])

# ---------------------------------------------------------------- disjoint baselines
def _membership(cl, nodes): return {v: c for v, c in zip(nodes, cl.membership)}

def fastgreedy(G, seed=0):
    g, nodes = to_igraph(G); return _membership(g.community_fastgreedy().as_clustering(), nodes)

def walktrap(G, seed=0):
    g, nodes = to_igraph(G); return _membership(g.community_walktrap(steps=4).as_clustering(), nodes)

def lpa(G, seed=0):
    igraph.set_random_number_generator(random.Random(seed))
    g, nodes = to_igraph(G); return _membership(g.community_label_propagation(), nodes)

def louvain(G, seed=0):
    igraph.set_random_number_generator(random.Random(seed))
    g, nodes = to_igraph(G); return _membership(g.community_multilevel(), nodes)

def infomap(G, seed=0):
    import infomap as im
    nodes = list(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    m = im.Infomap(two_level=True, silent=True, seed=seed + 1, num_trials=1)
    for u, v in G.edges(): m.add_link(idx[u], idx[v])
    m.run()
    mod = m.get_modules()
    return {v: mod.get(idx[v], -1 - idx[v]) for v in nodes}

def tpoc_paper(G, seed=0):
    from tpoc_lpa import tpoc_lpa
    return tpoc_lpa(G)[0]

BEST = dict(sim="tri_overlap", guard=True, refine="granularity", merge=True, merge_criterion="mapeq", moves="synwalk")
def tpoc_best(G, seed=0):
    from tpoc_lpa import tpoc_lpa
    return tpoc_lpa(G, **BEST)[0]

DISJOINT = {"Fastgreedy": (fastgreedy, False), "LPA": (lpa, True), "Louvain": (louvain, True),
            "Walktrap": (walktrap, False), "Infomap*": (infomap, True),
            "TPOC-LPA (repro)": (tpoc_paper, False), "Ours": (tpoc_best, False)}   # (fn, stochastic)

def _worker(q, fn, G, seed):
    try: q.put(("ok", fn(G, seed)))
    except Exception as e: q.put(("err", repr(e)[:200]))

def run_timeout(fn, G, seed, timeout):
    """Run fn(G, seed) in a child process so a hung/slow C call can be abandoned.
    The parent reads the queue BEFORE joining: a child blocks on q.put() for results larger than the pipe buffer."""
    import queue
    ctx = mp.get_context("fork"); q = ctx.Queue(); p = ctx.Process(target=_worker, args=(q, fn, G, seed)); p.start()
    try:
        st, val = q.get(timeout=timeout)
    except queue.Empty:
        p.terminate(); p.join(); return None, "timeout"
    p.join()
    return (val, None) if st == "ok" else (None, val)

# ---------------------------------------------------------------- overlapping baseline: SLPA
def slpa_memory(G, T=100, seed=0):
    rnd = random.Random(seed)
    nodes = list(G.nodes()); nb = {v: list(G[v]) for v in nodes}
    mem = {v: [v] for v in nodes}
    for _ in range(T):
        rnd.shuffle(nodes)
        for v in nodes:
            if not nb[v]: continue
            got = Counter(rnd.choice(mem[u]) for u in nb[v])
            top = max(got.values())
            mem[v].append(rnd.choice([l for l, c in got.items() if c == top]))
    return mem

def slpa_cover(G, mem, r):
    """Xie et al.: keep labels with relative frequency >= r (at least the most frequent one), drop nested communities."""
    assign = {}
    for v, m in mem.items():
        c = Counter(m); tot = len(m)
        keep = {l for l, k in c.items() if k / tot >= r} or {c.most_common(1)[0][0]}
        assign[v] = keep
    comm = defaultdict(set)
    for v, ls in assign.items():
        for l in ls: comm[l].add(v)
    cid = {l: i for i, l in enumerate(comm)}
    by_node = defaultdict(set)
    for l, mem_ in comm.items():
        for v in mem_: by_node[v].add(l)
    keep = set(comm)
    for l, mem_ in sorted(comm.items(), key=lambda kv: len(kv[1])):
        it = iter(mem_); cand = set(by_node[next(it)]) - {l}
        for v in it:
            cand &= by_node[v]
            if not cand: break
        if cand and any(len(comm[d]) >= len(mem_) for d in cand):
            keep.discard(l)
            for v in mem_: by_node[v].discard(l)
    return {v: {l for l in ls if l in keep} or ls for v, ls in assign.items()}

# ---------------------------------------------------------------- overlapping metrics
def omega_index(A, B, n):
    """Omega index (Collins & Dent 1988) of two covers (lists of node-index sets) over n nodes. Equals the ARI for partitions."""
    def pairs(cover):
        keys = []
        for c in cover:
            c = np.fromiter(c, dtype=np.int64); c.sort()
            if len(c) < 2: continue
            iu, ju = np.triu_indices(len(c), 1); keys.append(c[iu] * n + c[ju])
        if not keys: return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(keys), return_counts=True)
    N = n * (n - 1) // 2
    ka, ca = pairs(A); kb, cb = pairs(B)
    _, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    both = len(ia); agree = int((ca[ia] == cb[ib]).sum())
    union = len(ka) + len(kb) - both
    obs = (agree + N - union) / N
    ja = np.bincount(ca); jb = np.bincount(cb)
    a = np.zeros(max(len(ja), len(jb)) + 1); b = np.zeros_like(a)
    a[1:len(ja) + 0 + 0] = ja[1:] if len(ja) > 1 else 0; b[1:len(jb)] = jb[1:] if len(jb) > 1 else 0
    a[0] = N - len(ka); b[0] = N - len(kb)
    exp = float((a * b).sum()) / (N * N)
    return (obs - exp) / (1 - exp) if exp < 1 else 1.0

def overlap_scores(G, truth_cover, assign):
    """-> dict(LFK, NMImax, F1, Omega, flag) for node->set(labels) `assign` against `truth_cover` (list of node sets)."""
    from phase4_overlap import onmi, f1_avg, as_cover
    n = G.number_of_nodes(); cov = as_cover(assign)
    idx = {v: i for i, v in enumerate(G.nodes())}
    tc = [{idx[v] for v in c} for c in truth_cover]; dc = [{idx[v] for v in c} for c in cov]
    return dict(LFK=onmi(cov, truth_cover, n, norm="lfk"), NMImax=onmi(cov, truth_cover, n), F1=f1_avg(cov, truth_cover),
                Omega=omega_index(dc, tc, n), flag=sum(len(assign[v]) > 1 for v in G) / n)

if __name__ == "__main__":     # self-tests
    import networkx as nx
    from sklearn.metrics import adjusted_rand_score
    rs = random.Random(1); n = 60
    y1 = [rs.randrange(5) for _ in range(n)]; y2 = [l if rs.random() < .7 else rs.randrange(5) for l in y1]
    cov = lambda y: [set(i for i in range(n) if y[i] == k) for k in set(y)]
    print("omega==ARI on partitions:", round(omega_index(cov(y1), cov(y2), n), 6), round(adjusted_rand_score(y1, y2), 6))
    print("omega identical overlapping covers:", omega_index([{0, 1, 2}, {2, 3, 4}], [{0, 1, 2}, {2, 3, 4}], 6))
    G = nx.karate_club_graph()
    for name, (fn, _) in DISJOINT.items():
        L = fn(G, 0); print(f"{name:<18} karate comms {len(set(L.values()))}  Q {modularity_of(G, L):.4f}")
    mem = slpa_memory(G, 100, 0); c = slpa_cover(G, mem, 0.3); print("SLPA r=.3 flagged", sum(len(x) > 1 for x in c.values()))
