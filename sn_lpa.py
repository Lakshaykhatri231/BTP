"""
SN-LPA: Size Normalized Label Propagation Algorithm (final model).
Quick use:  from sn_lpa import sn_lpa;  labels, overlap, NI, iters = sn_lpa(G)

Built on TPOC-LPA:
Kong, Huo, Wang, Guo (2026), "Three-stage overlapping community detection
algorithm based on label propagation", Expert Systems With Applications 317, 131873.

Equations implemented (paper numbering):
  Eq.2-4  : improved node importance NI(i)
  Eq.6-8  : community tightness-sparsity CTS(c)
  Eq.11   : node attractiveness A(i,j) (gravitation analogy)
  Eq.12   : community attractiveness A(c,i)
  Eq.14   : weighted community affiliation WAS(i,c)
  Algorithm 1: three stages (pre-selection, evolution, label filtering)

WHERE THE MODIFICATIONS ARE (search this file for ">>> CHANGE" blocks and "[CHANGE k]" single lines).
Every modification is OPT-IN: with every switch off, this file IS the base paper (verified by
hashing the outputs of all four configurations on 8 graphs before/after each edit).

  CHANGE 1  motif similarity        tpoc_lpa(G, sim="tri_overlap")              tri_overlap()
  CHANGE 2  best-iterate guard      tpoc_lpa(G, guard=True)                     community_evolution(), _partition_of()
  CHANGE 3  size-corrected Stage 2  tpoc_lpa(G, gamma=1.0)                      edge_attractiveness(), community_evolution_null()
  CHANGE 4  granularity rule        tpoc_lpa(G, guard=True, refine="granularity")   tpoc_lpa()
  CHANGE 5  merge pass              tpoc_lpa(G, merge=True, merge_min_gain=0.03)    community_merge()
  CHANGE 6  map-equation merge stop tpoc_lpa(G, merge=True, merge_criterion="mapeq") community_merge(), map_codelength()
  CHANGE 8  boundary node moves     tpoc_lpa(G, moves=True)                     boundary_moves(), map_codelength(regularized=True)
  CHANGE 9  Synwalk boundary moves  tpoc_lpa(G, moves="synwalk")               boundary_moves(criterion="synwalk"), _synwalk_term()
  CHANGE 11 near-tie overlap rule   tpoc_lpa(G, overlap="near_tie", overlap_tau=0.5)  overlap_near_tie()
  CHANGE 10 Synwalk guard           tpoc_lpa(G, guard=True, guard_criterion="synwalk") community_evolution()   EXPLORED, NOT ADOPTED (Physics 0.2799, CS 0.4363: worse than the Q guard)
  CHANGE 7  map-equation guard      tpoc_lpa(G, guard=True, guard_criterion="mapeq") community_evolution()   EXPLORED, NOT ADOPTED (dev: email-Eu 0.64 -> 0.00)

Lines marked "PAPER" are the base paper, unchanged. Blocks marked "EXPLORED, NOT ADOPTED" are ideas
that were implemented and measured but did not help (kept so the negative result can be reproduced).
"""
import math
import time

import networkx as nx


# PAPER: Eq. 2-4 (node importance), unchanged
def node_importance(G: nx.Graph) -> dict:
    """NI(i) per Eq. 2-4."""
    c = {}
    for i in G.nodes():
        neigh = list(G.neighbors(i))
        cnt = 0
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                if G.has_edge(neigh[a], neigh[b]):
                    cnt += 1
        c[i] = cnt  # c(i): actual edges between neighbors of i

    total_c = sum(c.values())
    f = {}
    for i in G.nodes():
        d_i = G.degree(i)
        f[i] = d_i * (1 + c[i] / total_c) if total_c > 0 else float(d_i)

    f_min, f_max = min(f.values()), max(f.values())
    e = {}
    for i in G.nodes():
        e[i] = 0.5 + 0.5 * (f[i] - f_min) / (f_max - f_min) if f_max > f_min else 1.0

    NI = {}
    for i in G.nodes():
        neigh = list(G.neighbors(i))
        NI[i] = e[i] + (sum(e[j] for j in neigh) / len(neigh) if neigh else 0.0)
    return NI


# PAPER: the similarity term of Eq. 11, unchanged (CHANGE 1 replaces it, opt-in)
def jaccard(G: nx.Graph, i, j) -> float:
    Ni, Nj = set(G.neighbors(i)), set(G.neighbors(j))
    union = Ni | Nj
    return len(Ni & Nj) / len(union) if union else 0.0


# >>> CHANGE 1 START: motif similarity (triangle count) -- replaces Jaccard inside Eq. 11 ------------
def tri_overlap(G: nx.Graph, i, j) -> float:
    """Motif-aware similarity: saturating count of triangles on edge (i,j).

    |N(i) ∩ N(j)| is exactly the number of triangles the edge participates in.
    Jaccard divides that by |N(i) ∪ N(j)|, so a genuinely triangle-rich edge
    between two high-degree nodes is scored LOW simply because the union is
    large. This form depends only on the absolute triangle count, saturating
    in [0,1), which keeps hub-to-hub community edges strong. From the
    motif-aware label-propagation literature (H-LPMC, KAIS 2025; motif-aware
    LPA, ACM TKDD; graphlet edge weighting, JIIS 2026).
    """
    c = len(set(G.neighbors(i)) & set(G.neighbors(j)))
    return c / (1.0 + c)


SIMILARITIES = {"jaccard": jaccard, "tri_overlap": tri_overlap}   # [CHANGE 1] registers the new similarity
# <<< CHANGE 1 END -----------------------------------------------------------------------------


# >>> EXPLORED, NOT ADOPTED START: adaptive kappa in c/(kappa+c) (no gain, Dolphin -0.056) ---------
def make_tri_overlap(kappa: float):
    """`c/(kappa+c)` triangle-overlap similarity; kappa=1 is `tri_overlap`.

    A larger kappa demands more common neighbours before an edge counts as
    'triangle-rich', which discounts the chance triangles that abound in sparse
    graphs (Phase 3 regressions on Youtube / Cora / Citeseer).
    """
    def sim(G, i, j):
        c = len(set(G.neighbors(i)) & set(G.neighbors(j)))
        return c / (kappa + c)
    return sim


def auto_kappa(G: nx.Graph) -> float:
    """Mean number of common neighbours per edge (= 3 * triangles / edges).

    Label-free graph statistic: dense, clustered graphs get a large kappa, and
    triangle-poor sparse graphs a small one. Floored at 1 so the result never
    rewards chance triangles MORE than the Phase 3 default."""
    m = G.number_of_edges()
    if m == 0:
        return 1.0
    common = sum(len(set(G.neighbors(u)) & set(G.neighbors(v))) for u, v in G.edges())
    return max(1.0, common / m)

# Aggregations of the per-neighbour attractiveness terms into a community score.
# "sum" is TPOC-LPA Eq. 12 as published; the damped forms test the
# unnormalized-sum bias (they did NOT help across seeds -- kept for the ablation).
# <<< EXPLORED, NOT ADOPTED END ---------------------------------------------------------------
AGGREGATIONS = {
    "sum": sum,
    "sqrt": lambda terms: sum(math.sqrt(x) for x in terms),
    "log1p": lambda terms: sum(math.log1p(x) for x in terms),
}


# PAPER: Eq. 6-8 (community tightness-sparsity), unchanged
def cts_of_community(G: nx.Graph, members: set) -> float:
    """CTS(c) per Eq. 6-8, computed from scratch for a given member set."""
    n_c = len(members)
    if n_c == 0:
        return 0.0
    e_in = 0
    e_out = 0
    seen_internal = set()
    for u in members:
        for v in G.neighbors(u):
            if v in members:
                key = frozenset((u, v))
                if key not in seen_internal:
                    seen_internal.add(key)
                    e_in += 1
            else:
                e_out += 1
    ct = e_in / n_c
    cs = e_out / (e_in + e_out) if (e_in + e_out) > 0 else 0.0
    return ct / (1 + cs)


def _argmax_by_ni(items, NI, rng=None):
    """Highest-NI item. Ties broken deterministically by str() unless `rng`
    is given, in which case they are broken at random -- which is what the
    original paper specifies (it states ties are broken randomly, e.g. in
    its Fig. 4 walkthrough). `rng=None` (the default) is our deterministic
    variant."""
    items = list(items)
    best = max(NI[x] for x in items)
    tied = [x for x in items if NI[x] == best]
    if rng is not None and len(tied) > 1:
        return rng.choice(tied)
    return min(tied, key=str)


def _order_by_ni(items, NI, rng=None):
    """Descending-NI ordering; ties by str(), or shuffled when `rng` given."""
    items = list(items)
    if rng is None:
        return sorted(items, key=lambda x: (-NI[x], str(x)))
    shuffled = items[:]
    rng.shuffle(shuffled)
    return sorted(shuffled, key=lambda x: -NI[x])


# PAPER: Algorithm 1, Step 1 = Stage 1 (pre-selection), unchanged
def community_preselection(G: nx.Graph, NI: dict, alpha: float, rng=None):
    """Algorithm 1, Step 1.

    Literal reading of the pseudocode: for the current core node u, do a
    SINGLE pass over u's own direct neighbors (not a growing/BFS frontier
    over the whole community-so-far), in descending NI order, gating each
    addition on the CTS test.
    """
    U = set(G.nodes())
    label = {}
    pre_communities = []

    while U:
        u = _argmax_by_ni(U, NI, rng)
        U.remove(u)
        s = {u}
        label[u] = u
        cts_prev = 0.0

        candidates = _order_by_ni((v for v in G.neighbors(u) if v in U), NI, rng)
        for v in candidates:
            if v not in U:
                continue  # may have been claimed by an earlier core already
            trial = s | {v}
            cts_curr = cts_of_community(G, trial)
            if cts_curr >= alpha * cts_prev:
                s.add(v)
                label[v] = u
                U.discard(v)
                cts_prev = cts_curr

        pre_communities.append(s)
    return label, pre_communities


# >>> CHANGE 2 START (helper): turn a label dict into a partition so modularity can be measured ----
def _partition_of(label, nodes):
    groups = {}
    for n in nodes:
        groups.setdefault(label[n], set()).add(n)
    return list(groups.values())


# <<< CHANGE 2 END (helper) ----------------------------------------------------------------


# >>> CHANGE 6 START (helper): two-level map equation (Rosvall & Bergstrom 2008), undirected, unweighted ----
def _plogp(x):
    return x * math.log2(x) if x > 0 else 0.0


def map_codelength(G: nx.Graph, label: dict, regularized: bool = False) -> float:
    """L(M) = q log q - 2 sum_i q_i log q_i - sum_a p_a log p_a + sum_i (q_i+p_i) log(q_i+p_i)
    with p_a = k_a/2m (node visit rate) and q_i = cut(i)/2m (module exit rate).
    Lower is better. Its resolution limit is set by the links BETWEEN modules,
    not by all links as for modularity (Kawamoto & Rosvall, PRE 2015).

    [CHANGE 8] regularized=True: the Bayesian-regularized map equation (Smiljanic, Edler & Rosvall,
    PRE 2020; Smiljanic et al. 2021 Eq. 10): a uniform prior weight g = ln N / N on every node pair
    (self-pairs included), i.e. the map equation of G plus a complete graph of weight g. Matches
    infomap's `regularized=True` codelength (Karate: 5.2608 split / 4.9566 one module)."""
    m2 = 2.0 * G.number_of_edges()
    if m2 == 0:
        return 0.0
    if regularized:   # [CHANGE 8]
        N = G.number_of_nodes()   # [CHANGE 8]
        g = math.log(N) / N   # [CHANGE 8]
        W = m2 + N * N * g   # [CHANGE 8]
        p = {n: (G.degree(n) + N * g) / W for n in G.nodes()}   # [CHANGE 8]
        size, vol, cut = {}, {}, {}   # [CHANGE 8]
        for n in G.nodes():   # [CHANGE 8]
            size[label[n]] = size.get(label[n], 0) + 1   # [CHANGE 8]
            vol[label[n]] = vol.get(label[n], 0.0) + p[n]   # [CHANGE 8]
        for u, v in G.edges():   # [CHANGE 8]
            if label[u] != label[v]:   # [CHANGE 8]
                cut[label[u]] = cut.get(label[u], 0) + 1   # [CHANGE 8]
                cut[label[v]] = cut.get(label[v], 0) + 1   # [CHANGE 8]
        q = {c: (cut.get(c, 0) + g * size[c] * (N - size[c])) / W for c in size}   # [CHANGE 8]
        return (_plogp(sum(q.values())) - 2 * sum(_plogp(x) for x in q.values())   # [CHANGE 8]
                - sum(_plogp(x) for x in p.values()) + sum(_plogp(q[c] + vol[c]) for c in size))   # [CHANGE 8]
    vol, cut = {}, {}
    for n in G.nodes():
        vol[label[n]] = vol.get(label[n], 0) + G.degree(n)
    for u, v in G.edges():
        if label[u] != label[v]:
            cut[label[u]] = cut.get(label[u], 0) + 1
            cut[label[v]] = cut.get(label[v], 0) + 1
    q = sum(cut.values()) / m2
    return (_plogp(q) - 2 * sum(_plogp(c / m2) for c in cut.values())
            - sum(_plogp(G.degree(n) / m2) for n in G.nodes())
            + sum(_plogp((cut.get(c, 0) + vol[c]) / m2) for c in vol))


# <<< CHANGE 6 END (helper) ----------------------------------------------------------------


# PAPER: Algorithm 1, Step 2 = Stage 2 (Eq. 11-12). Extended by CHANGE 2 (guard) and CHANGE 3 (gamma), both opt-in
def community_evolution(G: nx.Graph, NI: dict, label: dict, max_iter: int = 100, rng=None,
                         sim="jaccard", agg="sum", guard=False, gamma=0.0, guard_criterion="q"):
    """Algorithm 1, Step 2: attractiveness-guided label reassignment.

    `sim` / `agg` / `guard` default to the published TPOC-LPA behaviour
    (Jaccard, plain sum, no guard) so Phase 1 results stay reproducible.

    `guard=True` returns the highest-modularity iterate instead of the last.
    Rationale: Eq. 12 aggregates neighbour attractiveness with an
    UNNORMALIZED sum, so a community that grows gains score purely by size --
    a rich-get-richer dynamic with nothing bounding it. On dense graphs this
    runs away: on email-Eu-core (986 nodes, 42 departments, avg degree 33)
    Stage 1 produces 335 communities at NMI 0.5415 and Stage 2 then collapses
    them into ONE community at NMI 0.0000. Keeping the best iterate recovers
    the full 0.5415 there and is a byte-exact no-op on Karate, Dolphin,
    Football, Polbooks, Polblogs, Cora and Citeseer.

    Caveat: the criterion is modularity, which can diverge from ground truth
    (splitting Karate raises Q while destroying NMI). The guard only ever
    *rejects* later iterates, never forces a move, and was a no-op on every
    benchmark except the one it rescues -- but it is not a free lunch on a
    graph whose truth is far from modularity-optimal.
    """
    sim_fn = SIMILARITIES[sim] if isinstance(sim, str) else sim
    agg_fn = AGGREGATIONS[agg] if isinstance(agg, str) else agg
    if gamma > 0:   # [CHANGE 3] switch to the size-corrected Stage 2
        # null-model Stage 2 (see community_evolution_null); gamma=0 is Eq. 12 as published
        return community_evolution_null(G, NI, label, sim_fn, gamma, max_iter, rng)
    order = _order_by_ni(G.nodes(), NI, rng)
    if guard:   # [CHANGE 2] remember the best-modularity iterate, starting from Stage 1's partition
        from networkx.algorithms.community.quality import modularity   # [CHANGE 2]
        nodes = list(G.nodes())   # [CHANGE 2]
        best_label = dict(label)   # [CHANGE 2]
        best_q = modularity(G, _partition_of(label, nodes))   # [CHANGE 2]
        if guard_criterion == "mapeq":   # [CHANGE 7] score = minus map codelength instead of modularity
            best_q = -map_codelength(G, label)   # [CHANGE 7]
        elif guard_criterion == "synwalk":   # [CHANGE 10] score = Synwalk objective (0 for a collapse)
            best_q = synwalk_objective(G, label)   # [CHANGE 10]
    it = 0
    while it < max_iter:
        changed = False
        for i in order:
            neigh = list(G.neighbors(i))
            RC = {label[j] for j in neigh}
            if len(RC) == 0:
                continue
            if len(RC) == 1:
                new_label = next(iter(RC))
            else:
                scores = {}
                for c in RC:
                    terms = [(NI[i] * NI[j]) / ((2 - sim_fn(G, i, j)) ** 2)   # PAPER Eq. 11 (sim_fn = Jaccard unless CHANGE 1)
                             for j in neigh if label[j] == c]
                    scores[c] = agg_fn(terms)   # PAPER Eq. 12: plain SUM = the unbounded rule (the defect)
                best_A = max(scores.values())
                tied = [c for c, v in scores.items() if v == best_A]
                if rng is not None and len(tied) > 1:
                    new_label = rng.choice(tied)
                else:
                    new_label = min(tied, key=str)
            if new_label != label[i]:
                label[i] = new_label
                changed = True
        it += 1
        if guard:   # [CHANGE 2] after every pass: measure modularity, keep the best pass
            if guard_criterion == "mapeq":   # [CHANGE 7]
                q = -map_codelength(G, label)   # [CHANGE 7]
            elif guard_criterion == "synwalk":   # [CHANGE 10]
                q = synwalk_objective(G, label)   # [CHANGE 10]
            else:
                q = modularity(G, _partition_of(label, nodes))   # [CHANGE 2]
            if q > best_q + 1e-12:   # [CHANGE 2]
                best_q, best_label = q, dict(label)   # [CHANGE 2]
        if not changed:
            break
    return (best_label if guard else label), it   # [CHANGE 2] return the best pass, not the last


# >>> CHANGE 3 START: size-corrected Stage 2 (subtract the expected-by-chance attraction; LPAm-style) ---
def edge_attractiveness(G: nx.Graph, NI: dict, sim_fn) -> dict:
    """Static per-edge A(i,j) = NI(i)NI(j)/(2-Sim(i,j))^2 (Eq. 11), both directions.
    NI and Sim never change between passes, so this can be computed once."""
    w = {}
    for i, j in G.edges():
        v = (NI[i] * NI[j]) / ((2 - sim_fn(G, i, j)) ** 2)
        w[(i, j)] = v
        w[(j, i)] = v
    return w


def community_evolution_null(G: nx.Graph, NI: dict, label: dict, sim_fn, gamma: float,
                             max_iter: int = 100, rng=None):
    """Stage 2 with a configuration-model null term (LPAm-style).

    Eq. 12 scores a candidate community c by  sum_{j in N(i) & c} A(i,j)  -- an
    UNNORMALIZED sum, i.e. the gamma = 0 endpoint of this family. Here the score
    is the weighted modularity-style gain

        gain(c) = sum_{j in N(i) & c} A(i,j)  -  gamma * s_i * S_c / W

    with s_i = sum_j A(i,j), S_c = sum of s over c's members (i itself excluded
    for its current community) and W = sum_i s_i. The subtracted term grows with
    community size, which bounds the rich-get-richer runaway of Eq. 12.
    gamma = 1 is LPAm (Barber & Clark 2009) with NI-weighted edges.
    """
    from collections import defaultdict
    w = edge_attractiveness(G, NI, sim_fn)
    order = _order_by_ni(G.nodes(), NI, rng)
    s = {i: sum(w[(i, j)] for j in G.neighbors(i)) for i in G.nodes()}
    W = sum(s.values())
    if W <= 0:
        return label, 0
    S = defaultdict(float)
    for i in G.nodes():
        S[label[i]] += s[i]
    it = 0
    while it < max_iter:
        changed = False
        for i in order:
            k = defaultdict(float)
            for j in G.neighbors(i):
                k[label[j]] += w[(i, j)]
            cur = label[i]
            gains = {}
            for c in set(k) | {cur}:
                tot = S[c] - (s[i] if c == cur else 0.0)
                gains[c] = k.get(c, 0.0) - gamma * s[i] * tot / W
            best = max(gains.values())
            tied = [c for c, g in gains.items() if g >= best - 1e-12]
            if cur in tied:
                new = cur  # never move for a tie: prevents oscillation
            elif rng is not None and len(tied) > 1:
                new = rng.choice(tied)
            else:
                new = min(tied, key=str)
            if new != cur:
                S[cur] -= s[i]
                S[new] += s[i]
                label[i] = new
                changed = True
        it += 1
        if not changed:
            break
    return label, it


# <<< CHANGE 3 END ------------------------------------------------------------------------------


# >>> CHANGE 5 START: merge pass (greedy modularity-gain merging, gated by total gain) -------------------
def community_merge(G: nx.Graph, label: dict, min_gain: float = 0.0, criterion: str = "q"):
    """Greedy agglomerative merge of communities while modularity gain > 0
    (LPAm+ / Clauset-Newman-Moore style), gated by a TOTAL-gain threshold.

    Returns (new_label, total_dq, n_merges). If the total modularity gain is
    below `min_gain` the merges are rejected and `label` is returned unchanged
    (n_merges = 0): a pass that barely moves Q is more likely to erase a real
    community (Football: +0.002) than to fix under-merging (Cora: +0.076).

    [CHANGE 6] criterion="mapeq": same greedy dQ merge ORDER, but the pass stops at the
    prefix of merges with the lowest map codelength (map_codelength), tracked incrementally;
    `min_gain` is then ignored. If no prefix lowers the codelength, nothing is merged.
    """
    import heapq
    from collections import defaultdict
    m = G.number_of_edges()
    if m == 0:
        return dict(label), 0.0, 0
    comm = defaultdict(set)
    for n, l in label.items():
        comm[l].add(n)
    deg = {c: sum(G.degree(n) for n in ns) for c, ns in comm.items()}
    e = defaultdict(lambda: defaultdict(int))
    for u, v in G.edges():
        a, b = label[u], label[v]
        if a != b:
            e[a][b] += 1
            e[b][a] += 1

    def dq(a, b):
        return e[a][b] / m - deg[a] * deg[b] / (2.0 * m * m)

    heap = []
    for a in e:
        for b in e[a]:
            if str(a) < str(b):
                g = dq(a, b)
                if g > 1e-12:
                    heap.append((-g, str(a), str(b), a, b))
    heapq.heapify(heap)
    new_label = dict(label)
    total, merges = 0.0, 0
    if criterion == "mapeq":   # [CHANGE 6] incremental map codelength (node-entropy term is constant, dropped)
        m2 = 2.0 * m   # [CHANGE 6]
        cut = {c: sum(e[c].values()) for c in comm}   # [CHANGE 6]
        sq = sum(cut.values()) / m2   # [CHANGE 6] total exit rate
        s_exit = sum(_plogp(x / m2) for x in cut.values())   # [CHANGE 6]
        s_mod = sum(_plogp((cut[c] + deg[c]) / m2) for c in comm)   # [CHANGE 6]
        best_L = cur_L = _plogp(sq) - 2 * s_exit + s_mod   # [CHANGE 6]
        best_k, best_total, seq = 0, 0.0, []   # [CHANGE 6]
    while heap:
        neg, _, _, a, b = heapq.heappop(heap)
        if a not in comm or b not in comm or b not in e[a]:
            continue  # stale entry: one side was merged away
        g = dq(a, b)
        if abs(g + neg) > 1e-12:
            if g > 1e-12:
                heapq.heappush(heap, (-g, str(a), str(b), a, b))
            continue  # stale gain: refreshed above
        if criterion == "mapeq":   # [CHANGE 6] update the codelength terms for merging b into a
            ca, cb, eab = cut[a], cut[b], e[a][b]   # [CHANGE 6]
            cn = ca + cb - 2 * eab   # [CHANGE 6]
            sq += (cn - ca - cb) / m2   # [CHANGE 6]
            s_exit += _plogp(cn / m2) - _plogp(ca / m2) - _plogp(cb / m2)   # [CHANGE 6]
            s_mod += (_plogp((cn + deg[a] + deg[b]) / m2) - _plogp((ca + deg[a]) / m2)   # [CHANGE 6]
                      - _plogp((cb + deg[b]) / m2))   # [CHANGE 6]
            cut[a] = cn   # [CHANGE 6]
            del cut[b]   # [CHANGE 6]
            cur_L = _plogp(sq) - 2 * s_exit + s_mod   # [CHANGE 6]
            seq.append((a, b))   # [CHANGE 6]
        # merge b into a
        for c, wgt in list(e[b].items()):
            if c == a:
                continue
            e[a][c] += wgt
            e[c][a] += wgt
            del e[c][b]
        del e[b]
        e[a].pop(b, None)
        for n in comm[b]:
            new_label[n] = a
        comm[a] |= comm.pop(b)
        deg[a] += deg.pop(b)
        total += g
        merges += 1
        if criterion == "mapeq" and cur_L < best_L - 1e-12:   # [CHANGE 6] remember the best prefix
            best_L, best_k, best_total = cur_L, merges, total   # [CHANGE 6]
        for c in e[a]:
            g2 = dq(a, c)
            if g2 > 1e-12:
                x, y = (a, c) if str(a) < str(c) else (c, a)
                heapq.heappush(heap, (-g2, str(x), str(y), x, y))
    if criterion == "mapeq":   # [CHANGE 6] replay only the best prefix of merges
        new_label, root = dict(label), {}   # [CHANGE 6]
        for a, b in seq[:best_k]:   # [CHANGE 6]
            root[b] = a   # [CHANGE 6]
        def find(c):   # [CHANGE 6]
            while c in root:   # [CHANGE 6]
                c = root[c]   # [CHANGE 6]
            return c   # [CHANGE 6]
        for n in new_label:   # [CHANGE 6]
            new_label[n] = find(new_label[n])   # [CHANGE 6]
        return new_label, best_total, best_k   # [CHANGE 6]
    if total < min_gain:
        return dict(label), total, 0
    return new_label, total, merges


# <<< CHANGE 5 END ------------------------------------------------------------------------------


# >>> CHANGE 8 START: boundary node moves under the map equation, gated by the regularized map equation ----
# >>> CHANGE 9 START (helper): one module's term of the Synwalk objective --------------------------
def _synwalk_term(vol, cut, m2):
    """Synwalk (Toth, Helic & Geiger, Data Min. Knowl. Disc. 2022, Eq. 13), undirected/unweighted:
    J = sum_i p_ii log(p_i->i / p_i) + p_i,~i log(p_i-/->i / p_~i), with p_i = vol/2m, p_ii = (vol-cut)/2m,
    p_i->i = 1 - cut/vol, p_i,~i = cut/2m, p_~i = 1 - p_i. J is MAXIMISED and is a sum over modules."""
    if vol <= 0:
        return 0.0
    p = vol / m2
    t = 0.0
    if vol > cut:
        t += (vol - cut) / m2 * math.log((1 - cut / vol) / p)
    if cut > 0 and p < 1:
        t += cut / m2 * math.log((cut / vol) / (1 - p))
    return t


def synwalk_objective(G: nx.Graph, label: dict) -> float:
    """[CHANGE 9/10] Synwalk objective J of a partition (sum of _synwalk_term over modules; 0 for one module)."""
    m2 = 2.0 * G.number_of_edges()
    vol, cut = {}, {}
    for n in G.nodes():
        vol[label[n]] = vol.get(label[n], 0) + G.degree(n)
    for u, v in G.edges():
        if label[u] != label[v]:
            cut[label[u]] = cut.get(label[u], 0) + 1
            cut[label[v]] = cut.get(label[v], 0) + 1
    return sum(_synwalk_term(vol[c], cut.get(c, 0), m2) for c in vol) if m2 else 0.0


# <<< CHANGE 9 END (helper) ----------------------------------------------------------------


def boundary_moves(G: nx.Graph, NI: dict, label: dict, sweeps: int = 20, criterion: str = "mapeq"):
    """Single-node moves to a neighbouring community whenever that lowers the map codelength
    (map_codelength; Infomap's local-moving step, Rosvall & Bergstrom 2008), in NI order, until
    no node moves. There is NO aggregation step, so communities are never merged wholesale
    (seeded Infomap with aggregation collapsed LFR mu=0.7 into one module).
    The plain map equation overfits small/sparse graphs (Smiljanic et al. 2020: one Karate move
    costs NMI 0.16), so the whole pass is accepted only if the REGULARIZED codelength also falls.
    Returns (new_label, n_moves); n_moves = 0 when the pass is rejected.
    [CHANGE 9] criterion="synwalk": the move score is the Synwalk objective (_synwalk_term) instead of
    the map codelength; same gate. Synwalk reports gains over Infomap at high mixing."""
    from collections import defaultdict
    m2 = 2.0 * G.number_of_edges()
    if m2 == 0:
        return dict(label), 0
    lab = dict(label)
    vol, cut = defaultdict(float), defaultdict(float)
    for n in G.nodes():
        vol[lab[n]] += G.degree(n)
    for u, v in G.edges():
        if lab[u] != lab[v]:
            cut[lab[u]] += 1
            cut[lab[v]] += 1
    sq = sum(cut.values())   # these three sums give L up to the constant node-entropy term
    s_exit = sum(_plogp(c / m2) for c in cut.values())
    s_mod = sum(_plogp((cut[c] + vol[c]) / m2) for c in vol)

    def length(sq_, se_, sm_):
        return _plogp(sq_ / m2) - 2 * se_ + sm_

    moves = 0
    for _ in range(sweeps):
        changed = 0
        for i in _order_by_ni(G.nodes(), NI):
            a, ki = lab[i], G.degree(i)
            kc = defaultdict(int)
            for j in G.neighbors(i):
                kc[lab[j]] += 1
            kia = kc.get(a, 0)
            ca2 = cut[a] - (ki - kia) + kia
            L0 = length(sq, s_exit, s_mod)
            best, best_d, best_state = a, 0.0, None
            if criterion == "synwalk":   # [CHANGE 9] score = minus the change in the Synwalk objective
                ja = _synwalk_term(vol[a] - ki, ca2, m2) - _synwalk_term(vol[a], cut[a], m2)   # [CHANGE 9]
            for b, kib in kc.items():
                if b == a:
                    continue
                cb2 = cut[b] + (ki - kib) - kib
                if criterion == "synwalk":   # [CHANGE 9]
                    d = -(ja + _synwalk_term(vol[b] + ki, cb2, m2) - _synwalk_term(vol[b], cut[b], m2))   # [CHANGE 9]
                    if d < best_d - 1e-15:   # [CHANGE 9]
                        best, best_d, best_state = b, d, (cb2, sq, s_exit, s_mod)   # [CHANGE 9]
                    continue   # [CHANGE 9]
                sq2 = sq + (ca2 - cut[a]) + (cb2 - cut[b])
                se2 = (s_exit - _plogp(cut[a] / m2) - _plogp(cut[b] / m2)
                       + _plogp(ca2 / m2) + _plogp(cb2 / m2))
                sm2 = (s_mod - _plogp((cut[a] + vol[a]) / m2) - _plogp((cut[b] + vol[b]) / m2)
                       + _plogp((ca2 + vol[a] - ki) / m2) + _plogp((cb2 + vol[b] + ki) / m2))
                d = length(sq2, se2, sm2) - L0
                if d < best_d - 1e-12:
                    best, best_d, best_state = b, d, (cb2, sq2, se2, sm2)
            if best != a:
                cb2, sq, s_exit, s_mod = best_state
                cut[a], cut[best] = ca2, cb2
                vol[a] -= ki
                vol[best] += ki
                lab[i] = best
                changed += 1
        moves += changed
        if not changed:
            break
    if moves == 0 or map_codelength(G, lab, regularized=True) >= map_codelength(G, label, regularized=True) - 1e-12:
        return dict(label), 0
    return lab, moves


# <<< CHANGE 8 END ------------------------------------------------------------------------------


# PAPER: Algorithm 1, Step 3 = Stage 3 (WAS label filtering, Eq. 14), unchanged
def community_label_filtering(G: nx.Graph, NI: dict, label: dict, beta: float):
    """Algorithm 1, Step 3: WAS-based overlap filtering -> dict node -> set(labels)."""
    communities = {i: {label[i]} for i in G.nodes()}
    for i in G.nodes():
        neigh = list(G.neighbors(i))
        CL = {label[j] for j in neigh}
        if len(CL) > 1:
            was = {}
            denom = sum(NI[k] for k in neigh)
            for c in CL:
                numer = sum(NI[j] for j in neigh if label[j] == c)
                was[c] = numer / denom if denom > 0 else 0.0
            avg_was = sum(was.values()) / len(CL)
            valid = {c for c in CL if was[c] >= beta * avg_was}
            communities[i] |= valid
    return communities


# >>> CHANGE 11 START: Stage 3 alternative -- a second label only for near-tied nodes ---------------------
def overlap_near_tie(G: nx.Graph, label: dict, tau: float = 0.5):
    """A node keeps its own label and also joins community c if it has >= 2 links into c and
    links_into_c >= tau * links_into_own_community, i.e. it is nearly undecided between them.
    Relative-to-own variant of COPRA's belonging coefficient (Gregory, New J. Phys. 2010).
    The paper's WAS rule compares each label to the AVERAGE over neighbour labels, so a node with
    many weakly linked neighbour communities is flagged too often (SNAP-DBLP: 24.7 % flagged vs 14.9 % true)."""
    from collections import defaultdict
    out = {i: {label[i]} for i in G.nodes()}
    for i in G.nodes():
        k = defaultdict(int)
        for j in G.neighbors(i):
            k[label[j]] += 1
        own = k.get(label[i], 0)
        for c, kc in k.items():
            if c != label[i] and kc >= 2 and kc >= tau * own:
                out[i].add(c)
    return out


# <<< CHANGE 11 END ------------------------------------------------------------------------------


def tpoc_lpa(G: nx.Graph, alpha: float = 1.0, beta: float = 0.8, max_iter: int = 100,
             rng=None, sim="jaccard", agg="sum", guard=False, gamma=0.0,
             fallback=None, fallback_gamma=1.0, merge=False, merge_min_gain=0.0,
             merge_criterion="q", guard_criterion="q", moves=False, overlap="was", overlap_tau=0.5, kappa=None, refine=None, refine_gamma=1.0, refine_rho=1.5, info=None):
    """Full pipeline.

    `rng=None` (default) gives our fully deterministic variant: every tie is
    broken by a fixed rule, so repeated runs on the same graph are identical.
    Passing a `random.Random` instance reproduces the ORIGINAL paper's stated
    behaviour of breaking ties at random, which is what makes TPOC-LPA
    (like classic LPA) unstable across runs.

    Returns:
      disjoint_label : dict node -> single community id (Stage 2 output;
                        this is what Table 8/9 in the paper compares as
                        "non-overlapping community detection results")
      overlap_communities : dict node -> set(community ids) (Stage 3 output;
                        used for the overlapping EQ/NMI_max/Omega comparison)
      NI, iters

    `sim`: "jaccard" (published Eq. 11) or "tri_overlap" (motif-aware).
    `kappa`: with sim="tri_overlap", use c/(kappa+c); a number, or "auto" for
             the graph's mean common-neighbours-per-edge (see auto_kappa).
    `agg`: "sum" (published Eq. 12), "sqrt" or "log1p" (damped variants).
    `guard`: keep the best-modularity iterate of Stage 2 (see
             community_evolution) -- prevents runaway merging.
    `gamma`: > 0 replaces Eq. 12's unnormalized sum with a null-model-corrected
             gain (see community_evolution_null); 0 = as published.
    `fallback`: "collapse" -- requires guard. If the guard rejected EVERY
             Stage-2 iterate (Stage 2 only ever hurt modularity), rerun Stage 2
             from the Stage 1 partition with gamma=`fallback_gamma`.
    `refine`: "granularity" -- requires guard. Also runs the null-model Stage 2
             (gamma=`refine_gamma`) from the same Stage 1 partition and keeps it iff it
             does not fragment the published-style partition:
             n_communities(null) <= `refine_rho` * n_communities(published).
             The null-model result then acts as a boundary refiner only; when the two
             regimes disagree on granularity (small graphs whose ground truth is coarser
             than modularity's optimum: Karate, Dolphin, Polbooks) the published-style
             partition is kept. Subsumes `fallback` (a collapsed Stage 2 leaves the
             published-style partition with MORE communities than the null model).
    `merge`: after Stage 2, greedy modularity-gain merge of communities
             (community_merge), accepted only if total gain >= `merge_min_gain`.
    `merge_criterion`: "q" (CHANGE 5, total-gain gate) or "mapeq" (CHANGE 6, stop the greedy
             merge sequence at its minimum map codelength; `merge_min_gain` ignored).
    `guard_criterion`: "q" (CHANGE 2) or "mapeq" (CHANGE 7: keep the Stage-2 iterate with the
             lowest map codelength instead of the highest modularity).
    `moves`: CHANGE 8 -- after Stage 2 (and the merge pass), single-node map-equation moves
             (boundary_moves), kept only if the regularized map codelength falls.
             moves="synwalk" (CHANGE 9): same, but each move maximises the Synwalk objective.
    `overlap`: "was" (paper Stage 3, Eq. 14) or "near_tie" (CHANGE 11, overlap_near_tie with `overlap_tau`).
    `info`: optional dict, filled with {"fell_back", "refined", "merge_gain", "merges"} and,
             for time-complexity analysis, wall-clock seconds per pipeline stage:
             {"t_node_importance", "t_stage1", "t_stage2", "t_fallback", "t_refine",
              "t_merge", "t_stage3", "t_total"}. Purely diagnostic: measured with
             time.perf_counter() around each stage boundary, gated entirely behind
             `info is not None` so passing no `info` (the default, used everywhere
             else in the project) has zero timing overhead and zero effect on the
             returned labels -- verified by hash-comparing outputs before/after.
    Defaults reproduce the paper exactly.
    """
    if fallback not in (None, "collapse"):
        raise ValueError(f"unknown fallback {fallback!r}")
    if fallback and not guard:
        raise ValueError("fallback='collapse' needs guard=True (it detects the guard rejecting Stage 2)")
    if refine not in (None, "granularity"):
        raise ValueError(f"unknown refine {refine!r}")
    if refine and (not guard or fallback):
        raise ValueError("refine='granularity' needs guard=True and is exclusive with fallback")
    if kappa is not None:
        if sim != "tri_overlap":
            raise ValueError("kappa only applies to sim='tri_overlap'")
        sim = make_tri_overlap(auto_kappa(G) if kappa == "auto" else float(kappa))
    # >>> TIMING (diagnostics only, no behavior change): checkpoint timestamps around each
    # pipeline stage, gated entirely behind `info is not None` -- see the `info` docstring above.
    _t0 = time.perf_counter() if info is not None else None
    NI = node_importance(G)
    _t_ni = time.perf_counter() if info is not None else None
    label, _ = community_preselection(G, NI, alpha, rng)
    stage1 = dict(label)
    _t_stage1 = time.perf_counter() if info is not None else None
    label, iters = community_evolution(G, NI, label, max_iter, rng, sim=sim, agg=agg,
                                        guard=guard, gamma=gamma, guard_criterion=guard_criterion)
    _t_stage2 = time.perf_counter() if info is not None else None
    fell_back = False
    # >>> EXPLORED, NOT ADOPTED START: collapse fallback (safe but only fixed one graph; superseded by CHANGE 4)
    if fallback == "collapse" and label == stage1:
        sim_fn = SIMILARITIES[sim] if isinstance(sim, str) else sim
        label, iters = community_evolution_null(G, NI, dict(stage1), sim_fn, fallback_gamma,
                                                max_iter, rng)
        fell_back = True
    # <<< EXPLORED, NOT ADOPTED END
    _t_fallback = time.perf_counter() if info is not None else None
    # >>> CHANGE 4 START: granularity rule -- use the size-corrected result only if it has at most
    #     refine_rho (1.5) x as many communities as the paper-style result ------------------------------
    refined = False
    if refine == "granularity":
        sim_fn = SIMILARITIES[sim] if isinstance(sim, str) else sim
        alt, _ = community_evolution_null(G, NI, dict(stage1), sim_fn, refine_gamma, max_iter, rng)
        if len(set(alt.values())) <= refine_rho * len(set(label.values())):
            label, refined = alt, True   # [CHANGE 4] granularity rule accepted -> take the size-corrected partition
    # <<< CHANGE 4 END -------------------------------------------------------------------------
    _t_refine = time.perf_counter() if info is not None else None
    merge_gain, merges = 0.0, 0
    if merge:   # [CHANGE 5] optional merge pass after Stage 2
        label, merge_gain, merges = community_merge(G, label, merge_min_gain, merge_criterion)   # [CHANGE 5, 6]
    n_moves = 0
    if moves:   # [CHANGE 8] boundary node moves, accepted only if the regularized codelength falls
        label, n_moves = boundary_moves(G, NI, label,   # [CHANGE 8]
                                        criterion="synwalk" if moves == "synwalk" else "mapeq")   # [CHANGE 9]
    _t_merge = time.perf_counter() if info is not None else None
    disjoint_label = dict(label)
    if overlap == "near_tie":   # [CHANGE 11] Stage 3 alternative
        overlap_communities = overlap_near_tie(G, label, overlap_tau)   # [CHANGE 11]
    else:
        overlap_communities = community_label_filtering(G, NI, label, beta)   # PAPER Stage 3
    if info is not None:
        _t_stage3 = time.perf_counter()
        info.update(
            fell_back=fell_back, refined=refined, merge_gain=merge_gain, merges=merges, moves=n_moves,
            t_node_importance=_t_ni - _t0,
            t_stage1=_t_stage1 - _t_ni,
            t_stage2=_t_stage2 - _t_stage1,
            t_fallback=_t_fallback - _t_stage2,
            t_refine=_t_refine - _t_fallback,
            t_merge=_t_merge - _t_refine,
            t_stage3=_t_stage3 - _t_merge,
            t_total=_t_stage3 - _t0,
        )
    # <<< TIMING
    return disjoint_label, overlap_communities, NI, iters


# ---------------------------------------------------------------------------
# SN-LPA: final model (M1-M6 = CHANGE 1, 2, 3, 4, 6, 9)
# ---------------------------------------------------------------------------
SN_LPA_CONFIG = dict(sim="tri_overlap", guard=True, refine="granularity",
                     merge=True, merge_criterion="mapeq", moves="synwalk")


def sn_lpa(G: nx.Graph, beta: float = 0.8, rng=None, **overrides):
    """SN-LPA with the final configuration. Same return values as tpoc_lpa:
    (disjoint_label, overlap_communities, NI, iters)."""
    cfg = {**SN_LPA_CONFIG, **overrides}
    return tpoc_lpa(G, beta=beta, rng=rng, **cfg)


if __name__ == "__main__":
    from networkx.algorithms.community.quality import modularity
    G = nx.karate_club_graph()
    labels, overlap, NI, iters = sn_lpa(G)
    comms = {}
    for v, c in labels.items():
        comms.setdefault(c, set()).add(v)
    print(f"Karate: {len(comms)} communities, Q = {modularity(G, comms.values(), weight=None):.4f}")
