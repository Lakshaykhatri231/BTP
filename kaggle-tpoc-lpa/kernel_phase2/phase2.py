# %% [markdown]
# # Phase 2 — LLM-Augmented Overlapping Label Propagation
#
# Builds on the Phase 1 TPOC-LPA reproduction (separate notebook, validated:
# formula-exact, Dolphin NMI exact 0.9053, LFR n=10000 mu=0.1 exact 1.0000,
# Stage 3 validated against the paper's own worked example).
#
# Phase 2 adds **Stage 4**: budgeted semantic correction, adapting Alper's
# (KDD'26) adaptive signal-selection policy from disjoint entity resolution to
# OVERLAPPING community detection, under a *local open-weight LLM* rather than
# a paid frontier API.
#
# This notebook is the **GPU-free** part: the whole policy is validated with
# mock responders (oracle / noisy oracle / random) so the algorithm is proven
# correct, and its operating requirements are known, before any model is loaded.
#
# Three experiments:
# 1. **Determinism** — does removing the paper's random tie-breaking change anything?
# 2. **Headroom** — what is the best a perfect semantic corrector could do?
# 3. **Fusion** — how should an *unreliable* LLM be combined with structure?

# %%
import math
import random
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community.quality import modularity

import os, glob

def _find_data_dir():
    """Locate the directory holding football.gml/dolphins.gml/lesmis.gml.
    Kaggle's mount point for an attached dataset is not always the slug we
    expect, so search rather than hardcode; fall back to the local copy when
    running outside Kaggle."""
    for cand in ["/kaggle/input/tpoc-lpa-benchmark-networks", "dataset", "../dataset"]:
        if os.path.isfile(os.path.join(cand, "football.gml")):
            return cand
    hits = glob.glob("/kaggle/input/**/football.gml", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError(
        "football.gml not found. /kaggle/input contains: "
        + str(glob.glob("/kaggle/input/*")))

DATA = _find_data_dir()
print("using data dir:", DATA)


# %% [markdown]
# ## Phase 1 core (inlined, unmodified)


# %%
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


def jaccard(G: nx.Graph, i, j) -> float:
    Ni, Nj = set(G.neighbors(i)), set(G.neighbors(j))
    union = Ni | Nj
    return len(Ni & Nj) / len(union) if union else 0.0


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


def community_evolution(G: nx.Graph, NI: dict, label: dict, max_iter: int = 100, rng=None):
    """Algorithm 1, Step 2: attractiveness-guided label reassignment."""
    order = _order_by_ni(G.nodes(), NI, rng)
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
                    A_ic = 0.0
                    for j in neigh:
                        if label[j] == c:
                            Jij = jaccard(G, i, j)
                            A_ic += (NI[i] * NI[j]) / ((2 - Jij) ** 2)
                    scores[c] = A_ic
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
        if not changed:
            break
    return label, it


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


def tpoc_lpa(G: nx.Graph, alpha: float = 1.0, beta: float = 0.8, max_iter: int = 100,
             rng=None):
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
    """
    NI = node_importance(G)
    label, _ = community_preselection(G, NI, alpha, rng)
    label, iters = community_evolution(G, NI, label, max_iter, rng)
    disjoint_label = dict(label)
    overlap_communities = community_label_filtering(G, NI, label, beta)
    return disjoint_label, overlap_communities, NI, iters

# %% [markdown]
# ## Stage 4: budgeted semantic correction


# %%
def was_distribution(G, NI, label, node):
    """WAS(i,c) over the communities node i's neighbours span (Eq. 14).
    Values sum to 1."""
    neigh = list(G.neighbors(node))
    if not neigh:
        return {}
    denom = sum(NI[k] for k in neigh)
    if denom <= 0:
        return {}
    dist = {}
    for j in neigh:
        dist[label[j]] = dist.get(label[j], 0.0) + NI[j]
    return {c: v / denom for c, v in dist.items()}


def structural_confidence(dist):
    """Alper Eq. 6 analog. Perplexity of the affiliation distribution turned
    into a confidence in [0,1]: peaked distribution -> high confidence,
    flat distribution -> low confidence (worth querying)."""
    if not dist:
        return 1.0
    k = len(dist)
    if k == 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in dist.values() if p > 0)
    perplexity = math.exp(entropy)
    # log_k(perplexity) in [0,1]: 1 = maximally uncertain, 0 = certain
    uncertainty = math.log(perplexity) / math.log(k)
    return (1.0 - uncertainty) * max(dist.values())


def _knapsack_threshold(spent, budget, L=0.01, U=1.0):
    """Online-knapsack admission threshold (Alper Thm 4.3 / Zhou et al.).
    Starts low and rises exponentially as budget depletes, so early cheap
    wins are taken but late budget is reserved for high-value queries."""
    if budget <= 0:
        return float('inf')
    frac = min(1.0, spent / budget)
    return (L / math.e) * ((U * math.e / L) ** frac)


def stage4_semantic_correction(G, disjoint_label, beta=0.8, budget=None,
                                responder=None, delta_llm=0.95, top_m=3,
                                NI=None, verbose=False, fusion="hard",
                                trust=None):
    """Returns (new_disjoint_label, overlap_communities, stats).

    `responder(node, candidates, context)` -> set of accepted community ids
    (a subset of `candidates`); returning an empty set means "none of these",
    which is left as a no-op.

    fusion:
      "hard" -- the LLM's answer replaces the structural one (Alper's
                behaviour; appropriate when the model is a reliable frontier
                LLM). Measured here to collapse once reliability drops below
                ~0.9, which is the regime a small local open-weight model
                actually lives in.
      "soft" -- blend the LLM vote with the structural WAS distribution:
                    score(c) = (1-w)*WAS(i,c) + w*vote(c)
                with w = `trust` (defaults to delta_llm). The method then
                degrades gracefully towards the pure-structural result as the
                model gets less reliable, instead of being dragged below it.
    """
    if NI is None:
        NI = node_importance(G)
    label = dict(disjoint_label)
    n_boundary = sum(
        1 for i in G.nodes() if len({label[j] for j in G.neighbors(i)}) > 1)
    if budget is None:
        budget = n_boundary  # unlimited-ish: allow one query per boundary node

    # Rank boundary nodes by how much a query could help (Alper's marginal
    # value gain g_i), so the budget goes to the most ambiguous nodes first.
    scored = []
    for i in G.nodes():
        dist = was_distribution(G, NI, label, i)
        if len(dist) < 2:
            continue  # not a boundary node: nothing to correct
        conf = structural_confidence(dist)
        gain = max(0.0, delta_llm - conf)
        scored.append((gain, conf, i, dist))
    scored.sort(key=lambda t: (-t[0], str(t[2])))

    spent = 0
    queried, accepted_changes, skipped_by_policy = 0, 0, 0
    overlap_extra = {}

    for gain, conf, i, dist in scored:
        if spent >= budget:
            skipped_by_policy += 1
            continue
        if gain < _knapsack_threshold(spent, budget):
            skipped_by_policy += 1
            continue  # structural signal deemed good enough

        candidates = [c for c, _ in sorted(dist.items(), key=lambda kv: -kv[1])][:top_m]
        context = {"was": dist, "confidence": conf, "graph": G, "label": label}
        result = responder(i, candidates, context) if responder else set()
        queried += 1
        spent += 1

        if result:
            if fusion == "hard":
                # LLM answer replaces the structural one
                best = max(result, key=lambda c: dist.get(c, 0.0))
                chosen = set(result)
            else:
                # Bayesian update: treat the WAS distribution as the prior and
                # the LLM's answer as evidence from a channel that is correct
                # with probability p. Linear interpolation does NOT work here:
                # a binary vote of 1.0 swamps a WAS distribution whose max is
                # ~0.5-0.7, so the LLM would always win for any weight >= 0.5.
                #
                #   P(answer = c* | true = c*) = p
                #   P(answer = c* | true = c)  = (1-p)/(k-1)   for c != c*
                #   posterior(c) ∝ prior(c) * likelihood(answer | c)
                #
                # When p -> 1/k (an uninformative model) the posterior collapses
                # back onto the prior, i.e. the method degrades gracefully to
                # the pure-structural result instead of being dragged below it.
                p = delta_llm if trust is None else trust
                universe = set(dist) | set(result)
                k = max(len(universe), 2)
                p = min(max(p, 1.0 / k), 1.0 - 1e-9)
                wrong = (1.0 - p) / (k - 1)
                post = {}
                for c in universe:
                    prior = dist.get(c, 0.0)
                    like = p if c in result else wrong
                    post[c] = prior * like
                total = sum(post.values())
                if total <= 0:
                    post = {c: 1.0 / len(universe) for c in universe}
                else:
                    post = {c: v / total for c, v in post.items()}
                best = max(post, key=lambda c: (post[c], str(c)))
                avg = sum(post.values()) / len(post)
                chosen = {c for c, v in post.items() if v >= beta * avg}
            if label[i] != best:
                accepted_changes += 1
            label[i] = best
            if len(chosen) > 1:
                overlap_extra[i] = set(chosen)

    # Stage 3 filtering on the corrected labels, then merge in any
    # LLM-asserted multi-membership.
    overlap = community_label_filtering(G, NI, label, beta)
    for i, extra in overlap_extra.items():
        overlap[i] |= extra

    stats = {
        "boundary_nodes": n_boundary,
        "candidates_considered": len(scored),
        "queried": queried,
        "budget": budget,
        "label_changes": accepted_changes,
        "skipped_by_policy": skipped_by_policy,
        "overlap_nodes": sum(1 for i in G.nodes() if len(overlap[i]) > 1),
    }
    if verbose:
        print("   stage4 stats:", stats)
    return label, overlap, stats


# %% [markdown]
# ## Helpers

# %%
def to_partition(label_dict, nodes):
    groups = {}
    for n in nodes:
        groups.setdefault(label_dict[n], set()).add(n)
    return list(groups.values())


def partition_signature(label_dict, nodes):
    groups = {}
    for n in nodes:
        groups.setdefault(label_dict[n], set()).add(n)
    return frozenset(frozenset(sorted(map(str, g))) for g in groups.values())


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var ** 0.5


def majority_map(truth, label, nodes):
    from collections import Counter, defaultdict
    counts = defaultdict(Counter)
    for n in nodes:
        counts[truth[n]][label[n]] += 1
    return {gt: c.most_common(1)[0][0] for gt, c in counts.items()}


def make_noisy_oracle(truth, gt_to_comm, p, seed=0):
    rng = random.Random(seed)
    def responder(node, candidates, context):
        gt_comm = gt_to_comm.get(truth[node])
        if rng.random() < p and gt_comm is not None:
            return {gt_comm}
        wrong = [c for c in candidates if c != gt_comm]
        return {rng.choice(wrong)} if wrong else set()
    return responder


def make_random_responder(seed=0):
    rng = random.Random(seed)
    def responder(node, candidates, context):
        return {rng.choice(candidates)} if candidates else set()
    return responder


G_foot = nx.read_gml(f"{DATA}/football.gml", label="id")
TRUTH_FOOT = {n: G_foot.nodes[n]["value"] for n in G_foot.nodes()}
G_kar = nx.karate_club_graph()
TRUTH_KAR = {n: (0 if G_kar.nodes[n]["club"] == "Mr. Hi" else 1) for n in G_kar.nodes()}
G_dol = nx.read_gml(f"{DATA}/dolphins.gml")
G_les = nx.Graph(nx.read_gml(f"{DATA}/lesmis.gml"))

# %% [markdown]
# ## Experiment 1 — Determinism
#
# TPOC-LPA as published breaks ties at random (stated explicitly in its Fig. 4
# walkthrough). Our variant fixes every tie deterministically. Does that
# actually buy any stability? 15 runs of each variant, reporting std and the
# number of DISTINCT partitions produced.

# %%
N_RUNS = 15

def determinism_report(name, G, truth):
    nodes = list(G.nodes())
    print(f"--- {name} ({G.number_of_nodes()}n/{G.number_of_edges()}e), {N_RUNS} runs each ---")
    for random_ties, tag in [(True, "paper (random ties)"), (False, "ours (deterministic)")]:
        nmis, qs, sigs = [], [], set()
        for r in range(N_RUNS):
            rng = random.Random(1000 + r) if random_ties else None
            dl, oc, NI, iters = tpoc_lpa(G, alpha=1.0, beta=0.8, rng=rng)
            if truth is not None:
                nmis.append(normalized_mutual_info_score(
                    [truth[n] for n in nodes], [dl[n] for n in nodes]))
            qs.append(modularity(G, to_partition(dl, nodes)))
            sigs.add(partition_signature(dl, nodes))
        qm, qsd = stats(qs)
        line = f"  {tag:22s} Q = {qm:.4f} +/- {qsd:.4f}"
        if nmis:
            nm, nsd = stats(nmis)
            line += f"   NMI = {nm:.4f} +/- {nsd:.4f}"
        line += f"   distinct partitions: {len(sigs)}/{N_RUNS}"
        print(line)
    print()

# how often are there actually ties to break?
from collections import Counter
for nm, Gx in [("Karate", G_kar), ("Dolphin", G_dol), ("Football", G_foot), ("Lesmis", G_les)]:
    NIx = node_importance(Gx)
    c = Counter(NIx.values())
    tied_nodes = sum(v for v in c.values() if v > 1)
    print(f"{nm}: {tied_nodes}/{Gx.number_of_nodes()} nodes sit in an exact NI tie")
print()

determinism_report("Karate", G_kar, TRUTH_KAR)
determinism_report("Dolphin", G_dol, None)
determinism_report("Football", G_foot, TRUTH_FOOT)
determinism_report("Lesmis", G_les, None)

# %% [markdown]
# ## Experiment 2 — Headroom
#
# Before spending GPU on a real model: what is the BEST any semantic corrector
# could achieve? A perfect oracle gives the upper bound. If the oracle barely
# helps, no LLM can.

# %%
def headroom(name, G, truth, budgets=(10, 25, 50, 1000)):
    nodes = list(G.nodes())
    y_true = [truth[n] for n in nodes]
    NI = node_importance(G)
    base_label, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8)
    base = normalized_mutual_info_score(y_true, [base_label[n] for n in nodes])
    print(f"=== {name} === structural baseline NMI = {base:.4f}, "
          f"comms={len(set(base_label.values()))}, truth={len(set(y_true))}")
    gt_to_comm = majority_map(truth, base_label, nodes)
    for B in budgets:
        resp = make_noisy_oracle(truth, gt_to_comm, p=1.0, seed=1)
        lab, ov, st = stage4_semantic_correction(
            G, base_label, beta=0.8, budget=B, responder=resp, NI=NI, fusion="hard")
        nmi = normalized_mutual_info_score(y_true, [lab[n] for n in nodes])
        print(f"  budget={B:<5} queried={st['queried']:<4} changes={st['label_changes']:<4} "
              f"NMI={nmi:.4f} ({nmi-base:+.4f})  overlap_nodes={st['overlap_nodes']}")
    print()

headroom("Football", G_foot, TRUTH_FOOT)
headroom("Karate", G_kar, TRUTH_KAR, budgets=(5, 10, 34))

# %% [markdown]
# ## Experiment 3 — Fusion: surviving an unreliable LLM
#
# The headroom sweep shows a sharp reliability cliff: hard override (Alper's
# behaviour, designed for a reliable frontier model) becomes HARMFUL below
# ~90% responder accuracy -- which is plausibly where a 4-bit quantized local
# 7B model lives.
#
# Our fix treats the WAS distribution as a Bayesian prior and the LLM answer as
# evidence from a channel correct with probability p. As p -> 1/k the posterior
# collapses onto the prior, so the method degrades gracefully to the
# pure-structural result. Note linear interpolation does NOT work here: a
# binary vote of 1.0 swamps a WAS distribution whose max is ~0.5-0.7, so the
# LLM would win for any weight >= 0.5 regardless of structural evidence.
#
# Columns compare: hard override | Bayesian fusion with CALIBRATED trust
# (w = true p) | Bayesian fusion with Alper's HARDCODED trust (w = 0.95).

# %%
def fusion_sweep(name, G, truth, budget=50, n_seeds=5):
    nodes = list(G.nodes())
    y_true = [truth[n] for n in nodes]
    NI = node_importance(G)
    base_label, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8)
    base = normalized_mutual_info_score(y_true, [base_label[n] for n in nodes])
    gt_to_comm = majority_map(truth, base_label, nodes)
    print(f"=== {name} === structural baseline NMI = {base:.4f}")
    print(f"{'reliability p':>13} | {'hard override':>17} | {'soft CALIBRATED':>17} | {'soft w=0.95':>17}")
    print("-" * 76)
    for p in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3):
        row = {}
        for mode, kw in [("hard", dict(fusion="hard")),
                          ("cal", dict(fusion="soft", trust=p)),
                          ("fixed", dict(fusion="soft", trust=0.95))]:
            vals = []
            for seed in range(n_seeds):
                resp = make_noisy_oracle(truth, gt_to_comm, p=p, seed=seed)
                lab, ov, st = stage4_semantic_correction(
                    G, base_label, beta=0.8, budget=budget, responder=resp, NI=NI, **kw)
                vals.append(normalized_mutual_info_score(y_true, [lab[n] for n in nodes]))
            row[mode] = sum(vals) / len(vals)
        f = lambda v: f"{v:.4f} ({v-base:+.4f})"
        print(f"{p:>13} | {f(row['hard']):>17} | {f(row['cal']):>17} | {f(row['fixed']):>17}")
    print()

fusion_sweep("Football", G_foot, TRUTH_FOOT)
fusion_sweep("Karate", G_kar, TRUTH_KAR)
