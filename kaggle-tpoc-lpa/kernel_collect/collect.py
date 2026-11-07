# %% [markdown]
# # Phase 2 — LLM evidence collection (GPU) for offline policy evaluation
#
# The first real-model run (Qwen2.5-7B, Football) made Stage 4 *worse*
# (NMI 0.9095 -> 0.7815) and, because it asked the model live and took one
# hard letter per node, could not say why. This notebook restructures the
# experiment so the cause can be found:
#
#  1. cache the model's SOFT preference (letter probabilities from the logits)
#     for EVERY boundary node once, on GPU;
#  2. evaluate any selection / fusion policy OFFLINE on CPU (no more GPU runs);
#  3. use cyclic permutations of candidate order so position bias cancels;
#  4. show up to 12 members per candidate, most central first, instead of 6
#     arbitrary ones (both variants are collected, to measure the effect);
#  5. two semantic graphs WITH ground truth -- Football (teams -> conference)
#     and Polbooks (book titles -> political leaning) -- so a rule fitted on
#     one can be tested on the other;
#  6. several models, to see whether conclusions are model-specific.

# %%
import os, re, json, glob, math, random, time, gc
import networkx as nx

T_START = time.time()
def log(*a):
    print(f"[{time.time()-T_START:7.1f}s]", *a, flush=True)


# %% [markdown]
# ## Phase 1 core + Stage 4 helpers (inlined, unmodified)


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
# ## Evidence collection
#
# For every boundary node we record the LLM's *soft* preference over the
# node's candidate communities, and cache it, so every fusion/selection
# policy can then be evaluated OFFLINE (CPU) with no further GPU runs.
#
# Two design choices that fix the problems found in the first real-model run:
#  * **Letter probabilities from the logits**, not a sampled letter. One
#    forward pass, deterministic, and gives a distribution rather than a hard
#    vote.
#  * **Cyclic permutations** of the candidate order: with k candidates every
#    community appears in every position exactly once, so any position bias
#    (the model preferring 'A') cancels exactly in the average. The first run
#    listed candidates in structural-ranking order, so answering 'A' meant
#    "agree with structure" -- a confound.
# The per-permutation probabilities are stored too, so the position bias
# itself can be measured.

# %%
LETTERS = "ABC"
SYSTEM = "You are an expert at grouping entities. Answer with a single option letter only."


def members_of(G, label, comm, exclude, NI, k):
    """Most central members first (rather than the first k in node order),
    excluding the node being asked about."""
    ms = [n for n in G.nodes() if label[n] == comm and n != exclude]
    ms.sort(key=lambda n: (-NI[n], str(n)))
    return ms[:k]


def build_prompt(G, label, NI, node, ordered_cands, hint, k_members):
    lines = [hint, "", f"Item: {node}", "", "Candidate groups (example members):"]
    for idx, c in enumerate(ordered_cands):
        ms = members_of(G, label, c, node, NI, k_members)
        lines.append(f"  {LETTERS[idx]}: " + (", ".join(map(str, ms)) if ms else "(no other members)"))
    lines.append("")
    lines.append(f"Which ONE group does '{node}' most naturally belong to? "
                 f"Reply with a single letter ({'/'.join(LETTERS[:len(ordered_cands)])}).")
    return "\n".join(lines)


def cyclic_orders(cands):
    k = len(cands)
    return [cands[r:] + cands[:r] for r in range(k)]


def collect_dataset(name, G, truth, hint, letter_probs_fn, k_members_variants=(6, 12),
                    top_m=3, log_every=20, log=print):
    """letter_probs_fn(list_of_prompts, n_letters_list) -> list of
    (probs_over_letters, letter_mass) using the model under test."""
    NI = node_importance(G)
    base, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8)
    records = []
    boundary = []
    for i in G.nodes():
        dist = was_distribution(G, NI, base, i)
        if len(dist) >= 2:
            cands = [c for c, _ in sorted(dist.items(), key=lambda kv: (-kv[1], str(kv[0])))][:top_m]
            boundary.append((i, cands, dist))
    log(f"[{name}] {len(boundary)} boundary nodes of {G.number_of_nodes()}; "
        f"prompt variants (members shown): {list(k_members_variants)}")
    t0 = time.time()
    for idx, (node, cands, dist) in enumerate(boundary):
        rec = {"node": str(node), "truth": str(truth[node]), "base": str(base[node]),
               "cands": [str(c) for c in cands],
               "was": {str(c): float(v) for c, v in dist.items()}, "variants": {}}
        for kmem in k_members_variants:
            orders = cyclic_orders(cands)
            prompts = [build_prompt(G, base, NI, node, o, hint, kmem) for o in orders]
            outs = letter_probs_fn(prompts, [len(o) for o in orders])
            per_perm = []
            ell = {str(c): 0.0 for c in cands}
            for o, (probs, mass) in zip(orders, outs):
                per_perm.append({"order": [str(c) for c in o],
                                 "probs": [float(p) for p in probs],
                                 "letter_mass": float(mass)})
                for pos, c in enumerate(o):
                    ell[str(c)] += float(probs[pos]) / len(orders)
            rec["variants"][str(kmem)] = {"ell": ell, "perms": per_perm}
        records.append(rec)
        if idx % log_every == 0:
            log(f"[{name}] node {idx}/{len(boundary)}  ({(time.time()-t0)/max(idx,1):.2f}s/node)")
    return {"dataset": name, "nodes": records,
            "base_label": {str(n): str(base[n]) for n in G.nodes()},
            "truth": {str(n): str(truth[n]) for n in G.nodes()}}



# %%
POLBOOKS_GML = r"""Creator "Mark Newman on Wed Oct 18 16:42:04 2006"
graph
[
  directed 0
  node
  [
    id 0
    label "1000 Years for Revenge"
    value "n"
  ]
  node
  [
    id 1
    label "Bush vs. the Beltway"
    value "c"
  ]
  node
  [
    id 2
    label "Charlie Wilson's War"
    value "c"
  ]
  node
  [
    id 3
    label "Losing Bin Laden"
    value "c"
  ]
  node
  [
    id 4
    label "Sleeping With the Devil"
    value "n"
  ]
  node
  [
    id 5
    label "The Man Who Warned America"
    value "c"
  ]
  node
  [
    id 6
    label "Why America Slept"
    value "n"
  ]
  node
  [
    id 7
    label "Ghost Wars"
    value "n"
  ]
  node
  [
    id 8
    label "A National Party No More"
    value "c"
  ]
  node
  [
    id 9
    label "Bush Country"
    value "c"
  ]
  node
  [
    id 10
    label "Dereliction of Duty"
    value "c"
  ]
  node
  [
    id 11
    label "Legacy"
    value "c"
  ]
  node
  [
    id 12
    label "Off with Their Heads"
    value "c"
  ]
  node
  [
    id 13
    label "Persecution"
    value "c"
  ]
  node
  [
    id 14
    label "Rumsfeld's War"
    value "c"
  ]
  node
  [
    id 15
    label "Breakdown"
    value "c"
  ]
  node
  [
    id 16
    label "Betrayal"
    value "c"
  ]
  node
  [
    id 17
    label "Shut Up and Sing"
    value "c"
  ]
  node
  [
    id 18
    label "Meant To Be"
    value "n"
  ]
  node
  [
    id 19
    label "The Right Man"
    value "c"
  ]
  node
  [
    id 20
    label "Ten Minutes from Normal"
    value "c"
  ]
  node
  [
    id 21
    label "Hillary's Scheme"
    value "c"
  ]
  node
  [
    id 22
    label "The French Betrayal of America"
    value "c"
  ]
  node
  [
    id 23
    label "Tales from the Left Coast"
    value "c"
  ]
  node
  [
    id 24
    label "Hating America"
    value "c"
  ]
  node
  [
    id 25
    label "The Third Terrorist"
    value "c"
  ]
  node
  [
    id 26
    label "Endgame"
    value "c"
  ]
  node
  [
    id 27
    label "Spin Sisters"
    value "c"
  ]
  node
  [
    id 28
    label "All the Shah's Men"
    value "n"
  ]
  node
  [
    id 29
    label "Dangerous Dimplomacy"
    value "c"
  ]
  node
  [
    id 30
    label "The Price of Loyalty"
    value "l"
  ]
  node
  [
    id 31
    label "House of Bush, House of Saud"
    value "l"
  ]
  node
  [
    id 32
    label "The Death of Right and Wrong"
    value "c"
  ]
  node
  [
    id 33
    label "Useful Idiots"
    value "c"
  ]
  node
  [
    id 34
    label "The O'Reilly Factor"
    value "c"
  ]
  node
  [
    id 35
    label "Let Freedom Ring"
    value "c"
  ]
  node
  [
    id 36
    label "Those Who Trespass"
    value "c"
  ]
  node
  [
    id 37
    label "Bias"
    value "c"
  ]
  node
  [
    id 38
    label "Slander"
    value "c"
  ]
  node
  [
    id 39
    label "The Savage Nation"
    value "c"
  ]
  node
  [
    id 40
    label "Deliver Us from Evil"
    value "c"
  ]
  node
  [
    id 41
    label "Give Me a Break"
    value "c"
  ]
  node
  [
    id 42
    label "The Enemy Within"
    value "c"
  ]
  node
  [
    id 43
    label "The Real America"
    value "c"
  ]
  node
  [
    id 44
    label "Who's Looking Out for You?"
    value "c"
  ]
  node
  [
    id 45
    label "The Official Handbook Vast Right Wing Conspiracy"
    value "c"
  ]
  node
  [
    id 46
    label "Power Plays"
    value "n"
  ]
  node
  [
    id 47
    label "Arrogance"
    value "c"
  ]
  node
  [
    id 48
    label "The Perfect Wife"
    value "n"
  ]
  node
  [
    id 49
    label "The Bushes"
    value "c"
  ]
  node
  [
    id 50
    label "Things Worth Fighting For"
    value "c"
  ]
  node
  [
    id 51
    label "Surprise, Security, the American Experience"
    value "n"
  ]
  node
  [
    id 52
    label "Allies"
    value "c"
  ]
  node
  [
    id 53
    label "Why Courage Matters"
    value "c"
  ]
  node
  [
    id 54
    label "Hollywood Interrupted"
    value "c"
  ]
  node
  [
    id 55
    label "Fighting Back"
    value "c"
  ]
  node
  [
    id 56
    label "We Will Prevail"
    value "c"
  ]
  node
  [
    id 57
    label "The Faith of George W Bush"
    value "c"
  ]
  node
  [
    id 58
    label "Rise of the Vulcans"
    value "c"
  ]
  node
  [
    id 59
    label "Downsize This!"
    value "l"
  ]
  node
  [
    id 60
    label "Stupid White Men"
    value "l"
  ]
  node
  [
    id 61
    label "Rush Limbaugh Is a Big Fat Idiot"
    value "l"
  ]
  node
  [
    id 62
    label "The Best Democracy Money Can Buy"
    value "l"
  ]
  node
  [
    id 63
    label "The Culture of Fear"
    value "l"
  ]
  node
  [
    id 64
    label "America Unbound"
    value "l"
  ]
  node
  [
    id 65
    label "The Choice"
    value "l"
  ]
  node
  [
    id 66
    label "The Great Unraveling"
    value "l"
  ]
  node
  [
    id 67
    label "Rogue Nation"
    value "l"
  ]
  node
  [
    id 68
    label "Soft Power"
    value "l"
  ]
  node
  [
    id 69
    label "Colossus"
    value "n"
  ]
  node
  [
    id 70
    label "The Sorrows of Empire"
    value "l"
  ]
  node
  [
    id 71
    label "Against All Enemies"
    value "l"
  ]
  node
  [
    id 72
    label "American Dynasty"
    value "l"
  ]
  node
  [
    id 73
    label "Big Lies"
    value "l"
  ]
  node
  [
    id 74
    label "The Lies of George W. Bush"
    value "l"
  ]
  node
  [
    id 75
    label "Worse Than Watergate"
    value "l"
  ]
  node
  [
    id 76
    label "Plan of Attack"
    value "n"
  ]
  node
  [
    id 77
    label "Bush at War"
    value "c"
  ]
  node
  [
    id 78
    label "The New Pearl Harbor"
    value "l"
  ]
  node
  [
    id 79
    label "Bushwomen"
    value "l"
  ]
  node
  [
    id 80
    label "The Bubble of American Supremacy"
    value "l"
  ]
  node
  [
    id 81
    label "Living History"
    value "l"
  ]
  node
  [
    id 82
    label "The Politics of Truth"
    value "l"
  ]
  node
  [
    id 83
    label "Fanatics and Fools"
    value "l"
  ]
  node
  [
    id 84
    label "Bushwhacked"
    value "l"
  ]
  node
  [
    id 85
    label "Disarming Iraq"
    value "l"
  ]
  node
  [
    id 86
    label "Lies and the Lying Liars Who Tell Them"
    value "l"
  ]
  node
  [
    id 87
    label "MoveOn's 50 Ways to Love Your Country"
    value "l"
  ]
  node
  [
    id 88
    label "The Buying of the President 2004"
    value "l"
  ]
  node
  [
    id 89
    label "Perfectly Legal"
    value "l"
  ]
  node
  [
    id 90
    label "Hegemony or Survival"
    value "l"
  ]
  node
  [
    id 91
    label "The Exception to the Rulers"
    value "l"
  ]
  node
  [
    id 92
    label "Freethinkers"
    value "l"
  ]
  node
  [
    id 93
    label "Had Enough?"
    value "l"
  ]
  node
  [
    id 94
    label "It's Still the Economy, Stupid!"
    value "l"
  ]
  node
  [
    id 95
    label "We're Right They're Wrong"
    value "l"
  ]
  node
  [
    id 96
    label "What Liberal Media?"
    value "l"
  ]
  node
  [
    id 97
    label "The Clinton Wars"
    value "l"
  ]
  node
  [
    id 98
    label "Weapons of Mass Deception"
    value "l"
  ]
  node
  [
    id 99
    label "Dude, Where's My Country?"
    value "l"
  ]
  node
  [
    id 100
    label "Thieves in High Places"
    value "l"
  ]
  node
  [
    id 101
    label "Shrub"
    value "l"
  ]
  node
  [
    id 102
    label "Buck Up Suck Up"
    value "l"
  ]
  node
  [
    id 103
    label "The Future of Freedom"
    value "n"
  ]
  node
  [
    id 104
    label "Empire"
    value "n"
  ]
  edge
  [
    source 1
    target 0
  ]
  edge
  [
    source 2
    target 0
  ]
  edge
  [
    source 3
    target 0
  ]
  edge
  [
    source 3
    target 1
  ]
  edge
  [
    source 4
    target 0
  ]
  edge
  [
    source 4
    target 2
  ]
  edge
  [
    source 5
    target 0
  ]
  edge
  [
    source 5
    target 1
  ]
  edge
  [
    source 5
    target 2
  ]
  edge
  [
    source 5
    target 3
  ]
  edge
  [
    source 5
    target 4
  ]
  edge
  [
    source 6
    target 0
  ]
  edge
  [
    source 6
    target 1
  ]
  edge
  [
    source 6
    target 4
  ]
  edge
  [
    source 6
    target 5
  ]
  edge
  [
    source 7
    target 2
  ]
  edge
  [
    source 7
    target 5
  ]
  edge
  [
    source 7
    target 6
  ]
  edge
  [
    source 8
    target 3
  ]
  edge
  [
    source 9
    target 3
  ]
  edge
  [
    source 9
    target 8
  ]
  edge
  [
    source 10
    target 3
  ]
  edge
  [
    source 10
    target 6
  ]
  edge
  [
    source 10
    target 8
  ]
  edge
  [
    source 11
    target 3
  ]
  edge
  [
    source 11
    target 8
  ]
  edge
  [
    source 11
    target 9
  ]
  edge
  [
    source 11
    target 10
  ]
  edge
  [
    source 12
    target 3
  ]
  edge
  [
    source 12
    target 6
  ]
  edge
  [
    source 12
    target 8
  ]
  edge
  [
    source 12
    target 9
  ]
  edge
  [
    source 12
    target 10
  ]
  edge
  [
    source 12
    target 11
  ]
  edge
  [
    source 13
    target 3
  ]
  edge
  [
    source 13
    target 8
  ]
  edge
  [
    source 13
    target 11
  ]
  edge
  [
    source 13
    target 12
  ]
  edge
  [
    source 14
    target 3
  ]
  edge
  [
    source 14
    target 8
  ]
  edge
  [
    source 14
    target 9
  ]
  edge
  [
    source 14
    target 11
  ]
  edge
  [
    source 14
    target 12
  ]
  edge
  [
    source 14
    target 7
  ]
  edge
  [
    source 15
    target 3
  ]
  edge
  [
    source 15
    target 10
  ]
  edge
  [
    source 15
    target 12
  ]
  edge
  [
    source 16
    target 3
  ]
  edge
  [
    source 16
    target 10
  ]
  edge
  [
    source 16
    target 15
  ]
  edge
  [
    source 17
    target 3
  ]
  edge
  [
    source 17
    target 11
  ]
  edge
  [
    source 17
    target 12
  ]
  edge
  [
    source 17
    target 13
  ]
  edge
  [
    source 18
    target 3
  ]
  edge
  [
    source 18
    target 6
  ]
  edge
  [
    source 18
    target 12
  ]
  edge
  [
    source 19
    target 3
  ]
  edge
  [
    source 19
    target 10
  ]
  edge
  [
    source 20
    target 3
  ]
  edge
  [
    source 20
    target 8
  ]
  edge
  [
    source 20
    target 9
  ]
  edge
  [
    source 20
    target 11
  ]
  edge
  [
    source 21
    target 3
  ]
  edge
  [
    source 21
    target 8
  ]
  edge
  [
    source 21
    target 10
  ]
  edge
  [
    source 21
    target 11
  ]
  edge
  [
    source 22
    target 3
  ]
  edge
  [
    source 22
    target 6
  ]
  edge
  [
    source 22
    target 8
  ]
  edge
  [
    source 22
    target 11
  ]
  edge
  [
    source 23
    target 3
  ]
  edge
  [
    source 23
    target 8
  ]
  edge
  [
    source 23
    target 12
  ]
  edge
  [
    source 23
    target 21
  ]
  edge
  [
    source 24
    target 3
  ]
  edge
  [
    source 24
    target 8
  ]
  edge
  [
    source 24
    target 9
  ]
  edge
  [
    source 24
    target 12
  ]
  edge
  [
    source 24
    target 20
  ]
  edge
  [
    source 25
    target 3
  ]
  edge
  [
    source 25
    target 6
  ]
  edge
  [
    source 25
    target 14
  ]
  edge
  [
    source 25
    target 22
  ]
  edge
  [
    source 26
    target 3
  ]
  edge
  [
    source 26
    target 8
  ]
  edge
  [
    source 26
    target 11
  ]
  edge
  [
    source 26
    target 14
  ]
  edge
  [
    source 26
    target 24
  ]
  edge
  [
    source 27
    target 3
  ]
  edge
  [
    source 27
    target 8
  ]
  edge
  [
    source 27
    target 9
  ]
  edge
  [
    source 27
    target 11
  ]
  edge
  [
    source 27
    target 23
  ]
  edge
  [
    source 28
    target 4
  ]
  edge
  [
    source 29
    target 4
  ]
  edge
  [
    source 29
    target 6
  ]
  edge
  [
    source 29
    target 11
  ]
  edge
  [
    source 29
    target 13
  ]
  edge
  [
    source 30
    target 4
  ]
  edge
  [
    source 30
    target 7
  ]
  edge
  [
    source 31
    target 4
  ]
  edge
  [
    source 31
    target 30
  ]
  edge
  [
    source 32
    target 8
  ]
  edge
  [
    source 32
    target 12
  ]
  edge
  [
    source 32
    target 13
  ]
  edge
  [
    source 32
    target 23
  ]
  edge
  [
    source 33
    target 32
  ]
  edge
  [
    source 33
    target 8
  ]
  edge
  [
    source 33
    target 10
  ]
  edge
  [
    source 33
    target 12
  ]
  edge
  [
    source 33
    target 23
  ]
  edge
  [
    source 35
    target 34
  ]
  edge
  [
    source 35
    target 8
  ]
  edge
  [
    source 35
    target 10
  ]
  edge
  [
    source 36
    target 34
  ]
  edge
  [
    source 36
    target 35
  ]
  edge
  [
    source 36
    target 12
  ]
  edge
  [
    source 37
    target 34
  ]
  edge
  [
    source 37
    target 8
  ]
  edge
  [
    source 37
    target 10
  ]
  edge
  [
    source 37
    target 35
  ]
  edge
  [
    source 37
    target 33
  ]
  edge
  [
    source 38
    target 34
  ]
  edge
  [
    source 38
    target 10
  ]
  edge
  [
    source 38
    target 35
  ]
  edge
  [
    source 38
    target 12
  ]
  edge
  [
    source 38
    target 37
  ]
  edge
  [
    source 38
    target 33
  ]
  edge
  [
    source 39
    target 34
  ]
  edge
  [
    source 39
    target 10
  ]
  edge
  [
    source 39
    target 35
  ]
  edge
  [
    source 39
    target 12
  ]
  edge
  [
    source 39
    target 38
  ]
  edge
  [
    source 39
    target 33
  ]
  edge
  [
    source 40
    target 8
  ]
  edge
  [
    source 40
    target 35
  ]
  edge
  [
    source 40
    target 12
  ]
  edge
  [
    source 40
    target 13
  ]
  edge
  [
    source 40
    target 20
  ]
  edge
  [
    source 40
    target 22
  ]
  edge
  [
    source 40
    target 39
  ]
  edge
  [
    source 40
    target 24
  ]
  edge
  [
    source 40
    target 25
  ]
  edge
  [
    source 40
    target 26
  ]
  edge
  [
    source 40
    target 27
  ]
  edge
  [
    source 41
    target 8
  ]
  edge
  [
    source 41
    target 9
  ]
  edge
  [
    source 41
    target 40
  ]
  edge
  [
    source 41
    target 12
  ]
  edge
  [
    source 41
    target 36
  ]
  edge
  [
    source 41
    target 27
  ]
  edge
  [
    source 42
    target 8
  ]
  edge
  [
    source 42
    target 40
  ]
  edge
  [
    source 42
    target 13
  ]
  edge
  [
    source 42
    target 39
  ]
  edge
  [
    source 43
    target 8
  ]
  edge
  [
    source 43
    target 35
  ]
  edge
  [
    source 43
    target 13
  ]
  edge
  [
    source 43
    target 42
  ]
  edge
  [
    source 44
    target 8
  ]
  edge
  [
    source 44
    target 40
  ]
  edge
  [
    source 44
    target 35
  ]
  edge
  [
    source 44
    target 12
  ]
  edge
  [
    source 44
    target 13
  ]
  edge
  [
    source 45
    target 8
  ]
  edge
  [
    source 45
    target 9
  ]
  edge
  [
    source 45
    target 40
  ]
  edge
  [
    source 45
    target 11
  ]
  edge
  [
    source 45
    target 26
  ]
  edge
  [
    source 46
    target 8
  ]
  edge
  [
    source 46
    target 12
  ]
  edge
  [
    source 47
    target 9
  ]
  edge
  [
    source 47
    target 40
  ]
  edge
  [
    source 47
    target 41
  ]
  edge
  [
    source 47
    target 11
  ]
  edge
  [
    source 47
    target 12
  ]
  edge
  [
    source 47
    target 13
  ]
  edge
  [
    source 47
    target 42
  ]
  edge
  [
    source 47
    target 36
  ]
  edge
  [
    source 47
    target 37
  ]
  edge
  [
    source 47
    target 17
  ]
  edge
  [
    source 47
    target 33
  ]
  edge
  [
    source 47
    target 45
  ]
  edge
  [
    source 47
    target 46
  ]
  edge
  [
    source 47
    target 23
  ]
  edge
  [
    source 47
    target 24
  ]
  edge
  [
    source 47
    target 26
  ]
  edge
  [
    source 47
    target 27
  ]
  edge
  [
    source 48
    target 9
  ]
  edge
  [
    source 48
    target 20
  ]
  edge
  [
    source 49
    target 9
  ]
  edge
  [
    source 49
    target 20
  ]
  edge
  [
    source 49
    target 31
  ]
  edge
  [
    source 49
    target 48
  ]
  edge
  [
    source 50
    target 9
  ]
  edge
  [
    source 50
    target 11
  ]
  edge
  [
    source 51
    target 9
  ]
  edge
  [
    source 52
    target 9
  ]
  edge
  [
    source 52
    target 22
  ]
  edge
  [
    source 52
    target 51
  ]
  edge
  [
    source 53
    target 40
  ]
  edge
  [
    source 53
    target 20
  ]
  edge
  [
    source 53
    target 24
  ]
  edge
  [
    source 53
    target 26
  ]
  edge
  [
    source 54
    target 40
  ]
  edge
  [
    source 54
    target 41
  ]
  edge
  [
    source 54
    target 12
  ]
  edge
  [
    source 54
    target 47
  ]
  edge
  [
    source 54
    target 23
  ]
  edge
  [
    source 54
    target 27
  ]
  edge
  [
    source 55
    target 10
  ]
  edge
  [
    source 55
    target 12
  ]
  edge
  [
    source 55
    target 15
  ]
  edge
  [
    source 55
    target 19
  ]
  edge
  [
    source 56
    target 11
  ]
  edge
  [
    source 56
    target 19
  ]
  edge
  [
    source 56
    target 43
  ]
  edge
  [
    source 57
    target 13
  ]
  edge
  [
    source 57
    target 56
  ]
  edge
  [
    source 57
    target 20
  ]
  edge
  [
    source 57
    target 48
  ]
  edge
  [
    source 57
    target 49
  ]
  edge
  [
    source 58
    target 14
  ]
  edge
  [
    source 58
    target 7
  ]
  edge
  [
    source 58
    target 30
  ]
  edge
  [
    source 58
    target 49
  ]
  edge
  [
    source 58
    target 50
  ]
  edge
  [
    source 58
    target 51
  ]
  edge
  [
    source 58
    target 52
  ]
  edge
  [
    source 60
    target 59
  ]
  edge
  [
    source 61
    target 59
  ]
  edge
  [
    source 62
    target 59
  ]
  edge
  [
    source 62
    target 60
  ]
  edge
  [
    source 63
    target 59
  ]
  edge
  [
    source 63
    target 60
  ]
  edge
  [
    source 63
    target 62
  ]
  edge
  [
    source 64
    target 58
  ]
  edge
  [
    source 64
    target 51
  ]
  edge
  [
    source 64
    target 52
  ]
  edge
  [
    source 65
    target 64
  ]
  edge
  [
    source 65
    target 58
  ]
  edge
  [
    source 65
    target 51
  ]
  edge
  [
    source 66
    target 64
  ]
  edge
  [
    source 66
    target 28
  ]
  edge
  [
    source 66
    target 30
  ]
  edge
  [
    source 67
    target 64
  ]
  edge
  [
    source 67
    target 65
  ]
  edge
  [
    source 67
    target 66
  ]
  edge
  [
    source 67
    target 30
  ]
  edge
  [
    source 68
    target 64
  ]
  edge
  [
    source 68
    target 58
  ]
  edge
  [
    source 68
    target 65
  ]
  edge
  [
    source 69
    target 64
  ]
  edge
  [
    source 69
    target 58
  ]
  edge
  [
    source 69
    target 65
  ]
  edge
  [
    source 69
    target 51
  ]
  edge
  [
    source 70
    target 64
  ]
  edge
  [
    source 70
    target 66
  ]
  edge
  [
    source 70
    target 30
  ]
  edge
  [
    source 71
    target 7
  ]
  edge
  [
    source 71
    target 68
  ]
  edge
  [
    source 71
    target 70
  ]
  edge
  [
    source 72
    target 71
  ]
  edge
  [
    source 72
    target 28
  ]
  edge
  [
    source 72
    target 66
  ]
  edge
  [
    source 72
    target 49
  ]
  edge
  [
    source 72
    target 70
  ]
  edge
  [
    source 73
    target 71
  ]
  edge
  [
    source 73
    target 72
  ]
  edge
  [
    source 73
    target 66
  ]
  edge
  [
    source 73
    target 30
  ]
  edge
  [
    source 73
    target 31
  ]
  edge
  [
    source 74
    target 71
  ]
  edge
  [
    source 74
    target 72
  ]
  edge
  [
    source 74
    target 73
  ]
  edge
  [
    source 74
    target 66
  ]
  edge
  [
    source 74
    target 30
  ]
  edge
  [
    source 74
    target 31
  ]
  edge
  [
    source 75
    target 71
  ]
  edge
  [
    source 75
    target 72
  ]
  edge
  [
    source 75
    target 73
  ]
  edge
  [
    source 75
    target 74
  ]
  edge
  [
    source 75
    target 30
  ]
  edge
  [
    source 75
    target 31
  ]
  edge
  [
    source 75
    target 70
  ]
  edge
  [
    source 76
    target 71
  ]
  edge
  [
    source 76
    target 72
  ]
  edge
  [
    source 76
    target 75
  ]
  edge
  [
    source 76
    target 66
  ]
  edge
  [
    source 76
    target 30
  ]
  edge
  [
    source 76
    target 31
  ]
  edge
  [
    source 76
    target 49
  ]
  edge
  [
    source 76
    target 53
  ]
  edge
  [
    source 77
    target 71
  ]
  edge
  [
    source 77
    target 75
  ]
  edge
  [
    source 77
    target 58
  ]
  edge
  [
    source 77
    target 30
  ]
  edge
  [
    source 77
    target 19
  ]
  edge
  [
    source 77
    target 31
  ]
  edge
  [
    source 77
    target 76
  ]
  edge
  [
    source 78
    target 71
  ]
  edge
  [
    source 78
    target 72
  ]
  edge
  [
    source 78
    target 74
  ]
  edge
  [
    source 78
    target 75
  ]
  edge
  [
    source 78
    target 31
  ]
  edge
  [
    source 79
    target 71
  ]
  edge
  [
    source 79
    target 72
  ]
  edge
  [
    source 79
    target 74
  ]
  edge
  [
    source 79
    target 75
  ]
  edge
  [
    source 79
    target 30
  ]
  edge
  [
    source 80
    target 71
  ]
  edge
  [
    source 80
    target 72
  ]
  edge
  [
    source 80
    target 66
  ]
  edge
  [
    source 80
    target 30
  ]
  edge
  [
    source 81
    target 71
  ]
  edge
  [
    source 82
    target 71
  ]
  edge
  [
    source 82
    target 72
  ]
  edge
  [
    source 82
    target 73
  ]
  edge
  [
    source 82
    target 74
  ]
  edge
  [
    source 82
    target 75
  ]
  edge
  [
    source 82
    target 30
  ]
  edge
  [
    source 82
    target 31
  ]
  edge
  [
    source 82
    target 76
  ]
  edge
  [
    source 83
    target 71
  ]
  edge
  [
    source 83
    target 73
  ]
  edge
  [
    source 83
    target 75
  ]
  edge
  [
    source 83
    target 30
  ]
  edge
  [
    source 83
    target 76
  ]
  edge
  [
    source 84
    target 72
  ]
  edge
  [
    source 84
    target 74
  ]
  edge
  [
    source 84
    target 75
  ]
  edge
  [
    source 84
    target 66
  ]
  edge
  [
    source 84
    target 73
  ]
  edge
  [
    source 84
    target 60
  ]
  edge
  [
    source 84
    target 30
  ]
  edge
  [
    source 84
    target 62
  ]
  edge
  [
    source 84
    target 76
  ]
  edge
  [
    source 84
    target 79
  ]
  edge
  [
    source 84
    target 81
  ]
  edge
  [
    source 84
    target 82
  ]
  edge
  [
    source 84
    target 83
  ]
  edge
  [
    source 85
    target 72
  ]
  edge
  [
    source 85
    target 7
  ]
  edge
  [
    source 85
    target 58
  ]
  edge
  [
    source 85
    target 65
  ]
  edge
  [
    source 85
    target 66
  ]
  edge
  [
    source 86
    target 72
  ]
  edge
  [
    source 86
    target 73
  ]
  edge
  [
    source 86
    target 66
  ]
  edge
  [
    source 86
    target 84
  ]
  edge
  [
    source 86
    target 60
  ]
  edge
  [
    source 86
    target 30
  ]
  edge
  [
    source 86
    target 61
  ]
  edge
  [
    source 86
    target 76
  ]
  edge
  [
    source 86
    target 81
  ]
  edge
  [
    source 87
    target 72
  ]
  edge
  [
    source 87
    target 74
  ]
  edge
  [
    source 87
    target 84
  ]
  edge
  [
    source 87
    target 83
  ]
  edge
  [
    source 88
    target 72
  ]
  edge
  [
    source 88
    target 74
  ]
  edge
  [
    source 88
    target 66
  ]
  edge
  [
    source 88
    target 84
  ]
  edge
  [
    source 89
    target 72
  ]
  edge
  [
    source 89
    target 73
  ]
  edge
  [
    source 89
    target 66
  ]
  edge
  [
    source 89
    target 84
  ]
  edge
  [
    source 89
    target 86
  ]
  edge
  [
    source 89
    target 88
  ]
  edge
  [
    source 90
    target 72
  ]
  edge
  [
    source 90
    target 66
  ]
  edge
  [
    source 90
    target 70
  ]
  edge
  [
    source 91
    target 72
  ]
  edge
  [
    source 91
    target 74
  ]
  edge
  [
    source 91
    target 75
  ]
  edge
  [
    source 91
    target 31
  ]
  edge
  [
    source 91
    target 90
  ]
  edge
  [
    source 91
    target 79
  ]
  edge
  [
    source 92
    target 72
  ]
  edge
  [
    source 92
    target 73
  ]
  edge
  [
    source 92
    target 75
  ]
  edge
  [
    source 93
    target 73
  ]
  edge
  [
    source 93
    target 66
  ]
  edge
  [
    source 93
    target 86
  ]
  edge
  [
    source 93
    target 30
  ]
  edge
  [
    source 94
    target 73
  ]
  edge
  [
    source 94
    target 84
  ]
  edge
  [
    source 94
    target 93
  ]
  edge
  [
    source 95
    target 73
  ]
  edge
  [
    source 95
    target 94
  ]
  edge
  [
    source 95
    target 61
  ]
  edge
  [
    source 96
    target 73
  ]
  edge
  [
    source 96
    target 66
  ]
  edge
  [
    source 96
    target 84
  ]
  edge
  [
    source 96
    target 94
  ]
  edge
  [
    source 97
    target 73
  ]
  edge
  [
    source 97
    target 66
  ]
  edge
  [
    source 97
    target 84
  ]
  edge
  [
    source 97
    target 86
  ]
  edge
  [
    source 97
    target 96
  ]
  edge
  [
    source 97
    target 81
  ]
  edge
  [
    source 98
    target 73
  ]
  edge
  [
    source 98
    target 74
  ]
  edge
  [
    source 98
    target 87
  ]
  edge
  [
    source 98
    target 91
  ]
  edge
  [
    source 99
    target 73
  ]
  edge
  [
    source 99
    target 74
  ]
  edge
  [
    source 99
    target 66
  ]
  edge
  [
    source 99
    target 84
  ]
  edge
  [
    source 99
    target 93
  ]
  edge
  [
    source 99
    target 60
  ]
  edge
  [
    source 99
    target 59
  ]
  edge
  [
    source 99
    target 30
  ]
  edge
  [
    source 99
    target 62
  ]
  edge
  [
    source 99
    target 63
  ]
  edge
  [
    source 99
    target 90
  ]
  edge
  [
    source 100
    target 73
  ]
  edge
  [
    source 100
    target 66
  ]
  edge
  [
    source 100
    target 84
  ]
  edge
  [
    source 100
    target 86
  ]
  edge
  [
    source 100
    target 96
  ]
  edge
  [
    source 100
    target 98
  ]
  edge
  [
    source 100
    target 99
  ]
  edge
  [
    source 100
    target 62
  ]
  edge
  [
    source 100
    target 79
  ]
  edge
  [
    source 100
    target 91
  ]
  edge
  [
    source 100
    target 83
  ]
  edge
  [
    source 101
    target 84
  ]
  edge
  [
    source 101
    target 86
  ]
  edge
  [
    source 101
    target 94
  ]
  edge
  [
    source 101
    target 100
  ]
  edge
  [
    source 101
    target 61
  ]
  edge
  [
    source 102
    target 93
  ]
  edge
  [
    source 102
    target 94
  ]
  edge
  [
    source 102
    target 95
  ]
  edge
  [
    source 102
    target 46
  ]
  edge
  [
    source 103
    target 67
  ]
  edge
  [
    source 104
    target 67
  ]
  edge
  [
    source 104
    target 69
  ]
  edge
  [
    source 104
    target 103
  ]
]
"""
POLBLOGS_JSON = r"""{"labels":["home.earthlink.net/~kevin.omeara","thecommonvirtue.com","oliverwillis.com","instapundit.com","conservativecat.com","wizbangblog.com","coxandforkum.com","imao.us","scrappleface.com","legendgames.net/myblog.asp","gcruse.typepad.com","drudgereport.com","gevkaffeegal.typepad.com/the_alliance","nickiegoomba.blogspot.com","jewishworldreview.com","realclearpolitics.com","truthlaidbear.com","homespunbloggers.blogspot.com","conservativeblogger.com","blogsforbush.com","gopbloggers.org","evilpundit.com","timblair.net","diplomadic.blogspot.com","chrenkoff.blogspot.com","dissectleft.blogspot.com","rightwingnews.com","asmallvictory.net","littlegreenfootballs.com/weblog","belmontclub.blogspot.com","windsofchange.net","etalkinghead.com","punditz.journalspace.com","my2cent.blogspot.com","anncoulter.org","blindmindseye.com","thepatriette.com","bobhayes.net","turnpikefeminist.com","livejournal.com/users/jmhm","pinkofeministhellcat.typepad.com/pinko_feminist_hellcat","bodyandsoul.typepad.com","economicswithaface.com/weblog","janegalt.net","j-bradford-delong.net/movable_type","yglesias.typepad.com/matthew","danieldrezner.com/blog","cmillertime.blogspot.com","neonbrainiac.nu/mt/blog","professorbainbridge.com","qando.net","southernappeal.blogspot.com","buzzmachine.com","pejmanesque.com","captainsquartersblog.com/mt","althouse.blogspot.com","powerlineblog.com","juancole.com","volokh.com","redstate.org","samizdata.net/blog","andrewsullivan.com","marksteyn.com","oxblog.blogspot.com","denbeste.nu","nationalreview.com/thecorner","dynamist.com/weblog","vodkapundit.com","dailypundit.com","hughhewitt.com","kausfiles.com","sgtstryker.com","michaeltotten.com","nicedoggie.net","townhall.com/clog","billhobbs.com","thatliberalmedia.com","patterico.com","therightcoast.blogspot.com","thespoonsexperience.com","balloon-juice.com","deanesmay.com","greatestjeneration.com","patrickruffini.com","washingtonmonthly.com","crookedtimber.org","outsidethebeltway.com","tagorda.com","vikingpundit.blogspot.com","cdharris.net","horologium.net","talkingpointsmemo.com","gaypatriot.blogspot.com","polipundit.com","rogerlsimon.com","stolenthunder.blogspot.com","ibe.blogspot.com","politicalvicesquad.blogspot.com","rightrainbow.com","conservativeeyes.blogspot.com","michellemalkin.com","billroggio.com/easongate","democraticunderground.com","brainshavings.com","aintnobaddude.com","newleftblogs.blogspot.com","rogerailes.blogspot.com","atrios.blogspot.com","busybusybusy.com","corrente.blogspot.com","dailykos.com","the-hamster.com","digbysblog.blogspot.com","leanleft.com","maxspeak.org/mt","offthekuff.com/mt","pandagon.net","pacificviews.org","celluloid-wisdom.com/pw","rittenhouse.blogspot.com","ruminatethis.com","suburbanguerrilla.blogspot.com","thetalkingdog.com","talkleft.com","prospect.org/weblog","tbogg.blogspot.com","jimtreacher.com","wampum.wabanaki.net","billmon.org","jameswolcott.com","reachm.com/amstreet","dailyhowler.com","discourse.net","nielsenhayden.com/electrolite","patriotboy.blogspot.com","madkane.com/notable.html","markarkleiman.com","mydd.com","nathannewman.org/log","dneiwert.blogspot.com","thepoorman.net","seetheforest.blogspot.com","thismodernworld.com","politicaltracker.com","freerepublic.com","iraqthemodel.blogspot.com","nerepublican.blogspot.com","christian-patriot.blogspot.com","chiefsblog.blogspot.com","achicknamedmarzi.com","moorewatch.com","conservativepunk.com","right-thinking.com","americablog.org","archpundit.com","burntorangereport.com","blogforamerica.com","bopnews.com","bullmooseblog.com","davidsirota.com","blog.dccc.org","democrats.org/blog","emergingdemocraticmajorityweblog.com/donkeyrising","fromtheroots.org","electablog.com","theleftcoaster.com","mathewgross.com/blog","liberaloasis.com","node707.com","myleftbrain.com","ndnblog.org","notgeniuses.com","politicalwire.com","polizeros.com","xnerg.blogspot.com","stevegilliard.blogspot.com","swingstateproject.com","kydem.blogspot.com","americablog.blogspot.com","motgs.com","lucianne.com","allahpundit.com","alphapatriot.com","donaldsensing.com","dummocrats.com","healingiraq.blogspot.com","jihadwatch.org","lashawnbarber.com","lt-smash.us","martinipundit.com","poliblogger.com","spreadingunderstanding.blogspot.com","mementomoron.blogspot.com","evangelicaloutpost.com","valuerightnews.blogspot.com","ogresview.blogspot.com","prolifeblogs.com","diggersrealm.com/mt","opiniontimes.com","zeke01.typepad.com","deserttusk.blogspot.com","daddypundit.blogspot.com","kitchennetwork.blogspot.com","awideawake.blogspot.com","thejessefactor.blogspot.com","squiggler.braveblog.com","aldaynet.org","hectorvex.blogspot.com","sixmeatbuffet.com","ace.mu.nu","gopandthecity.blogspot.com","beautifulatrocities.com","electricvenom.com","crankyneocon.com/crankyneocon","mypetjawa.mu.nu","acertainslantoflight.blogspot.com","writingcompany.blogs.com/this_isnt_writing_its_typ","beldar.blogs.com/beldarblog","betsyspage.blogspot.com","victorhanson.com","indcjournal.com","jquinton.com","mudvillegazette.com","theamericanmind.com","fraterslibertas.com","wonkette.com","americanthinker.com","thetarpit.blogspot.com","americanreport.blogspot.com","nykola.com","americandigest.org","slantpoint.com","smirkingchimp.com","mahablog.com","thewashingtonnote.com","markschmitt.typepad.com/decembrist","newdonkey.com","counterspin.blogspot.com","dir.salon.com/topics/joe_conason/index.html","warandpiece.com","nationalreview.com/frum/frum-diary.asp","americanpolitics.com","scoobiedavis.blogspot.com","takebackthemedia.com","tristero.blogspot.com","angrybear.blogspot.com","bartcop.com","amptoons.com/blog","roadtosurfdom.com","uggabugga.blogspot.com","ospolitics.org","liquidlist.com","interestingtimes.blogspot.com","elemming2.blogspot.com","southknoxbubba.net/skblog","politicalstrategy.org","elayneriggs.blogspot.com","agonist.org","msnbc.msn.com/id/3449870","rising-hegemon.blogspot.com","mycatharsis.com","patrickmcclarty.blogspot.com","jrogg.blogspot.com","drumwaster.com","mdcbowen.org/cobb","mhking.mu.nu","baldilocks.typepad.com","riteturnonly.blogspot.com","punditguy.com","incite1.blogspot.com","alarmingnews.com","hillcountryviews.blogspot.com","catscape.com","conservativedialysis.com","interested-participant.blogspot.com","hal9000report.blogdrive.com","raedinthemiddle.blogspot.com","riverbendblog.blogspot.com","shakingspears.blogspot.com","mathewgross.com","changeforamerica.com/blog","diypolitics.blogdrive.com","gadflyer.com","islanddave.blogspot.com","scripting.com","devildawg.blogspot.com","indepundit.com","messopotamian.blogspot.com","thecarpetbaggerreport.com","demagogue.blogspot.com","emergingdemocraticmajority.com","upper-left.blogspot.com","thetalentshow.org","blog.ncsweb.us","blogsagainsthillary.com","adviceforlefty.blogspot.com","bluelemur.com","theurbangrind.blogspot.com","gisleson.com/blog","12thharmonic.com/wordpress","hotflashesfromthecampaigntrail.blogspot.com","iddybud.blogspot.com","simbaud.blogspot.com","journals.aol.com/breuvers/madashellnotgoingtotakeitanymore","needlenose.com","rooksrant.com","roxanne.typepad.com","theyetimn.blogspot.com","margaretcho.com/blog/blog.htm","mirocat.com","crooksandliars.com","juliusblog.blogspot.com","risinghegemon.blogspot.com","righthandthief.blogspot.com","sadlyno.com","s88172659.onlinehome.us/spockosbrain.html","presidentboxer.blogspot.com","alvintostig.typepad.com","stephenbates.com/yellowdoggereldemocrat","liberalavenger.com","tomburka.com","higherpieproductions.com","anoldsoul.blogspot.com","michaelberube.com","fafblog.blogspot.com","dolphinsdock.com/blog","blogitics.com","dailyrecycler.com/blog","annatopia.com/home.html","amleft.blogspot.com","andymatic.com","thomasblog.typepad.com/bloggg","boffo.typepad.com","stagefour.typepad.com/commonprejudice","thefulcrum.blogspot.com","norbizness.com","loadedmouth.com","ignatz.blogspot.com","progressiveink.com","aboutpolitics.blogspot.com","madtechspeaks.blogspot.com","robot-invasion.com","scottcsmith.net","home.houston.rr.com/skeptical","whyareweback.blogspot.com","maruthecrankpot.blogspot.com","alicublog.blogspot.com","allspinzone.blogspot.com","one38.org","brilliantatbreakfast.blogspot.com","jadbury.com/blog","hereswhatsleft.typepad.com","mstabile.blogspot.com","loud-liberal.blogspot.com","nielsenhayden.com/makinglight","nomoremister.blogspot.com","obsidianwings.blogs.com","red-harvest.blogspot.com","robschumacher.blogspot.com","rudepundit.blogspot.com","scrutinyhooligans.blogspot.com","whoviating.blogspot.com","blogs.salon.com/0002874","jimgilliam.com","the-goddess.org","dohiyimir.typepad.com","kicktheleftist.blogspot.com","permanentred.com","home.earthlink.net/~fsrhine","blog.johnkerry.com","trainorphans.org","folkbum.blogspot.com","markadams.blogdrive.com","ihatepatrobertson.com","netpolitik.blogspot.com","rowhouselogic.com","thatgoodnight.com","acepilots.com/mt","thewritewing.blogspot.com","althippo.blogspot.com","windingroad.typepad.com","limericksavant.blogspot.com","radgeek.com","left2right.typepad.com","blueoregon.com","slowplay.com","clarified.blogspot.com","disintegrator.blogspot.com","jlynnsmith30.blogspot.com","chainik_hocker.blogspot.com","downtownlad.blogspot.com","thomasgalvin.blogspot.com","therightwingconspiracy.org","clubforgrowth.org/blog","unconventionalwisdom.typepad.com","annika.mu.nu","bushblog.us","maybeithinktoomuch.com","ofthemind.blogspot.com","rightmoment.blogspot.com","shotofpolitics.blogspot.com","jeffblanco.blogdrive.com","chepooka.com","ivotedforgeorge.com","jchristopherfeng.com","californiaconservatives4truth.blogspot.com","twentyfirstcenturyrepublican.blogspot.com","claycalhoun.com","mudandphud.blogspot.com","thepaytons.org/essays/considerettes","dgci.net","conservativepush.blogspot.com","madanthony.org","blubberhollow.com/blog","nonboxthinking.blogspot.com","ccs178.com","theliberalslies.blogspot.com","americanwarmonger.blogspot.com","virginiapolitics.blogspot.com","gregpalast.com","hategun.com","toteota.blogspot.com","goose3five.blogspot.com","politicalanimals.blogspot.com","farleft.blogspot.com","boloboffin.blogspot.com","dailywarnews.blogspot.com","timblair.spleenville.com","atease.blogspot.com","eastbaykerry.com","massachusetts-liberal.com","openeyesmemo.com","truthliesandcommonsense.blogspot.com","electionprojection.com","ajbenjaminjr.blogspot.com","blogspirator.blogspot.com","dalythoughts.com","shadydowns.blogspot.com","dogwooddreams.com/serendipity","georgewbush.com","gopinsight.com","tsrightdominion.blogspot.com","onthewire.blogspot.com","leftthought.blogspot.com","deathpenaltyusa.blogspot.com","lefti.blogspot.com","polstate.com","pudentilla.blogspot.com","blogs.salon.com/0003364","religiousleft.blogspot.com","tallone.org","lendmesomesugar.blogs.com/shut_up_already","sondrak.com","la4israel.org/wordpress","ghostofaflea.com","angrygwn.blogspot.com","therogueangel.com/blog","grupo-utopia.com/blog/isou","misschin.bloggage.com","bluestateredstate.blogspot.com","kcdems.blogspot.com","poeticvalues.blogspot.com","stcfoundation.blogspot.com","shrillblog.blogspot.com","jadbury.com","thecore4.com","barkingdingo.blogspot.com","slublog.blogspot.com","pardonmyenglish.com","mtvirtus.blogspot.com","strategicintelligence.blogspot.com","mydamascusroad.blogspot.com","thinkingright.us","viewfromaheight.blogspot.com","blogsofwar.com","cayankee.blogs.com","inthebullpen.com","realordeal.blogspot.com","amigo13.blogspot.com","homicidalmaniak.com","all-encompassingly.com","winceandnod.blogspot.com","lawguy.blogspot.com","ryanshead.com","freedomofthought.com","williamwebbdotorg.blogspot.com","jg-spot.blogspot.com","blog.willrobinson47.com","papadoc.net/pinkflamingobar.html","elderbearden.blogspot.com","strangesemantics.blogspot.com","everythingiknowiswrong.com","home.midsouth.rr.com/joefish","sneakeasysjoint.com","kalblog.com","patriotsforbush.com","patriotblog.com","blogodob.blogspot.com","laughingwolf.net","variousorthodoxies.com","soundingthetrumpet.blogspot.com","texasbug.blogspot.com","philomathean.blogs.com","theunderscore.blogspot.com","pajamaeditors.blogspot.com","americandinosaur.blogspot.com","unfetteredmusings.blogspot.com","coffee.ondragonswing.com","gullyborg.typepad.com","thisliberal.com","roseoftheknights.com/yank/pride.html","respublica2004.blogspot.com","blithered.blogspot.com","pennsylvanianinexile.blogspot.com","templeofjennifer.com/blog","ayerdis.com/rants.php","jperspective.com","whatsakyer.blogspot.com","newcounterculture.com/log","discerningtexan.blogspot.com","stix1972.blogspot.com","jacklewis.net","rightwingthinking.blogspot.com","kellipundit.blogspot.com","aceoftrumpblog.blogspot.com","rightvoices.com","zebrax.blogs.com","commonsenserunswild.typepad.com","made4theinternet.blogspot.com","redstatealliance.blogspot.com","confederateyankee.blogspot.com","poweragerambleson.blogspot.com","cynicalnation.com","stmonk.blogspot.com","romeocat.typepad.com","blogicus.com","decision08.blogspot.com","theneo-progressive.blogspot.com","conservativelife.com/blog","nationalbird.net","cbcbcbcb.blogspot.com","ratherbiased.com/news","headland.blogspot.com","weblogs.bakshi.us/redskunk","bakshi.us/redskunk","keithy1028.bravejournal.com","blog01.kintera.com/dnccblog","greaterdemocracy.org","snunes.blogspot.com","teenageenthusiast.blogspot.com","joebrent.blogspot.com","politicalmonitor.us/blog","margieburns.com","readingpost.com","a100wwe.blogspot.com","noguru.tblog.com","pennywit.com/drupal","sayanythingblog.com","thedonovan.com","millers_time.typepad.com","qoae.net","quibbles-n-bits.com","danzfamily.com","stillmanvalley.org/aoo","sinfactor.blogspot.com","okieonthelam.com","ilovecynics.com","demleft.blogspot.com","billrice.org","americanfuture.typepad.com","hasidicgentile.org","wehavesomeplanes.blogspot.com","ejectejecteject.com","kimdutoit.com/dr/weblog.php","randomjottings.net","alittlemoretotheright.com","activistchat.com/blogiran","dartblog.com","750volts.blogspot.com","vernsblog.thegillfamily.us:8180","fringeblog.com","theglitteringeye.com","politics.feedster.com","politicalmusings.net","rightonred.net","slingsnarrows.erudite-absurdity.com","ironmouth.com","rosenblog.com","presidentreagan.info/discussion","radioequalizer.blogspot.com","angrychineseblogger.blog-city.com","planetdave.com/blogs.php3?mid=1","truebluegal.blogspot.com","itlookslikethis.blogeasy.com","lennonreport.blogspot.com","thepeach.blogspot.com","secureliberty.org/","dmobley.blogspot.com","reckersau.blogspot.com","bittersweetme.net/blog/index.php","daisycutter.blogspot.com","shininglight.us","andrightlyso.blogspot.com","civilcommotion.com","powerpundit.com","mwilliams.info","babytrollblog.com","achingandspinning.blogspot.com","woneffe.typepad.com","henerd.com","radio.weblogs.com/0131587/categories/flush","karmicsoup.blogspot.com","hotflashreport.com","usliberals.about.com","frederickclarkson.com","robertp17.blogspot.com","jmbzine.com","oipp.blogspot.com","digestiblenews.blogspot.com","carpetblogger.com","obamablog.com","markwarnerforpresident2008.blogspot.com","manifestcontent.net","nomayo.mu.nu","rathergate.com","100monkeystyping.com","kensain.com","astroturf.uni.cc","littlegreenfootballs.com/weblog/weblog.php","liberaleric.blogspot.com","yoder.ru","holecard.blogspot.com","elevendayempire.com/movabletype","unodemocrats.com/blog","bernardmoon.blogspot.com","luxeword.com","retiredwarrant.blogdrive.com","centinel.blogspot.com","bensworld.patriotforum.org","bitheads.blogspot.com","prosandcons.us","shallows.blogspot.com","stillfighting.blogspot.com","donspoliticalblog.blogspot.com","viewpointjournal.com","kippsblog.com","insanepreschoolmom.blogspot.com","dystopia.blog-city.com","oddhours.blogspot.com","dems2004.org/blog","collegedems.com/blog","larsonreport.com","mediajunkie.com/edgewise","battlegroundstates.blogspot.com","schadelmann.com","txfx.net","king-of-fools.com","cubanstories.blogspot.com","rant.st","antijohnkerry.blogspot.com","conservativeresponse.com","thepoliticalteen.net","sarcasticsoutherner.blogspot.com","nationalcenter.org/blog.html","adeimantus.blogspot.com","hathaby.net/weblog.php","regimechangeiran.com","joatmoaf.typepad.com/i_love_jet_noise","itznewstome.blogspot.com","denisdekat.com","newappeal.blogspot.com","shoegal.modblog.com","peskytherat.com/pesky","alamonation.blogspot.com","rhetoricrhythm.blogspot.com","mobile.blogcastlive.com","andifeelfine.blogspot.com","tesibria.typepad.com/democracy_is_coming","orient-lodge.com","watdanook.blogspot.com","arkanssouri.blogspot.com","tom-hanna.org","bradcarson.com/blog","underscorebleach.net/content/jotsheet","truedemocrat.blogspot.com","eeg3.com","antiprotester.blogspot.com","thelucideye.blogspot.com","themeofabsence.com","blog.mintruth.com","charlesking.blogspot.com","ltobs.blogspot.com","ludovicspeaks.typepad.com/real_deal","ludovicspeaks.typepad.com","hereswhatsleft.typepad.com/home","dedspace.blogspot.com","claudialong.com/blog","whopundit.blogspot.com","celluloid-wisdom.com/pw/","littletinylies.com","dizzy-girl.net/index.php","datingrepublicans.com","martydee.com/poligov","rockpaperscissor.blogspot.com","southernwatch.blogspot.com","athena.blogs.com","mysterypollster.com/main","chiron.typepad.com","waveflux.net","brutallyhonest.org","techievampire.net/wppol","think2004.com","thetemplarpundit.blogspot.com","roughstock.blogspot.com","blog.kenlan.net","shortfatcorporal.blogspot.com","home.earthlink.net/~techservant","rightword.blogspot.com","lashawnbarber.blogspot.com","sweetgrrrl.blogspot.com","scha-den-freu-de.blogspot.com","iowageek.blogspot.com","marknicodemo.blogspot.com","conservativezone.blogspot.com","theconservativezone.com","kurtkk.blogspot.com","thesakeofargument.com","undercaffeinated.com","theradiomansays.blogspot.com","gotdesign.blogspot.com","conservativerevolution.com","streetlog.typepad.com","praguewriter.typepad.com","d-42.com","blog.dennisfox.net","thingsasitis.net","oldfashionedwhippersnapper.blogspot.com","jdeeth.blogspot.com","jokerstotheright.blogspot.com","patriotism101.blogspot.com","theworldaroundyou.com","atrios.blogspot.com/ ","stalinism.com/shot-by-both-sides","robbernard.com","john.hoke.org","bentcorner.blogspot.com","fearfulsymmetry.blogspot.com","meckz.net","iamaugustus.blogspot.com","discountblogger.com","lastdaysoftherepublic.fatoprofugus.net","nycitystoop.com","punditician.blogspot.com","julietterossant.com/superchefblog/superchefblog.html","boston.com/news/blogs/dnc","joshferguson.blogspot.com","tomherod.blogspot.com","idlehandsmag.com/yellowdog04","phillybri.blogspot.com","patriot-paradox.com","thevaticanofliberalism.com","pajamahadin.com","officesmileman.com","aldaron.modblog.com","quimundus.squarespace.com","talkshowamerica.blogspot.com","redmindbluestate.blogspot.com","quante.blogspot.com","deansjournal.blogspot.com","livejournal.com/users/hoptkov","jewssansfrontieres.blogspot.com","meateatingleftist.com","netweed.com/postmodernanarchist","arancaytar.blogspot.com","tuckereskew.typepad.com","smithersmpls.com","blog01.kintera.com/emilysblog","beastsbelly.blogspot.com","hugozoom.blogspot.com","retrogrouch.net","bidinotto.journalspace.com","laughatliberals.com","rightwingtroll.blogspot.com","damntheman.net","sharpeblog.cybercatholics.com","politicswithrichard.blogspot.com","nathancallahan.com","ergio.blogspot.com","allanjenkins.typepad.com","ladymac.com","nofearoffreedom.blogspot.com","emptyisform.com","spacedog26.blogspot.com","mariestwocents.blogspot.com","midwestrightwingers.blogspot.com","lorenjavier.blogspot.com","claytoncorner.blogspot.com","sistertoldjah.com","billsrants.blogspot.com","bushlies.net/pages/10/index.htm","pollsterseek.blogspot.com","buckeyepundit.blogspot.com","moxiegrrrl.com","bullmooserepublicans.com/blog","libertarianjackass.blogspot.com","matthewstinson.com/blog","lahargis.blogspot.com","xtremerightwing.net","amradio.blogspot.com","homepage.mac.com/elisa_camahort/iblog","sccdp.org/blog.php","philoponia.blogspot.com","borgardblog.blogspot.com","urbanecoinc.com/wordpress1","kenshepherd.blogspot.com","thislousytshirt.net","waitingfordorothy.blogspot.com","criticalviewer.com","anarchyxero.robfindlay.org","nashvilletruth.blogspot.com","montages.blogspot.com","portapulpit.com","tomcoburnisabigfatjerk.blogspot.com","techtrend.com/blog/taranicole","asmodeus-space.com/weblog/blog/index.php","proponentofreason.blogspot.com","blogfreeordie.blogspot.com","alternateworlds.blogspot.com","loveamericahatebush.com","stevecopy.blogharbor.com","swiftreport.blogs.com","hiddennook.blogspot.com","watchandwait.blogspot.com","rfburnhertz.net/rf_blog","c-pol.blogspot.com","opiniontimes.blogspot.com","thinkingrocks.blogspot.com","journals.aol.com/mark24609/myblog","mantic.squarespace.com","ackackack.com","fritzs-thoughts.blogspot.com","angryhomo.blogspot.com","johnmccrory.com","radikal_papi.pitas.com","realitybasednation.com","writehouse.us","cbaker.org/blog","bigfatroid.blogspot.com","forewarned.blogspot.com","musicfromthebluelight.com/b","politicaldogs.org","hipublican.com","dewar.journalspace.com","wpblog.ohpinion.com","vocis.com/frank","uberdude.blogspot.com","spectatorsport.blogspot.com","therightblog.blogspot.com","americanchronicle.blogspot.com","kiosan.blogharbor.com","shakespearessister.blogspot.com","viewfromtonka.blogspot.com","beefalwayswins.blogspot.com","stevemacek.blogspot.com","anonymoussources.blogspot.com","americankernel.com","howienfriends.blogspot.com","cecile.squarespace.com","jbcougar.blogspot.com","oppb.com","palominas.blogspot.com","thecrankfiles.blogspot.com","conservativenation.us","trueconservative.typepad.com","mayflowerhill.blogspot.com","debunker.net","pajamajournalist.blogspot.com","astrosoup.blogspot.com","blog.nam.org","gindy.blogspot.com","cemonitor.blogspot.com","bowles2004.com/weblog","watchingthewatchers.org","mixtersmix.blogspot.com","eltcamerica.blogspot.com","blog.veggiedude.com","o4tom.blogspot.com","emilythedog.blogspot.com","blackmanforbush.blogspot.com","piano-tuning.net/blog","bamapachyderm.blogspot.com","brianscott.blogspot.com","seeingtheforest.com","lefterer.com","resistoppression.blogspot.com","maggiespants.co.uk","itscomedowntothis.blogspot.com","leftinthereign.blogspot.com","bluestates.blogspot.com","conniptions.net","grassrootsmom.blogspot.com","greendogdemocrat.blogspot.com","markheimonen.blogspot.com","idiosyncratictendencies.com","jadedreality.blogspot.com","jinkythecat.blogspot.com","kerryforpresident2008.blogspot.com","johnnynobody.blogspot.com","lightupthedarkness.org","lowrentrat.blogdrive.com","maultasch.us","angrydesi.typepad.com","mrleft.org","newdemocrat.blogdrive.com","pacdaily.com","peacegarden.blogspot.com","shamelessantagonist.blogspot.com","gccpundit.blogspot.com","askhoudari.blogspot.com","thedailygoodnews.com","youngconservative.blogspot.com","dogfight04.typepad.com","worldonfire.typepad.com/world_on_fire","sebourn.com/weblog","turnspit.com","unusualhabits.blogspot.com","calpoly.edu/~doleary/blog.htm","magnificentfreedom.us","sharedthought.com/weblog","lawdork.blogspot.com","alexpelan.blogspot.com","phoblographer.com","reporterette.com","swamiuptown.com","reggiescorner.blogspot.com","question-dean.blogspot.com","drunkensamurai.blogspot.com","conservativeobserver.blogspot.com","therostra.blogspot.com","contrapositive.blogspot.com","parabasis.typepad.com","wholewheatblogger.com","toneofvoice.blogspot.com","ludis.blogspot.com","tepid-observer.blogspot.com","95theses.blogspot.com","newsbloggers.blogspot.com","myaisling.blogspot.com","mostlycajun.com","rightstate.blogspot.com","elissa.typepad.com","centerpiece.blogspot.com","bkennelly.com/vox","slipofthepen.blogspot.com","steveguy.me.uk","dailyblatt.blogspot.com","georgewbuy.blogspot.com","rightwingjournal.blogspot.com","tobyemden.blogspot.com","angryfinger.org","politicalthought.net","tseytlin.blogspot.com","mediaprima.com/nv1962","republicoft.typepad.com","gopandcollege.blogspot.com","tokenyoungconservative.blogspot.com","alamodispatches.blogspot.com","minagahet.blogspot.com","metropol47.blogspot.com","answerguy.blogspot.com","liberalangst.org","thebloghill.blogspot.com","noseyonline.com","johnnbrown.blogspot.com","godubya.com","powervacuum.org","x-pollen.com/many/blog.html","jeremybrendan.blogspot.com","totels.com/bush04","ron-james.com/blog","thunewatch.squarespace.com","cleancutkid.com","johnkerrymustlose.com","hellskitchennyc.blogspot.com","batr.net","batr.org/commentary.html","americanworldview.tripod.com/weltansblog","home.att.net/~gentle.reader","politicaltracker.com/blog/weblog.php","zeph1z.tripod.com/blog","iflipflop.com","travalex.blogspot.com","alphapredator.com/wdtgw.htm","revone.net/jarhead","metallicity.com/mb.asp","liberalreview.blogspot.com","thelastword.blogdrive.com","r2korn.blogdrive.com","profgoose.blogspot.com","thatcoloredfellasweblog.bloghorn.com","rightwingbob.com","conservativessuck.blogspot.com","yarsrevenge.blogspot.com","sethdmichaels.com","flagrancy.net","wtpm.blogspot.com","irregulartimes.com/blogger.html","philosophers-stone.blogspot.com","btcnews.com/btcnews","lefttimes.com","paulchasman.com","nearouterheaven.blogspot.com","wsrblood.typepad.com/wiserblog","abbadabbaduo.blogspot.com","nastyboys.blogspot.com","blogforarizona.com","donkey2004.blogspot.com","armchairgeneralist.typepad.com","leonards-digest.blogspot.com","dilusionl.blogspot.com","propagandamachine.blogspot.com","boottotheskull.com","blosteam.blogspot.com","stealthebandwagon.blogspot.com","grassrootsnation.com","boycottsbg.com","winthewest.blogspot.com","crankybastard.blogspot.com","theangryliberal.blogspot.com","strangedoctrines.typepad.com","perryvsworld.blogspot.com","bradfriedman.com/bradblog","dawnofnewamerica.blogspot.com","theblueview.blogspot.com","americansforbayh.blogspot.com","loosecoin.blogspot.com","rightwingbastardo.blogspot.com","matthewyglesias.com","bluegoldfish.blogs.com/surface","arkansastonight.com","virginiaprogressive.blogspot.com","dimmykarras.blogspot.com","purplestates.blogspot.com","quimundus.modblog.com","isdl.weblogs.us","isdl.blogspot.com","depressedlonelybored.typepad.com","swfwithattitude.blogspot.com","delraysteve.com/blog","blue6golf.blogspot.com","grownups.blogspot.com","heavylifting.blogspot.com","livinginbabylon.blogspot.com","section-31.blogspot.com","firstdownpolitics.com","churchofcriticalthinking.com","someonetakethewheel.blogspot.com","zeke01.blogspot.com","stitt.blogspot.com","thenudnikfile.blogspot.com","apologiesdemanded.blogspot.com","karlhenderson.blogspot.com","juntoboys.blogspot.com","usademocrazy.blogspot.com","battleground-wisconsin.blogspot.com","theantibush.org","tinylittledots.com","gmscorner.blogspot.com","thedispatch.blogspot.com","mattresspolice.com","etherealgirl.blogspot.com","fiercepoet.com","justmark.com","interestingtimes.typepad.com","progressivesociety.com/blog","simplyappalling.blogspot.com","neoconswatch.blogspot.com","bigjourney.blogspot.com","stan-istan.blogspot.com","dadougster.blogspot.com","liberalthunderdome.blogspot.com","thesnooter.blogspot.com","digitaljay.blogspot.com","nationalnewsregister.com/modules/wordpress","franklog.com","thomaswharry.com","deadcantrant.com/blog","therinsecycle.com","joepublicus.blogspot.com","happyhomeschooler.blog-city.com","sickofliberals.blogspot.com","raincrossblog.com","allthingsmark.com","poorjustin.blogspot.com","kevincourser.blogspot.com","mindspring.com/~gsb3/blogger.html","renpro.kwcom.net","selfish-souls.com/blog","rightwingdude.squarespace.com","wildweasel.blog-city.com","summation.net","web.utk.edu/~agroves","neoconservatives.blogspot.com","rightperspective.net","blog.so-amazing.net","hobsonschoice.us","newsnik.com","smarthomeschool.com","bellusbellum.info","warriorscreed.000k3.com","mickikrimmel.com/redcarpet","charlineandjamie.com/dotnetweb01a/blogdisplay.aspx?logname=jamie&#38;logcatid=48","adamtalib.blogspot.com","theagitator.blogspot.com","talesofawanderingmind.blogspot.com","iwantmycountryback.org","politicalpie.modblog.com","bloganonymous.com","theaggpro.blogspot.com","absurd-canadian.blogspot.com","philvbblog.blogspot.com","blog.glinka.com","rebrand-democrat.txfx.net","theapologist.blogspot.com","americanmuslim.blogs.com","njfordemocracy.org","translucence.org","pattymurray.com/blog.php","bluegrassroots.blogspot.com","thepoliticalanimal.com","theblogwarrior.blogspot.com","dashnier.blogspot.com","michaelphillips.blogspot.com","justalkin.us","cptnwinky.com","global-equality.org/news/blog/index.shtml","cybermenace.blogspot.com","awards5.tripod.com/tarasblog","blogslut.com","erikack.blogspot.com","fairshot.org","garbage-house.com","lightupthedarkness.org/blog/default.asp","newsjungle.blogspot.com","truthprobe.blogspot.com","nationalreview.com/thecorner/corner.asp","journals.aol.com/lily990016/pissoffliberals","ibrutus.com","southernrightwingextremist.blogspot.com","charleypatton.blogspot.com","mcwil.blogspot.com","amliberal.com/blog","winnegar.blog-city.com","democratreport.blogspot.com","meanspirit.blogspot.com","midguard.blogspot.com","conservativepolitics.blogspot.com","rockpaperscissors.blogs.com","smellyduck.blogspot.com","penheaded.blogspot.com","richmar.blogspot.com","xanga.com/lockjawtheogre","livejournal.com/users/bambenek","ikilledcheguevara.blogspot.com","the-torch.com","michael124.blogspot.com","leftcoastconservative.blogspot.com","neverwrongnever.blogspot.com","hellafaded.blogspot.com","fuckthisblog.blogspot.com","gerberatetra.blogspot.com","usdlaw.blogspot.com","chadster.capitalfocus.org","prestoncaldwellblog.blogspot.com","browncross.com/usualsuspects","younglibs.com","democratvoice.org","enemykombatant.blogspot.com","bushmisunderestimated.blogspot.com","notbush.com","bloghogger.blogspot.com","estropundit.blogspot.com","messengerpuppet.com/blog","dems2000.net","nicoladellarciprete.blogs.com","imprescindibile.ilcannocchiale.it","marcbrazeau.blogspot.com","janm.blogspot.com","nebursworld.blogspot.com","crankylittleblog.blogspot.com","orthogonian.blogspot.com","lonewacko.com","bettyblog.com","leftwingnation.com","mikesamerica.blogspot.com","myechochamber.blogspot.com","anywhichway.net","b-patton.blogspot.com","soonerthought.com","jetage.blogspot.com","myleftthoughts.blogspot.com","erickohn.com","nosmallplans.com/rants","nerofiddled.blogspot.com","saveoursenate.blogspot.com","spydrzweb.blogspot.com","anybodybutbushyall.blogspot.com","urbandemocracy.blogspot.com","smashingzero.blogspot.com","rednyc.blogspot.com","calirep.blogspot.com","endthenightmare.blogspot.com","bensgtusmc.com","republog.net","rabner.blogspot.com","restoreamerica.blogspot.com","duckdaotsu.blogspot.com","scrawlville.com","r21online.com","anothervrwc.blogspot.com","thirdredeye.blogspot.com","sedenteanimo.com"],"value":[0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,0,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,0,0,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,1,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,1,1,1,1,1,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,1,1,1,0,0,0,0,1,0,1,1,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,1,0,1],"edges":[[0,1],[0,2],[0,3],[0,339],[1,789],[1,797],[1,19],[1,57],[1,65],[1,110],[1,107],[1,91],[1,11],[1,188],[1,558],[1,75],[1,222],[1,54],[1,6],[1,183],[1,20],[1,3],[1,145],[1,100],[1,53],[1,377],[1,56],[1,59],[1,67],[1,576],[1,710],[1,474],[2,38],[2,104],[2,137],[2,112],[2,166],[2,298],[2,259],[2,135],[2,365],[2,242],[2,248],[2,337],[2,41],[2,117],[2,467],[2,122],[2,272],[2,361],[2,93],[2,593],[2,43],[2,619],[2,299],[2,176],[2,72],[2,639],[2,650],[2,708],[2,457],[2,711],[2,223],[2,163],[2,741],[2,5],[2,333],[2,584],[2,775],[2,795],[2,50],[2,379],[2,52],[2,67],[2,779],[2,445],[2,447],[2,358],[2,548],[2,84],[2,902],[2,165],[2,302],[2,254],[2,317],[2,109],[2,161],[2,128],[2,105],[2,140],[2,816],[2,323],[2,108],[2,256],[2,335],[2,322],[2,1038],[2,355],[2,133],[2,292],[2,252],[2,160],[2,674],[2,244],[2,258],[2,341],[2,3],[2,594],[2,625],[2,990],[2,255],[2,312],[2,1002],[2,462],[2,422],[2,291],[2,311],[2,321],[2,57],[2,156],[2,220],[2,130],[2,172],[2,107],[2,557],[2,842],[2,316],[2,349],[2,116],[2,167],[2,80],[2,758],[2,1007],[2,391],[2,972],[2,1170],[2,532],[2,344],[2,136],[2,1186],[2,139],[2,1191],[2,991],[2,549],[2,911],[2,1074],[2,1203],[2,310],[2,374],[2,174],[2,125],[2,424],[2,380],[3,21],[3,53],[3,92],[3,104],[3,143],[3,179],[3,207],[3,215],[3,124],[3,112],[3,261],[3,285],[3,135],[3,270],[3,426],[3,405],[3,16],[3,467],[3,469],[3,346],[3,122],[3,184],[3,12],[3,518],[3,531],[3,568],[3,571],[3,29],[3,6],[3,7],[3,93],[3,58],[3,145],[3,183],[3,580],[3,30],[3,582],[3,587],[3,46],[3,118],[3,60],[3,606],[3,43],[3,622],[3,73],[3,71],[3,72],[3,77],[3,51],[3,101],[3,66],[3,427],[3,639],[3,100],[3,220],[3,662],[3,491],[3,209],[3,22],[3,222],[3,680],[3,224],[3,576],[3,221],[3,572],[3,573],[3,88],[3,327],[3,698],[3,24],[3,624],[3,706],[3,394],[3,707],[3,214],[3,581],[3,213],[3,223],[3,596],[3,472],[3,103],[3,723],[3,504],[3,623],[3,82],[3,661],[3,743],[3,269],[3,5],[3,470],[3,508],[3,69],[3,392],[3,481],[3,775],[3,777],[3,473],[3,25],[3,519],[3,171],[3,659],[3,506],[3,666],[3,797],[3,561],[3,494],[3,50],[3,271],[3,812],[3,379],[3,52],[3,83],[3,640],[3,760],[3,67],[3,540],[3,56],[3,535],[3,632],[3,528],[3,583],[3,407],[3,721],[3,874],[3,876],[3,65],[3,510],[3,484],[3,418],[3,668],[3,917],[3,84],[3,761],[3,475],[3,932],[3,871],[3,488],[3,947],[3,201],[3,949],[3,605],[3,705],[3,600],[3,686],[3,87],[3,425],[3,965],[3,254],[3,433],[3,498],[3,90],[3,500],[3,409],[3,13],[3,988],[3,94],[3,994],[3,85],[3,140],[3,204],[3,216],[3,468],[3,362],[3,497],[3,1038],[3,225],[3,569],[3,63],[3,515],[3,217],[3,523],[3,1029],[3,408],[3,436],[3,258],[3,635],[3,575],[3,759],[3,712],[3,1053],[3,54],[3,230],[3,187],[3,49],[3,59],[3,55],[3,131],[3,11],[3,240],[3,14],[3,70],[3,91],[3,172],[3,62],[3,61],[3,76],[3,96],[3,27],[3,266],[3,474],[3,89],[3,286],[3,80],[3,44],[3,64],[3,133],[3,81],[3,75],[3,28],[3,212],[3,136],[3,287],[3,68],[3,15],[3,26],[3,174],[3,79],[3,123],[3,126],[3,284],[3,45],[3,185],[3,604],[3,492],[3,203],[3,663],[3,228],[3,36],[3,857],[3,1062],[3,648],[3,502],[3,1067],[3,834],[3,1027],[3,597],[3,1069],[3,479],[3,146],[3,74],[3,189],[3,1114],[3,1115],[3,520],[3,831],[3,557],[3,530],[3,1060],[3,1040],[3,717],[3,336],[3,538],[3,598],[3,1129],[3,356],[3,710],[3,1086],[3,1004],[3,114],[3,562],[3,729],[3,1144],[3,631],[3,278],[3,385],[3,527],[3,398],[3,391],[3,78],[3,720],[3,1168],[3,193],[3,532],[3,703],[3,9],[3,344],[3,1178],[3,1166],[3,482],[3,1186],[3,1171],[3,206],[3,86],[3,804],[3,1191],[3,480],[3,1195],[3,525],[3,1201],[3,1205],[3,536],[3,862],[3,182],[3,1209],[3,704],[3,738],[3,1059],[3,577],[3,1214],[3,589],[3,724],[3,509],[3,273],[4,5],[4,6],[4,7],[4,8],[4,9],[4,10],[4,11],[4,12],[4,13],[4,14],[4,15],[4,16],[4,17],[4,268],[4,535],[4,600],[4,440],[4,534],[5,21],[5,92],[5,104],[5,179],[5,191],[5,207],[5,215],[5,270],[5,432],[5,455],[5,467],[5,184],[5,12],[5,395],[5,145],[5,582],[5,377],[5,118],[5,60],[5,622],[5,628],[5,77],[5,66],[5,100],[5,220],[5,657],[5,660],[5,212],[5,209],[5,221],[5,685],[5,88],[5,24],[5,655],[5,624],[5,214],[5,457],[5,213],[5,223],[5,596],[5,274],[5,623],[5,730],[5,269],[5,681],[5,94],[5,86],[5,187],[5,557],[5,58],[5,74],[5,197],[5,604],[5,52],[5,605],[5,29],[5,80],[5,27],[5,190],[5,656],[5,54],[5,84],[5,53],[5,26],[5,67],[5,56],[5,746],[5,747],[5,748],[5,266],[5,599],[5,418],[5,733],[5,81],[5,749],[5,31],[5,580],[5,454],[5,7],[5,89],[5,744],[5,50],[5,558],[5,493],[5,193],[5,82],[5,79],[5,740],[5,481],[5,345],[5,453],[5,524],[5,181],[5,663],[5,188],[5,476],[5,28],[5,83],[5,49],[5,385],[5,391],[5,470],[5,69],[5,392],[5,765],[5,519],[5,789],[5,791],[5,506],[5,797],[5,812],[5,528],[5,583],[5,721],[5,767],[5,542],[5,668],[5,761],[5,475],[5,932],[5,488],[5,600],[5,686],[5,965],[5,286],[5,433],[5,498],[5,13],[5,265],[5,216],[5,468],[5,313],[5,768],[5,515],[5,217],[5,523],[5,408],[5,635],[5,575],[5,857],[5,146],[5,422],[5,189],[5,565],[5,397],[5,739],[5,538],[5,598],[5,59],[5,710],[5,1020],[5,1004],[5,562],[5,1158],[5,1162],[5,527],[5,398],[5,972],[5,720],[5,367],[5,344],[5,482],[5,556],[5,1171],[5,206],[5,241],[5,227],[5,1210],[5,704],[5,417],[5,1213],[5,577],[5,1214],[5,32],[5,879],[5,273],[6,42],[6,179],[6,68],[6,285],[6,12],[6,272],[6,219],[6,572],[6,14],[6,28],[6,575],[6,27],[6,453],[6,182],[6,61],[6,73],[6,29],[6,576],[6,474],[6,188],[6,81],[6,186],[6,75],[6,220],[6,145],[6,100],[6,183],[6,86],[6,56],[6,26],[6,94],[6,427],[6,30],[6,8],[6,181],[6,16],[6,563],[6,606],[6,634],[6,209],[6,655],[6,624],[6,98],[6,476],[6,723],[6,82],[6,661],[6,521],[6,269],[6,495],[6,392],[6,535],[6,407],[6,874],[6,668],[6,923],[6,599],[6,871],[6,605],[6,952],[6,87],[6,441],[6,210],[6,498],[6,13],[6,996],[6,204],[6,468],[6,497],[6,523],[6,604],[6,889],[6,203],[6,36],[6,1067],[6,146],[6,189],[6,393],[6,511],[6,1116],[6,557],[6,1060],[6,397],[6,739],[6,1129],[6,1086],[6,385],[6,527],[6,391],[6,720],[6,1170],[6,703],[6,206],[6,480],[6,794],[6,1209],[6,1059],[6,54],[6,1221],[6,273],[6,722],[7,179],[7,191],[7,135],[7,432],[7,469],[7,122],[7,12],[7,518],[7,73],[7,496],[7,481],[7,19],[7,28],[7,8],[7,22],[7,26],[7,65],[7,93],[7,58],[7,606],[7,220],[7,209],[7,222],[7,572],[7,573],[7,730],[7,269],[7,470],[7,508],[7,495],[7,392],[7,25],[7,519],[7,67],[7,507],[7,632],[7,869],[7,505],[7,486],[7,668],[7,917],[7,513],[7,526],[7,871],[7,605],[7,600],[7,512],[7,498],[7,994],[7,216],[7,468],[7,497],[7,1044],[7,515],[7,217],[7,575],[7,604],[7,203],[7,36],[7,857],[7,1067],[7,514],[7,74],[7,187],[7,1079],[7,189],[7,739],[7,538],[7,1129],[7,1045],[7,710],[7,631],[7,527],[7,391],[7,9],[7,482],[7,206],[7,86],[7,480],[7,474],[7,704],[7,273],[8,53],[8,92],[8,179],[8,191],[8,215],[8,135],[8,432],[8,469],[8,122],[8,477],[8,518],[8,144],[8,93],[8,58],[8,30],[8,587],[8,60],[8,606],[8,43],[8,73],[8,77],[8,51],[8,662],[8,209],[8,222],[8,573],[8,655],[8,394],[8,707],[8,213],[8,476],[8,596],[8,736],[8,743],[8,521],[8,470],[8,508],[8,69],[8,392],[8,765],[8,519],[8,506],[8,190],[8,83],[8,640],[8,67],[8,56],[8,673],[8,840],[8,632],[8,528],[8,583],[8,656],[8,721],[8,505],[8,542],[8,668],[8,488],[8,201],[8,441],[8,433],[8,13],[8,588],[8,216],[8,256],[8,497],[8,63],[8,523],[8,1029],[8,36],[8,857],[8,1062],[8,834],[8,1069],[8,681],[8,479],[8,146],[8,187],[8,1079],[8,393],[8,1115],[8,397],[8,538],[8,1129],[8,356],[8,729],[8,80],[8,1144],[8,631],[8,385],[8,1162],[8,75],[8,391],[8,720],[8,1170],[8,193],[8,532],[8,482],[8,206],[8,86],[8,480],[8,1194],[8,474],[8,536],[8,862],[8,182],[8,174],[8,704],[8,1219],[8,509],[8,273],[9,12],[9,77],[9,267],[9,25],[9,201],[9,37],[9,19],[9,1020],[9,118],[9,709],[9,42],[9,11],[9,247],[9,55],[9,65],[9,80],[9,100],[9,16],[9,67],[9,174],[9,575],[9,453],[9,673],[9,182],[9,599],[9,475],[9,99],[9,409],[9,410],[9,580],[9,623],[9,214],[9,659],[9,517],[9,187],[9,412],[9,189],[9,192],[9,408],[9,33],[9,1106],[9,190],[9,377],[9,710],[9,49],[9,869],[9,511],[9,399],[9,782],[9,587],[9,208],[9,467],[9,79],[9,712],[9,592],[9,441],[9,406],[9,832],[9,294],[9,273],[10,261],[10,12],[10,680],[10,560],[10,274],[10,100],[10,509],[10,503],[10,16],[10,731],[10,475],[10,932],[10,19],[11,33],[11,53],[11,92],[11,143],[11,191],[11,207],[11,215],[11,261],[11,268],[11,285],[11,293],[11,297],[11,135],[11,270],[11,405],[11,442],[11,455],[11,467],[11,469],[11,122],[11,184],[11,14],[11,180],[11,70],[11,61],[11,129],[11,477],[11,28],[11,554],[11,531],[11,571],[11,183],[11,34],[11,615],[11,628],[11,634],[11,636],[11,639],[11,657],[11,294],[11,680],[11,688],[11,24],[11,394],[11,211],[11,213],[11,223],[11,596],[11,208],[11,454],[11,723],[11,730],[11,731],[11,269],[11,470],[11,508],[11,584],[11,495],[11,392],[11,765],[11,529],[11,205],[11,775],[11,777],[11,519],[11,784],[11,789],[11,659],[11,793],[11,796],[11,797],[11,801],[11,494],[11,640],[11,366],[11,819],[11,67],[11,399],[11,673],[11,840],[11,478],[11,632],[11,528],[11,855],[11,858],[11,721],[11,874],[11,877],[11,505],[11,887],[11,533],[11,418],[11,897],[11,668],[11,917],[11,919],[11,924],[11,926],[11,927],[11,599],[11,932],[11,934],[11,526],[11,937],[11,871],[11,941],[11,488],[11,944],[11,605],[11,705],[11,600],[11,686],[11,961],[11,87],[11,441],[11,965],[11,913],[11,968],[11,254],[11,210],[11,975],[11,199],[11,317],[11,979],[11,13],[11,988],[11,94],[11,994],[11,996],[11,204],[11,555],[11,468],[11,1036],[11,497],[11,292],[11,202],[11,225],[11,768],[11,63],[11,515],[11,523],[11,408],[11,575],[11,712],[11,1053],[11,1054],[11,147],[11,604],[11,492],[11,847],[11,203],[11,36],[11,857],[11,200],[11,153],[11,594],[11,502],[11,276],[11,834],[11,1027],[11,1069],[11,146],[11,601],[11,187],[11,19],[11,189],[11,511],[11,231],[11,342],[11,1115],[11,1116],[11,520],[11,831],[11,1117],[11,1119],[11,530],[11,717],[11,155],[11,1130],[11,416],[11,1133],[11,1086],[11,729],[11,1144],[11,1146],[11,631],[11,1148],[11,1157],[11,1161],[11,278],[11,833],[11,385],[11,1162],[11,1052],[11,527],[11,398],[11,391],[11,720],[11,1168],[11,1169],[11,193],[11,532],[11,782],[11,1178],[11,406],[11,1166],[11,482],[11,1181],[11,136],[11,1184],[11,1171],[11,206],[11,1191],[11,480],[11,1088],[11,1201],[11,1094],[11,536],[11,862],[11,227],[11,1210],[11,704],[11,125],[11,738],[11,1059],[11,1212],[11,577],[11,641],[11,1214],[11,54],[11,724],[11,55],[11,1219],[11,1167],[11,1221],[11,509],[11,273],[11,1136],[11,722],[12,179],[12,191],[12,207],[12,432],[12,184],[12,477],[12,86],[12,478],[12,268],[12,479],[12,195],[12,73],[12,480],[12,395],[12,481],[12,482],[12,385],[12,147],[12,483],[12,269],[12,265],[12,484],[12,485],[12,486],[12,487],[12,103],[12,488],[12,489],[12,490],[12,35],[12,491],[12,492],[12,493],[12,494],[12,33],[12,495],[12,496],[12,410],[12,206],[12,31],[12,193],[12,497],[12,274],[12,468],[12,498],[12,499],[12,278],[12,500],[12,501],[12,502],[12,503],[12,267],[12,273],[12,504],[12,505],[12,200],[12,417],[12,441],[12,199],[12,506],[12,194],[12,507],[12,508],[12,389],[12,509],[12,510],[12,511],[12,512],[12,401],[12,99],[12,187],[12,203],[12,513],[12,514],[12,515],[12,516],[12,517],[12,518],[12,519],[12,520],[12,412],[12,521],[12,522],[12,523],[12,210],[12,524],[12,525],[12,526],[12,527],[12,528],[12,529],[12,208],[12,530],[12,531],[12,271],[12,532],[12,533],[12,534],[12,535],[12,536],[12,537],[12,538],[12,340],[12,27],[12,212],[12,793],[12,561],[12,418],[12,761],[12,926],[12,204],[12,1044],[12,739],[12,1045],[12,1162],[12,794],[13,17],[13,600],[13,512],[13,56],[13,145],[13,209],[13,417],[13,28],[13,15],[13,100],[13,65],[13,16],[13,262],[13,29],[13,22],[13,69],[13,118],[13,624],[13,340],[13,584],[13,886],[13,222],[13,20],[13,294],[13,942],[13,503],[14,207],[14,432],[14,28],[14,183],[14,587],[14,34],[14,628],[14,186],[14,624],[14,708],[14,596],[14,269],[14,266],[14,519],[14,67],[14,306],[14,528],[14,858],[14,927],[14,599],[14,941],[14,949],[14,498],[14,699],[14,228],[14,187],[14,462],[14,1060],[14,1129],[14,833],[14,720],[14,406],[14,57],[14,1088],[14,301],[14,227],[14,182],[14,310],[15,53],[15,143],[15,137],[15,215],[15,240],[15,69],[15,100],[15,65],[15,56],[15,432],[15,435],[15,467],[15,469],[15,184],[15,28],[15,554],[15,395],[15,272],[15,568],[15,571],[15,29],[15,93],[15,183],[15,587],[15,563],[15,46],[15,606],[15,43],[15,628],[15,77],[15,51],[15,101],[15,220],[15,662],[15,209],[15,88],[15,24],[15,706],[15,394],[15,707],[15,623],[15,352],[15,508],[15,236],[15,495],[15,763],[15,529],[15,205],[15,519],[15,797],[15,83],[15,760],[15,673],[15,845],[15,528],[15,583],[15,407],[15,865],[15,721],[15,887],[15,510],[15,668],[15,84],[15,475],[15,927],[15,526],[15,947],[15,600],[15,686],[15,87],[15,433],[15,988],[15,162],[15,216],[15,555],[15,322],[15,569],[15,768],[15,63],[15,217],[15,523],[15,436],[15,1053],[15,1054],[15,147],[15,604],[15,492],[15,228],[15,1062],[15,1034],[15,648],[15,1067],[15,1069],[15,146],[15,19],[15,393],[15,565],[15,1129],[15,729],[15,1144],[15,631],[15,278],[15,75],[15,391],[15,720],[15,1168],[15,290],[15,532],[15,482],[15,1194],[15,1201],[15,536],[15,862],[15,227],[15,1059],[15,1213],[15,888],[16,21],[16,33],[16,53],[16,104],[16,143],[16,179],[16,191],[16,207],[16,215],[16,298],[16,259],[16,365],[16,270],[16,377],[16,19],[16,94],[16,93],[16,126],[16,65],[16,145],[16,287],[16,52],[16,54],[16,28],[16,56],[16,69],[16,101],[16,187],[16,70],[16,467],[16,122],[16,184],[16,477],[16,272],[16,518],[16,578],[16,580],[16,30],[16,587],[16,563],[16,60],[16,606],[16,43],[16,522],[16,628],[16,73],[16,634],[16,51],[16,66],[16,491],[16,209],[16,221],[16,267],[16,230],[16,707],[16,214],[16,197],[16,472],[16,208],[16,103],[16,274],[16,623],[16,731],[16,736],[16,412],[16,743],[16,521],[16,508],[16,537],[16,266],[16,529],[16,481],[16,386],[16,25],[16,489],[16,519],[16,789],[16,506],[16,50],[16,818],[16,67],[16,826],[16,673],[16,478],[16,632],[16,528],[16,583],[16,721],[16,334],[16,510],[16,533],[16,486],[16,418],[16,542],[16,668],[16,761],[16,932],[16,526],[16,331],[16,941],[16,488],[16,605],[16,705],[16,600],[16,388],[16,686],[16,512],[16,965],[16,971],[16,210],[16,483],[16,498],[16,90],[16,199],[16,500],[16,317],[16,988],[16,265],[16,588],[16,105],[16,468],[16,115],[16,35],[16,322],[16,345],[16,497],[16,517],[16,292],[16,160],[16,1044],[16,768],[16,217],[16,523],[16,1029],[16,436],[16,575],[16,695],[16,147],[16,604],[16,372],[16,203],[16,663],[16,195],[16,200],[16,1027],[16,146],[16,189],[16,340],[16,511],[16,342],[16,1115],[16,739],[16,538],[16,123],[16,999],[16,1045],[16,579],[16,710],[16,1137],[16,1086],[16,1138],[16,1004],[16,1140],[16,631],[16,1162],[16,75],[16,527],[16,398],[16,391],[16,972],[16,720],[16,193],[16,532],[16,415],[16,367],[16,1166],[16,482],[16,1182],[16,1191],[16,485],[16,40],[16,1198],[16,534],[16,525],[16,301],[16,1206],[16,536],[16,862],[16,174],[16,704],[16,417],[16,860],[16,273],[17,191],[17,401],[17,409],[17,195],[17,596],[17,271],[17,408],[17,417],[17,540],[17,147],[17,527],[17,597],[17,457],[17,24],[17,96],[17,146],[17,519],[17,77],[17,598],[17,599],[17,488],[17,103],[17,99],[17,600],[17,601],[17,602],[17,504],[17,523],[17,603],[17,199],[17,510],[17,203],[17,604],[17,201],[17,537],[17,536],[17,480],[17,524],[17,387],[17,521],[17,761],[17,81],[17,833],[18,19],[18,20],[18,401],[18,187],[19,92],[19,148],[19,149],[19,179],[19,215],[19,297],[19,270],[19,432],[19,437],[19,405],[19,452],[19,455],[19,467],[19,469],[19,184],[19,477],[19,28],[19,545],[19,554],[19,395],[19,518],[19,571],[19,93],[19,582],[19,563],[19,622],[19,634],[19,636],[19,51],[19,657],[19,660],[19,491],[19,294],[19,438],[19,20],[19,638],[19,679],[19,221],[19,196],[19,88],[19,689],[19,24],[19,214],[19,98],[19,596],[19,208],[19,504],[19,730],[19,731],[19,661],[19,743],[19,269],[19,755],[19,508],[19,537],[19,69],[19,584],[19,495],[19,392],[19,765],[19,192],[19,519],[19,793],[19,190],[19,807],[19,494],[19,83],[19,760],[19,823],[19,824],[19,535],[19,673],[19,838],[19,840],[19,632],[19,528],[19,849],[19,583],[19,858],[19,585],[19,407],[19,869],[19,505],[19,501],[19,887],[19,510],[19,484],[19,26],[19,668],[19,781],[19,761],[19,920],[19,926],[19,475],[19,930],[19,513],[19,932],[19,938],[19,488],[19,944],[19,949],[19,953],[19,388],[19,686],[19,512],[19,936],[19,964],[19,441],[19,414],[19,433],[19,210],[19,483],[19,976],[19,199],[19,978],[19,500],[19,265],[19,81],[19,994],[19,216],[19,555],[19,468],[19,1038],[19,517],[19,355],[19,523],[19,408],[19,436],[19,411],[19,1049],[19,635],[19,759],[19,712],[19,492],[19,889],[19,36],[19,857],[19,200],[19,502],[19,597],[19,681],[19,479],[19,146],[19,187],[19,1078],[19,286],[19,65],[19,439],[19,80],[19,831],[19,604],[19,203],[19,54],[19,558],[19,27],[19,476],[19,530],[19,263],[19,1083],[19,598],[19,525],[19,729],[19,524],[19,639],[19,68],[19,398],[19,147],[19,603],[19,794],[19,75],[19,385],[19,511],[19,557],[19,739],[19,415],[19,507],[19,391],[19,222],[19,577],[19,1084],[19,189],[19,532],[19,599],[19,474],[19,417],[19,589],[19,193],[19,562],[19,82],[19,738],[19,206],[19,713],[19,231],[19,721],[19,32],[19,470],[19,644],[19,722],[19,89],[19,485],[19,503],[19,340],[19,496],[19,531],[19,1085],[19,406],[19,565],[19,278],[19,480],[19,272],[19,31],[19,293],[19,520],[19,877],[19,1086],[19,96],[19,1087],[19,401],[19,923],[19,99],[19,835],[19,482],[19,1088],[19,143],[19,182],[19,389],[19,516],[19,509],[19,410],[19,1060],[19,559],[19,1052],[19,942],[19,194],[19,1089],[19,971],[19,396],[19,416],[19,453],[19,472],[19,397],[19,202],[19,1090],[19,1091],[19,1092],[19,404],[19,675],[19,1093],[19,888],[19,806],[19,720],[19,782],[19,699],[19,805],[19,493],[19,1094],[19,1095],[19,704],[19,749],[19,803],[19,487],[19,1096],[19,1097],[19,1098],[19,1099],[19,413],[19,1100],[19,1101],[19,716],[19,1102],[19,1103],[19,709],[19,515],[19,273],[19,1104],[19,412],[19,1105],[19,973],[19,710],[19,1106],[19,1107],[19,1108],[19,1109],[19,997],[19,59],[19,631],[19,1162],[19,527],[19,564],[19,1166],[19,1191],[19,1212],[19,1213],[20,215],[20,293],[20,297],[20,405],[20,455],[20,184],[20,545],[20,531],[20,30],[20,209],[20,187],[20,223],[20,86],[20,83],[20,88],[20,98],[20,504],[20,661],[20,743],[20,521],[20,508],[20,765],[20,192],[20,793],[20,673],[20,528],[20,583],[20,721],[20,767],[20,501],[20,761],[20,925],[20,475],[20,599],[20,932],[20,130],[20,512],[20,210],[20,199],[20,500],[20,265],[20,588],[20,468],[20,1038],[20,712],[20,857],[20,502],[20,189],[20,520],[20,831],[20,557],[20,194],[20,530],[20,397],[20,263],[20,385],[20,527],[20,391],[20,482],[20,1191],[20,480],[20,794],[20,503],[20,536],[20,738],[20,589],[20,722],[21,22],[21,23],[21,24],[21,25],[21,26],[21,27],[21,28],[21,29],[21,30],[21,31],[21,32],[21,427],[21,1068],[21,710],[22,53],[22,179],[22,215],[22,270],[22,184],[22,28],[22,531],[22,29],[22,145],[22,118],[22,606],[22,73],[22,77],[22,427],[22,220],[22,62],[22,66],[22,69],[22,61],[22,126],[22,65],[22,26],[22,67],[22,50],[22,24],[22,94],[22,212],[22,72],[22,248],[22,68],[22,82],[22,52],[22,71],[22,43],[22,27],[22,56],[22,60],[22,224],[22,225],[22,573],[22,88],[22,541],[22,214],[22,581],[22,743],[22,470],[22,23],[22,761],[22,600],[22,87],[22,409],[22,81],[22,79],[22,604],[22,663],[22,146],[22,1114],[22,598],[22,1129],[22,278],[22,527],[22,532],[22,415],[22,722],[23,191],[23,215],[23,467],[23,184],[23,60],[23,606],[23,73],[23,220],[23,491],[23,269],[23,481],[23,812],[23,583],[23,721],[23,24],[23,56],[23,218],[23,427],[23,475],[23,488],[23,605],[23,705],[23,686],[23,965],[23,498],[23,94],[23,216],[23,569],[23,523],[23,663],[23,857],[23,1067],[23,146],[23,189],[23,1114],[23,565],[23,598],[23,1129],[23,710],[23,1020],[23,562],[23,729],[23,385],[23,1162],[23,527],[23,78],[23,532],[23,86],[23,1194],[23,534],[23,474],[23,536],[23,704],[23,1059],[24,53],[24,143],[24,285],[24,184],[24,28],[24,531],[24,571],[24,145],[24,30],[24,628],[24,101],[24,427],[24,222],[24,220],[24,93],[24,268],[24,536],[24,67],[24,56],[24,54],[24,94],[24,81],[24,701],[24,62],[24,61],[24,29],[24,52],[24,46],[24,96],[24,185],[24,69],[24,187],[24,100],[24,65],[24,63],[24,26],[24,27],[24,569],[24,211],[24,635],[24,103],[24,409],[24,538],[24,580],[24,476],[24,408],[24,83],[24,77],[24,488],[24,702],[24,560],[24,703],[24,704],[24,705],[24,266],[24,25],[24,598],[24,213],[24,755],[24,537],[24,766],[24,519],[24,666],[24,797],[24,721],[24,876],[24,501],[24,897],[24,668],[24,475],[24,599],[24,600],[24,498],[24,988],[24,216],[24,189],[24,511],[24,520],[24,717],[24,1129],[24,710],[24,729],[24,385],[24,527],[24,1170],[24,474],[25,268],[25,60],[25,606],[25,73],[25,573],[25,274],[25,623],[25,632],[25,453],[25,223],[25,211],[25,29],[25,218],[25,780],[25,68],[25,639],[25,599],[25,294],[25,99],[25,186],[25,81],[25,212],[25,491],[25,391],[25,92],[25,454],[25,69],[25,89],[25,82],[25,452],[25,187],[25,781],[25,28],[25,100],[25,150],[25,663],[25,504],[25,53],[25,93],[25,485],[25,49],[25,50],[25,59],[25,441],[25,782],[25,574],[25,152],[25,26],[25,87],[25,31],[25,74],[25,592],[25,88],[25,67],[25,572],[25,30],[25,43],[25,427],[25,598],[25,248],[25,144],[25,114],[25,56],[25,219],[25,319],[25,626],[25,519],[25,528],[25,721],[25,686],[25,1068],[25,1020],[25,278],[25,833],[25,527],[25,182],[26,53],[26,191],[26,215],[26,285],[26,270],[26,435],[26,452],[26,455],[26,469],[26,122],[26,184],[26,518],[26,531],[26,29],[26,93],[26,145],[26,46],[26,60],[26,606],[26,43],[26,73],[26,77],[26,427],[26,639],[26,220],[26,660],[26,662],[26,186],[26,491],[26,573],[26,219],[26,88],[26,213],[26,730],[26,743],[26,269],[26,470],[26,69],[26,392],[26,519],[26,506],[26,561],[26,190],[26,83],[26,760],[26,67],[26,56],[26,673],[26,632],[26,528],[26,583],[26,721],[26,58],[26,100],[26,668],[26,926],[26,488],[26,947],[26,441],[26,965],[26,433],[26,90],[26,265],[26,94],[26,1019],[26,555],[26,497],[26,217],[26,436],[26,411],[26,575],[26,712],[26,1053],[26,604],[26,663],[26,36],[26,857],[26,1069],[26,479],[26,146],[26,601],[26,74],[26,187],[26,189],[26,520],[26,1060],[26,49],[26,717],[26,538],[26,598],[26,59],[26,1129],[26,1004],[26,80],[26,1144],[26,631],[26,385],[26,1162],[26,75],[26,527],[26,391],[26,720],[26,1170],[26,193],[26,57],[26,482],[26,206],[26,86],[26,480],[26,474],[26,794],[26,835],[26,273],[27,179],[27,270],[27,467],[27,395],[27,518],[27,145],[27,118],[27,606],[27,73],[27,71],[27,72],[27,427],[27,212],[27,209],[27,222],[27,576],[27,698],[27,126],[27,214],[27,476],[27,197],[27,596],[27,623],[27,731],[27,269],[27,584],[27,392],[27,765],[27,766],[27,271],[27,52],[27,67],[27,399],[27,306],[27,475],[27,498],[27,90],[27,94],[27,468],[27,335],[27,497],[27,217],[27,436],[27,1049],[27,1051],[27,575],[27,558],[27,604],[27,663],[27,36],[27,1027],[27,146],[27,189],[27,231],[27,710],[27,80],[27,1144],[27,631],[27,398],[27,391],[27,1170],[27,532],[27,206],[27,86],[27,474],[27,182],[27,1213],[28,53],[28,92],[28,179],[28,191],[28,215],[28,261],[28,293],[28,297],[28,270],[28,435],[28,467],[28,469],[28,184],[28,102],[28,56],[28,125],[28,530],[28,110],[28,540],[28,64],[28,541],[28,185],[28,145],[28,287],[28,226],[28,219],[28,69],[28,186],[28,73],[28,542],[28,63],[28,68],[28,518],[28,531],[28,570],[28,571],[28,29],[28,93],[28,188],[28,30],[28,587],[28,118],[28,60],[28,609],[28,628],[28,72],[28,634],[28,77],[28,51],[28,101],[28,66],[28,427],[28,100],[28,220],[28,218],[28,209],[28,222],[28,576],[28,221],[28,572],[28,573],[28,88],[28,655],[28,624],[28,214],[28,476],[28,715],[28,472],[28,664],[28,623],[28,730],[28,470],[28,537],[28,495],[28,392],[28,775],[28,519],[28,789],[28,506],[28,666],[28,811],[28,52],[28,83],[28,818],[28,640],[28,760],[28,67],[28,478],[28,528],[28,583],[28,407],[28,865],[28,876],[28,767],[28,65],[28,505],[28,501],[28,886],[28,668],[28,761],[28,475],[28,871],[28,941],[28,488],[28,201],[28,705],[28,130],[28,600],[28,309],[28,762],[28,965],[28,433],[28,498],[28,90],[28,94],[28,1000],[28,140],[28,204],[28,216],[28,497],[28,517],[28,569],[28,515],[28,217],[28,523],[28,436],[28,575],[28,524],[28,604],[28,492],[28,663],[28,228],[28,36],[28,200],[28,502],[28,1067],[28,1027],[28,146],[28,74],[28,187],[28,189],[28,565],[28,1116],[28,316],[28,717],[28,538],[28,598],[28,1129],[28,710],[28,1086],[28,1020],[28,562],[28,729],[28,80],[28,1144],[28,1145],[28,631],[28,385],[28,75],[28,527],[28,398],[28,391],[28,78],[28,720],[28,532],[28,703],[28,344],[28,482],[28,206],[28,1191],[28,480],[28,485],[28,474],[28,1201],[28,1094],[28,301],[28,536],[28,862],[28,704],[28,1059],[28,577],[28,879],[29,148],[29,179],[29,215],[29,112],[29,68],[29,285],[29,435],[29,469],[29,184],[29,518],[29,568],[29,571],[29,63],[29,144],[29,67],[29,56],[29,60],[29,230],[29,43],[29,75],[29,572],[29,185],[29,69],[29,573],[29,72],[29,100],[29,183],[29,574],[29,87],[29,94],[29,71],[29,58],[29,64],[29,30],[29,145],[29,587],[29,563],[29,606],[29,628],[29,73],[29,66],[29,427],[29,209],[29,224],[29,581],[29,623],[29,269],[29,519],[29,506],[29,83],[29,478],[29,583],[29,721],[29,874],[29,505],[29,542],[29,475],[29,871],[29,488],[29,947],[29,605],[29,705],[29,600],[29,686],[29,936],[29,498],[29,216],[29,569],[29,768],[29,217],[29,436],[29,1053],[29,604],[29,663],[29,502],[29,597],[29,146],[29,189],[29,400],[29,538],[29,598],[29,710],[29,729],[29,278],[29,1162],[29,527],[29,398],[29,720],[29,1170],[29,532],[29,86],[29,474],[29,182],[29,1209],[29,704],[29,1059],[30,53],[30,104],[30,179],[30,215],[30,381],[30,270],[30,435],[30,122],[30,518],[30,145],[30,257],[30,209],[30,183],[30,54],[30,186],[30,85],[30,83],[30,81],[30,581],[30,498],[30,60],[30,58],[30,572],[30,63],[30,61],[30,64],[30,86],[30,181],[30,46],[30,606],[30,43],[30,73],[30,72],[30,77],[30,101],[30,222],[30,576],[30,88],[30,707],[30,269],[30,69],[30,154],[30,519],[30,666],[30,52],[30,67],[30,583],[30,721],[30,874],[30,542],[30,84],[30,475],[30,488],[30,605],[30,705],[30,87],[30,265],[30,94],[30,239],[30,569],[30,217],[30,436],[30,885],[30,604],[30,1070],[30,255],[30,189],[30,400],[30,538],[30,59],[30,356],[30,1144],[30,1162],[30,75],[30,527],[30,398],[30,391],[30,720],[30,532],[30,482],[30,1186],[30,206],[30,474],[30,227],[30,182],[30,704],[31,148],[31,135],[31,381],[31,450],[31,566],[31,591],[31,491],[31,690],[31,274],[31,623],[31,734],[31,751],[31,508],[31,537],[31,786],[31,802],[31,535],[31,837],[31,632],[31,848],[31,850],[31,852],[31,510],[31,895],[31,358],[31,917],[31,923],[31,475],[31,331],[31,488],[31,433],[31,498],[31,444],[31,1022],[31,497],[31,674],[31,515],[31,575],[31,341],[31,889],[31,663],[31,36],[31,466],[31,1027],[31,681],[31,1082],[31,342],[31,530],[31,1060],[31,739],[31,316],[31,717],[31,373],[31,325],[31,1134],[31,729],[31,1156],[31,193],[31,614],[31,1180],[31,482],[31,480],[31,737],[31,1195],[31,911],[31,992],[31,1201],[31,174],[31,1149],[31,860],[32,780],[32,510],[32,206],[32,852],[32,144],[32,294],[33,34],[33,35],[33,36],[33,37],[34,191],[34,261],[34,405],[34,609],[34,636],[34,209],[34,731],[34,519],[34,673],[34,528],[34,407],[34,927],[34,441],[34,989],[34,468],[34,408],[34,492],[34,681],[34,74],[34,187],[34,717],[34,416],[34,631],[34,527],[34,391],[34,720],[34,703],[34,406],[34,1186],[34,480],[34,1088],[34,182],[34,722],[35,191],[35,535],[35,60],[35,193],[35,187],[35,801],[35,196],[36,184],[36,508],[36,495],[36,766],[36,519],[36,850],[36,861],[36,932],[36,498],[36,204],[36,468],[36,558],[36,638],[36,188],[36,667],[36,71],[36,222],[36,266],[36,54],[36,675],[36,56],[36,224],[36,573],[36,185],[36,287],[36,453],[36,182],[36,480],[36,395],[36,73],[36,1060],[36,563],[36,221],[36,487],[36,273],[36,99],[36,81],[36,410],[36,572],[36,193],[36,851],[36,69],[36,486],[36,656],[36,187],[36,100],[36,482],[36,560],[36,520],[36,705],[36,499],[36,760],[36,576],[36,294],[36,439],[36,396],[36,710],[36,720],[36,474],[36,722],[37,42],[37,247],[37,467],[37,587],[37,636],[37,267],[37,213],[37,623],[37,192],[37,659],[37,190],[37,673],[37,869],[37,475],[37,408],[37,575],[37,712],[37,187],[37,189],[37,340],[37,49],[37,710],[37,782],[37,182],[37,273],[38,39],[38,40],[38,41],[39,112],[39,135],[39,247],[39,354],[39,41],[39,346],[39,122],[39,107],[39,160],[39,116],[39,355],[39,119],[39,306],[39,174],[39,257],[39,741],[39,333],[39,779],[39,856],[39,254],[39,127],[39,105],[39,108],[39,362],[39,256],[39,322],[39,255],[39,316],[39,106],[39,167],[39,136],[39,125],[40,247],[40,41],[40,567],[40,457],[40,306],[40,335],[40,322],[40,695],[40,991],[40,382],[40,379],[40,1073],[40,1196],[40,428],[40,107],[40,277],[40,1197],[40,280],[40,402],[40,422],[40,256],[40,301],[40,676],[40,744],[40,302],[40,929],[40,352],[40,336],[40,135],[40,308],[40,374],[40,678],[40,117],[40,915],[40,84],[40,305],[40,360],[40,125],[40,157],[40,434],[40,456],[40,364],[40,882],[40,344],[40,331],[41,104],[41,124],[41,298],[41,135],[41,365],[41,247],[41,354],[41,248],[41,343],[41,444],[41,257],[41,130],[41,245],[41,328],[41,277],[41,158],[41,108],[41,109],[41,237],[41,85],[41,110],[41,289],[41,132],[41,162],[41,133],[41,107],[41,324],[41,111],[41,112],[41,129],[41,134],[41,57],[41,168],[41,161],[41,352],[41,105],[41,113],[41,165],[41,445],[41,167],[41,251],[41,136],[41,45],[41,114],[41,323],[41,137],[41,138],[41,304],[41,355],[41,335],[41,171],[41,356],[41,115],[41,250],[41,320],[41,139],[41,117],[41,116],[41,256],[41,84],[41,446],[41,173],[41,447],[41,276],[41,448],[41,449],[41,119],[41,106],[41,306],[41,120],[41,314],[41,141],[41,44],[41,174],[41,254],[41,175],[41,121],[41,172],[41,123],[41,91],[41,125],[41,233],[41,140],[41,359],[41,292],[41,364],[41,142],[41,244],[41,249],[41,291],[41,127],[41,239],[41,128],[41,362],[41,318],[41,299],[41,649],[41,650],[41,694],[41,163],[41,741],[41,333],[41,347],[41,154],[41,546],[41,779],[41,548],[41,922],[41,465],[41,302],[41,351],[41,322],[41,345],[41,160],[41,674],[41,258],[41,695],[41,1005],[41,255],[41,1002],[41,1115],[41,321],[41,842],[41,316],[41,1140],[41,421],[41,1144],[41,391],[41,375],[42,43],[42,44],[42,45],[42,46],[42,47],[42,48],[42,49],[42,50],[42,51],[42,52],[42,262],[43,53],[43,215],[43,247],[43,270],[43,435],[43,122],[43,58],[43,46],[43,60],[43,606],[43,61],[43,80],[43,572],[43,44],[43,84],[43,65],[43,85],[43,68],[43,81],[43,183],[43,89],[43,118],[43,52],[43,188],[43,136],[43,45],[43,86],[43,63],[43,190],[43,49],[43,50],[43,574],[43,71],[43,140],[43,427],[43,64],[43,66],[43,67],[43,628],[43,72],[43,209],[43,88],[43,274],[43,766],[43,561],[43,839],[43,542],[43,922],[43,87],[43,90],[43,335],[43,497],[43,217],[43,436],[43,189],[43,1115],[43,356],[43,1144],[43,527],[43,391],[43,1178],[43,174],[43,1218],[43,509],[44,104],[44,112],[44,365],[44,247],[44,242],[44,426],[44,248],[44,117],[44,122],[44,582],[44,46],[44,66],[44,617],[44,650],[44,697],[44,741],[44,742],[44,333],[44,154],[44,777],[44,171],[44,790],[44,190],[44,50],[44,45],[44,379],[44,652],[44,632],[44,779],[44,855],[44,138],[44,245],[44,856],[44,168],[44,447],[44,548],[44,84],[44,928],[44,943],[44,165],[44,87],[44,425],[44,254],[44,317],[44,132],[44,109],[44,161],[44,85],[44,128],[44,127],[44,105],[44,140],[44,323],[44,108],[44,314],[44,324],[44,313],[44,113],[44,233],[44,335],[44,235],[44,322],[44,239],[44,292],[44,252],[44,160],[44,674],[44,257],[44,1029],[44,648],[44,255],[44,462],[44,1113],[44,1115],[44,321],[44,107],[44,49],[44,842],[44,354],[44,114],[44,91],[44,156],[44,106],[44,116],[44,167],[44,356],[44,1004],[44,1140],[44,353],[44,670],[44,391],[44,332],[44,972],[44,720],[44,1168],[44,448],[44,110],[44,367],[44,136],[44,1186],[44,1132],[44,139],[44,375],[44,911],[44,374],[44,174],[44,822],[45,53],[45,104],[45,215],[45,124],[45,112],[45,166],[45,288],[45,135],[45,247],[45,242],[45,426],[45,248],[45,346],[45,122],[45,58],[45,46],[45,607],[45,299],[45,72],[45,51],[45,66],[45,696],[45,697],[45,223],[45,741],[45,750],[45,69],[45,333],[45,775],[45,154],[45,251],[45,171],[45,791],[45,798],[45,50],[45,91],[45,234],[45,107],[45,116],[45,114],[45,235],[45,136],[45,239],[45,84],[45,236],[45,245],[45,52],[45,626],[45,67],[45,839],[45,306],[45,855],[45,138],[45,407],[45,874],[45,548],[45,940],[45,943],[45,302],[45,425],[45,254],[45,317],[45,132],[45,109],[45,162],[45,85],[45,128],[45,105],[45,140],[45,323],[45,108],[45,324],[45,249],[45,113],[45,233],[45,256],[45,1037],[45,335],[45,322],[45,253],[45,355],[45,133],[45,292],[45,252],[45,160],[45,674],[45,250],[45,569],[45,1029],[45,436],[45,258],[45,604],[45,648],[45,255],[45,187],[45,1002],[45,422],[45,311],[45,565],[45,169],[45,400],[45,360],[45,842],[45,336],[45,349],[45,155],[45,999],[45,167],[45,356],[45,1140],[45,421],[45,398],[45,391],[45,972],[45,720],[45,1168],[45,290],[45,289],[45,367],[45,1011],[45,1132],[45,139],[45,86],[45,474],[45,981],[45,1207],[45,1073],[45,860],[46,53],[46,215],[46,247],[46,122],[46,58],[46,52],[46,69],[46,54],[46,100],[46,66],[46,86],[46,84],[46,85],[46,61],[46,70],[46,91],[46,63],[46,225],[46,65],[46,136],[46,67],[46,94],[46,606],[46,615],[46,72],[46,662],[46,88],[46,706],[46,623],[46,742],[46,333],[46,154],[46,583],[46,475],[46,87],[46,433],[46,132],[46,140],[46,216],[46,324],[46,335],[46,497],[46,239],[46,674],[46,569],[46,217],[46,408],[46,436],[46,604],[46,663],[46,146],[46,462],[46,400],[46,49],[46,1129],[46,356],[46,1004],[46,80],[46,1144],[46,75],[46,391],[46,78],[46,720],[46,532],[46,1186],[46,1132],[46,804],[46,182],[46,1059],[47,467],[47,491],[47,470],[47,190],[47,399],[47,632],[47,475],[47,559],[47,78],[47,193],[47,794],[47,264],[48,262],[48,267],[49,53],[49,143],[49,179],[49,215],[49,135],[49,469],[49,184],[49,58],[49,77],[49,51],[49,224],[49,221],[49,88],[49,69],[49,584],[49,190],[49,271],[49,52],[49,760],[49,67],[49,840],[49,583],[49,84],[49,475],[49,488],[49,87],[49,988],[49,132],[49,85],[49,216],[49,217],[49,436],[49,663],[49,1027],[49,189],[49,393],[49,65],[49,226],[49,55],[49,212],[49,193],[49,75],[49,86],[49,223],[49,78],[49,59],[49,356],[49,562],[49,1144],[49,1162],[49,391],[49,720],[49,804],[49,536],[49,182],[50,53],[50,179],[50,455],[50,60],[50,73],[50,427],[50,639],[50,220],[50,491],[50,209],[50,680],[50,221],[50,88],[50,655],[50,197],[50,715],[50,103],[50,269],[50,190],[50,56],[50,112],[50,110],[50,54],[50,107],[50,84],[50,232],[50,81],[50,86],[50,165],[50,123],[50,166],[50,175],[50,556],[50,178],[50,534],[50,626],[50,67],[50,721],[50,505],[50,668],[50,761],[50,932],[50,947],[50,705],[50,686],[50,87],[50,140],[50,569],[50,515],[50,217],[50,436],[50,648],[50,146],[50,187],[50,189],[50,831],[50,557],[50,1129],[50,356],[50,710],[50,80],[50,75],[50,78],[50,972],[50,264],[50,592],[50,89],[50,1221],[50,879],[51,270],[51,346],[51,184],[51,518],[51,58],[51,65],[51,193],[51,59],[51,383],[51,289],[51,98],[51,144],[51,637],[51,69],[51,86],[51,190],[51,78],[51,187],[51,221],[51,638],[51,639],[51,99],[51,100],[51,93],[51,56],[51,640],[51,265],[51,641],[51,66],[51,85],[51,61],[51,124],[51,396],[51,662],[51,88],[51,519],[51,528],[51,932],[51,488],[51,515],[51,217],[51,575],[51,604],[51,999],[51,1129],[51,710],[51,527],[51,398],[51,720],[51,532],[52,53],[52,104],[52,179],[52,215],[52,298],[52,270],[52,164],[52,568],[52,145],[52,587],[52,60],[52,72],[52,634],[52,77],[52,427],[52,649],[52,671],[52,88],[52,327],[52,624],[52,581],[52,69],[52,546],[52,379],[52,91],[52,107],[52,284],[52,185],[52,110],[52,81],[52,183],[52,86],[52,156],[52,439],[52,161],[52,287],[52,277],[52,70],[52,58],[52,71],[52,142],[52,67],[52,126],[52,63],[52,66],[52,83],[52,652],[52,535],[52,528],[52,583],[52,767],[52,505],[52,810],[52,952],[52,87],[52,498],[52,976],[52,94],[52,993],[52,105],[52,216],[52,569],[52,436],[52,258],[52,575],[52,604],[52,1061],[52,189],[52,80],[52,1144],[52,631],[52,1159],[52,75],[52,527],[52,398],[52,532],[52,206],[52,1191],[52,474],[52,301],[52,182],[52,310],[52,374],[52,1218],[53,54],[53,55],[53,56],[53,57],[53,58],[53,59],[53,60],[53,61],[53,62],[53,63],[53,64],[53,65],[53,66],[53,67],[53,68],[53,69],[53,70],[53,71],[53,72],[53,73],[53,74],[53,75],[53,76],[53,77],[53,78],[53,79],[53,80],[53,81],[53,82],[53,83],[53,84],[53,85],[53,86],[53,87],[53,88],[53,89],[53,90],[53,91],[53,179],[53,582],[53,427],[53,662],[53,576],[53,221],[53,540],[53,583],[53,668],[53,475],[53,705],[53,265],[53,497],[53,569],[53,217],[53,1029],[53,575],[53,604],[53,1067],[53,146],[53,189],[53,1129],[53,710],[53,1158],[53,527],[53,391],[53,720],[53,804],[53,574],[53,474],[53,1059],[53,1163],[54,92],[54,215],[54,467],[54,469],[54,184],[54,582],[54,587],[54,606],[54,628],[54,77],[54,101],[54,639],[54,100],[54,220],[54,212],[54,491],[54,209],[54,222],[54,224],[54,576],[54,573],[54,327],[54,698],[54,624],[54,214],[54,98],[54,476],[54,197],[54,596],[54,472],[54,623],[54,730],[54,82],[54,269],[54,470],[54,69],[54,584],[54,766],[54,529],[54,519],[54,789],[54,506],[54,797],[54,812],[54,67],[54,56],[54,535],[54,478],[54,583],[54,721],[54,767],[54,668],[54,761],[54,475],[54,929],[54,488],[54,947],[54,201],[54,498],[54,500],[54,265],[54,94],[54,216],[54,1032],[54,359],[54,515],[54,217],[54,436],[54,1049],[54,1053],[54,524],[54,604],[54,492],[54,663],[54,200],[54,502],[54,1067],[54,1027],[54,597],[54,146],[54,74],[54,187],[54,189],[54,511],[54,1114],[54,565],[54,520],[54,1060],[54,538],[54,710],[54,1086],[54,562],[54,278],[54,385],[54,75],[54,527],[54,398],[54,720],[54,1170],[54,193],[54,532],[54,406],[54,482],[54,86],[54,536],[54,862],[54,1209],[54,704],[54,1213],[54,453],[54,180],[54,110],[54,107],[54,144],[54,273],[55,215],[55,268],[55,184],[55,58],[55,72],[55,126],[55,394],[55,581],[55,519],[55,190],[55,83],[55,67],[55,583],[55,475],[55,94],[55,85],[55,217],[55,1029],[55,436],[55,604],[55,1027],[55,514],[55,187],[55,189],[55,1114],[55,1115],[55,1144],[55,1145],[55,527],[55,78],[55,720],[55,536],[55,60],[55,84],[55,1217],[56,92],[56,143],[56,148],[56,137],[56,179],[56,191],[56,215],[56,112],[56,285],[56,293],[56,259],[56,270],[56,442],[56,455],[56,467],[56,469],[56,346],[56,184],[56,272],[56,568],[56,571],[56,93],[56,58],[56,145],[56,582],[56,587],[56,60],[56,606],[56,622],[56,634],[56,77],[56,101],[56,427],[56,100],[56,220],[56,662],[56,491],[56,209],[56,224],[56,221],[56,685],[56,88],[56,230],[56,624],[56,214],[56,76],[56,581],[56,457],[56,713],[56,596],[56,472],[56,274],[56,623],[56,730],[56,82],[56,521],[56,269],[56,755],[56,508],[56,537],[56,69],[56,765],[56,775],[56,473],[56,519],[56,789],[56,659],[56,506],[56,666],[56,797],[56,271],[56,812],[56,379],[56,83],[56,640],[56,760],[56,540],[56,94],[56,180],[56,535],[56,478],[56,528],[56,583],[56,656],[56,407],[56,861],[56,721],[56,874],[56,767],[56,878],[56,501],[56,887],[56,418],[56,542],[56,668],[56,475],[56,599],[56,488],[56,947],[56,605],[56,705],[56,600],[56,686],[56,87],[56,965],[56,433],[56,498],[56,90],[56,978],[56,500],[56,409],[56,979],[56,988],[56,265],[56,85],[56,140],[56,204],[56,216],[56,324],[56,555],[56,249],[56,468],[56,362],[56,335],[56,235],[56,497],[56,569],[56,768],[56,63],[56,515],[56,217],[56,523],[56,436],[56,575],[56,1053],[56,604],[56,492],[56,889],[56,663],[56,228],[56,857],[56,1062],[56,502],[56,625],[56,1067],[56,1041],[56,597],[56,146],[56,74],[56,187],[56,1002],[56,189],[56,565],[56,1115],[56,557],[56,440],[56,1060],[56,717],[56,59],[56,1129],[56,356],[56,710],[56,1020],[56,562],[56,729],[56,80],[56,1144],[56,1145],[56,631],[56,496],[56,278],[56,385],[56,1162],[56,75],[56,527],[56,398],[56,916],[56,391],[56,720],[56,193],[56,532],[56,482],[56,86],[56,480],[56,387],[56,574],[56,474],[56,301],[56,536],[56,862],[56,227],[56,1209],[56,704],[56,125],[56,738],[56,417],[56,1059],[56,1213],[56,577],[56,1214],[56,1163],[56,879],[56,273],[57,104],[57,124],[57,298],[57,259],[57,365],[57,426],[57,248],[57,117],[57,122],[57,184],[57,550],[57,361],[57,434],[57,299],[57,642],[57,348],[57,567],[57,694],[57,696],[57,698],[57,707],[57,708],[57,581],[57,352],[57,741],[57,742],[57,750],[57,333],[57,347],[57,329],[57,775],[57,154],[57,777],[57,171],[57,379],[57,827],[57,306],[57,846],[57,779],[57,855],[57,138],[57,168],[57,447],[57,892],[57,893],[57,358],[57,548],[57,84],[57,933],[57,902],[57,943],[57,950],[57,403],[57,130],[57,165],[57,302],[57,425],[57,967],[57,317],[57,277],[57,132],[57,351],[57,162],[57,161],[57,85],[57,128],[57,127],[57,140],[57,323],[57,108],[57,324],[57,249],[57,1032],[57,313],[57,113],[57,233],[57,256],[57,335],[57,253],[57,308],[57,355],[57,239],[57,133],[57,252],[57,674],[57,257],[57,569],[57,63],[57,1029],[57,244],[57,258],[57,341],[57,969],[57,304],[57,141],[57,648],[57,269],[57,625],[57,1067],[57,368],[57,990],[57,255],[57,909],[57,601],[57,462],[57,1008],[57,422],[57,291],[57,311],[57,1113],[57,1114],[57,1115],[57,321],[57,107],[57,169],[57,557],[57,400],[57,842],[57,349],[57,156],[57,999],[57,167],[57,356],[57,1135],[57,1004],[57,1140],[57,114],[57,421],[57,1145],[57,319],[57,175],[57,78],[57,332],[57,972],[57,290],[57,289],[57,532],[57,820],[57,110],[57,367],[57,1075],[57,1178],[57,282],[57,129],[57,65],[57,1174],[57,1132],[57,139],[57,1189],[57,1191],[57,375],[57,119],[57,173],[57,241],[57,301],[57,536],[57,310],[57,374],[57,125],[57,822],[57,1073],[57,91],[57,307],[57,1216],[58,179],[58,215],[58,124],[58,247],[58,270],[58,354],[58,383],[58,469],[58,122],[58,184],[58,518],[58,577],[58,84],[58,65],[58,132],[58,136],[58,78],[58,61],[58,240],[58,70],[58,66],[58,225],[58,218],[58,63],[58,145],[58,60],[58,72],[58,77],[58,662],[58,224],[58,88],[58,698],[58,476],[58,596],[58,623],[58,743],[58,69],[58,481],[58,154],[58,519],[58,171],[58,271],[58,640],[58,67],[58,528],[58,855],[58,583],[58,475],[58,929],[58,488],[58,605],[58,87],[58,498],[58,94],[58,85],[58,816],[58,216],[58,324],[58,497],[58,292],[58,674],[58,569],[58,217],[58,1029],[58,408],[58,436],[58,1053],[58,604],[58,663],[58,648],[58,597],[58,681],[58,187],[58,189],[58,1060],[58,538],[58,123],[58,1129],[58,356],[58,562],[58,1144],[58,1161],[58,527],[58,398],[58,391],[58,332],[58,720],[58,1168],[58,532],[58,482],[58,1171],[58,1132],[58,86],[58,480],[58,474],[58,536],[58,182],[58,1209],[58,1059],[58,1220],[59,92],[59,179],[59,215],[59,135],[59,270],[59,469],[59,184],[59,531],[59,582],[59,587],[59,606],[59,101],[59,88],[59,624],[59,223],[59,664],[59,743],[59,269],[59,69],[59,766],[59,775],[59,519],[59,506],[59,83],[59,583],[59,407],[59,510],[59,475],[59,949],[59,87],[59,500],[59,409],[59,674],[59,768],[59,217],[59,523],[59,436],[59,681],[59,146],[59,187],[59,189],[59,393],[59,1060],[59,316],[59,100],[59,279],[59,80],[59,75],[59,805],[59,193],[59,67],[59,118],[59,86],[59,93],[59,60],[59,356],[59,710],[59,278],[59,385],[59,527],[59,398],[59,720],[59,482],[59,1191],[59,738],[59,1213],[59,1221],[60,104],[60,179],[60,270],[60,435],[60,122],[60,284],[60,61],[60,73],[60,80],[60,85],[60,110],[60,68],[60,81],[60,572],[60,212],[60,133],[60,90],[60,269],[60,89],[60,72],[60,605],[60,65],[60,190],[60,574],[60,119],[60,71],[60,91],[60,223],[60,82],[60,427],[60,64],[60,66],[60,67],[60,606],[60,573],[60,581],[60,742],[60,561],[60,84],[60,498],[60,1115],[60,1129],[60,1020],[60,278],[60,75],[60,391],[60,720],[60,532],[60,1186],[60,1191],[60,182],[60,704],[61,92],[61,104],[61,179],[61,215],[61,112],[61,261],[61,285],[61,135],[61,247],[61,270],[61,432],[61,467],[61,469],[61,122],[61,184],[61,164],[61,550],[61,145],[61,582],[61,72],[61,66],[61,427],[61,348],[61,662],[61,669],[61,576],[61,230],[61,706],[61,394],[61,211],[61,707],[61,463],[61,472],[61,623],[61,742],[61,743],[61,269],[61,236],[61,69],[61,529],[61,775],[61,791],[61,666],[61,795],[61,796],[61,817],[61,67],[61,540],[61,535],[61,846],[61,583],[61,407],[61,874],[61,65],[61,917],[61,810],[61,475],[61,932],[61,488],[61,958],[61,87],[61,317],[61,979],[61,94],[61,109],[61,140],[61,816],[61,324],[61,178],[61,113],[61,497],[61,292],[61,225],[61,569],[61,63],[61,998],[61,217],[61,1029],[61,408],[61,436],[61,258],[61,575],[61,1053],[61,341],[61,604],[61,228],[61,153],[61,648],[61,1067],[61,1027],[61,158],[61,1069],[61,1072],[61,74],[61,187],[61,312],[61,1002],[61,422],[61,402],[61,1116],[61,1117],[61,155],[61,999],[61,106],[61,356],[61,1004],[61,80],[61,1144],[61,631],[61,398],[61,391],[61,78],[61,720],[61,1168],[61,532],[61,344],[61,1178],[61,1166],[61,898],[61,1186],[61,1132],[61,1187],[61,86],[61,1201],[61,182],[61,1209],[61,125],[61,1059],[61,1213],[61,577],[61,509],[62,261],[62,432],[62,73],[62,72],[62,427],[62,211],[62,213],[62,797],[62,640],[62,542],[62,488],[62,600],[62,87],[62,140],[62,1053],[62,492],[62,834],[62,1069],[62,644],[62,565],[62,1119],[62,75],[62,480],[62,536],[62,1059],[63,215],[63,298],[63,435],[63,122],[63,164],[63,568],[63,145],[63,582],[63,587],[63,622],[63,72],[63,634],[63,66],[63,649],[63,662],[63,209],[63,88],[63,581],[63,742],[63,743],[63,69],[63,333],[63,546],[63,171],[63,626],[63,67],[63,84],[63,475],[63,87],[63,286],[63,81],[63,94],[63,85],[63,216],[63,497],[63,569],[63,70],[63,86],[63,91],[63,277],[63,217],[63,1029],[63,1053],[63,604],[63,648],[63,189],[63,1110],[63,1115],[63,618],[63,1129],[63,116],[63,710],[63,1144],[63,527],[63,391],[63,720],[63,1132],[63,804],[63,704],[63,1221],[64,262],[64,435],[64,145],[64,183],[64,209],[64,266],[64,263],[64,377],[64,78],[64,606],[64,72],[64,636],[64,66],[64,572],[64,573],[64,581],[64,723],[64,623],[64,269],[64,791],[64,561],[64,67],[64,583],[64,475],[64,130],[64,286],[64,498],[64,94],[64,1035],[64,497],[64,436],[64,1053],[64,834],[64,1069],[64,189],[64,1060],[64,356],[64,710],[64,80],[64,391],[64,720],[64,1170],[64,482],[64,417],[64,1213],[64,724],[65,92],[65,104],[65,179],[65,215],[65,124],[65,285],[65,293],[65,259],[65,135],[65,270],[65,432],[65,467],[65,346],[65,184],[65,518],[65,145],[65,587],[65,72],[65,77],[65,66],[65,427],[65,348],[65,639],[65,100],[65,220],[65,222],[65,224],[65,88],[65,596],[65,103],[65,412],[65,743],[65,269],[65,69],[65,584],[65,392],[65,765],[65,473],[65,519],[65,797],[65,271],[65,379],[65,83],[65,673],[65,306],[65,528],[65,583],[65,219],[65,886],[65,668],[65,917],[65,84],[65,488],[65,947],[65,605],[65,600],[65,686],[65,302],[65,87],[65,94],[65,994],[65,468],[65,362],[65,497],[65,239],[65,252],[65,225],[65,515],[65,217],[65,523],[65,1029],[65,436],[65,1053],[65,604],[65,663],[65,1062],[65,834],[65,1027],[65,514],[65,1069],[65,909],[65,146],[65,74],[65,187],[65,1008],[65,1114],[65,1115],[65,1060],[65,316],[65,538],[65,155],[65,1129],[65,106],[65,356],[65,710],[65,1004],[65,80],[65,1145],[65,1158],[65,391],[65,720],[65,289],[65,532],[65,1017],[65,898],[65,804],[65,119],[65,1201],[65,301],[65,536],[65,182],[65,1209],[65,310],[65,704],[65,125],[65,1221],[65,509],[65,722],[66,354],[66,469],[66,122],[66,70],[66,427],[66,85],[66,240],[66,136],[66,94],[66,72],[66,67],[66,185],[66,581],[66,223],[66,623],[66,69],[66,528],[66,84],[66,488],[66,87],[66,433],[66,216],[66,115],[66,497],[66,1029],[66,436],[66,663],[66,1062],[66,74],[66,189],[66,75],[66,391],[66,720],[66,532],[66,86],[67,92],[67,104],[67,143],[67,215],[67,68],[67,270],[67,467],[67,469],[67,518],[67,145],[67,606],[67,73],[67,72],[67,77],[67,427],[67,100],[67,212],[67,576],[67,88],[67,698],[67,624],[67,214],[67,581],[67,223],[67,95],[67,730],[67,743],[67,269],[67,470],[67,69],[67,519],[67,190],[67,83],[67,640],[67,780],[67,101],[67,118],[67,266],[67,220],[67,86],[67,94],[67,181],[67,71],[67,572],[67,80],[67,75],[67,79],[67,230],[67,81],[67,187],[67,84],[67,123],[67,183],[67,90],[67,174],[67,126],[67,91],[67,142],[67,74],[67,70],[67,124],[67,390],[67,583],[67,721],[67,874],[67,767],[67,761],[67,475],[67,488],[67,87],[67,988],[67,265],[67,994],[67,216],[67,497],[67,217],[67,436],[67,1049],[67,575],[67,1053],[67,604],[67,857],[67,648],[67,1027],[67,146],[67,189],[67,1060],[67,400],[67,538],[67,1129],[67,356],[67,631],[67,385],[67,527],[67,398],[67,391],[67,720],[67,532],[67,482],[67,1171],[67,474],[67,536],[67,1059],[67,1221],[68,104],[68,179],[68,220],[68,278],[68,587],[68,606],[68,73],[68,427],[68,209],[68,573],[68,698],[68,197],[68,743],[68,254],[68,433],[68,90],[68,497],[68,436],[68,575],[68,1053],[68,187],[68,189],[68,1115],[68,336],[68,123],[68,1129],[68,75],[68,391],[68,78],[68,86],[68,1191],[68,574],[68,704],[68,264],[69,92],[69,104],[69,191],[69,215],[69,270],[69,467],[69,469],[69,184],[69,164],[69,518],[69,571],[69,93],[69,587],[69,563],[69,606],[69,628],[69,77],[69,101],[69,427],[69,100],[69,220],[69,491],[69,209],[69,222],[69,224],[69,573],[69,88],[69,698],[69,624],[69,713],[69,472],[69,103],[69,623],[69,470],[69,755],[69,193],[69,436],[69,433],[69,729],[69,407],[69,187],[69,759],[69,577],[69,760],[69,188],[69,572],[69,226],[69,218],[69,474],[69,89],[69,70],[69,183],[69,86],[69,87],[69,94],[69,91],[69,144],[69,216],[69,166],[69,230],[69,264],[69,565],[69,223],[69,498],[69,761],[69,297],[69,231],[69,201],[69,529],[69,481],[69,775],[69,473],[69,519],[69,506],[69,666],[69,271],[69,83],[69,535],[69,478],[69,528],[69,583],[69,656],[69,858],[69,721],[69,876],[69,505],[69,542],[69,668],[69,475],[69,932],[69,941],[69,488],[69,947],[69,605],[69,705],[69,600],[69,441],[69,965],[69,966],[69,409],[69,265],[69,140],[69,323],[69,468],[69,362],[69,359],[69,497],[69,1038],[69,768],[69,515],[69,217],[69,523],[69,1050],[69,1053],[69,604],[69,492],[69,663],[69,857],[69,502],[69,146],[69,644],[69,189],[69,1114],[69,1115],[69,557],[69,1060],[69,400],[69,538],[69,1129],[69,710],[69,1086],[69,80],[69,1144],[69,631],[69,278],[69,833],[69,385],[69,1162],[69,75],[69,527],[69,398],[69,720],[69,532],[69,1075],[69,564],[69,482],[69,804],[69,1088],[69,862],[69,227],[69,1209],[69,704],[70,104],[70,285],[70,270],[70,432],[70,469],[70,122],[70,550],[70,518],[70,93],[70,72],[70,77],[70,662],[70,209],[70,706],[70,394],[70,743],[70,83],[70,760],[70,874],[70,542],[70,668],[70,84],[70,87],[70,105],[70,239],[70,225],[70,217],[70,436],[70,1053],[70,1069],[70,146],[70,74],[70,1115],[70,80],[70,1144],[70,631],[70,278],[70,75],[70,532],[70,1132],[70,91],[71,104],[71,179],[71,285],[71,435],[71,469],[71,587],[71,606],[71,73],[71,222],[71,118],[71,427],[71,730],[71,519],[71,425],[71,254],[71,498],[71,216],[71,497],[71,479],[71,146],[71,189],[71,538],[71,1129],[71,80],[71,75],[71,1166],[71,482],[71,1186],[71,206],[71,474],[71,182],[72,104],[72,215],[72,270],[72,145],[72,587],[72,81],[72,94],[72,230],[72,474],[72,572],[72,185],[72,126],[72,287],[72,356],[72,77],[72,140],[72,91],[72,427],[72,219],[72,247],[72,84],[72,101],[72,390],[72,861],[72,721],[72,874],[72,475],[72,130],[72,433],[72,498],[72,569],[72,217],[72,523],[72,436],[72,1067],[72,189],[72,1115],[72,400],[72,538],[72,80],[72,1159],[72,75],[72,527],[72,86],[72,182],[72,1163],[72,722],[73,179],[73,215],[73,297],[73,270],[73,455],[73,184],[73,587],[73,606],[73,273],[73,187],[73,222],[73,80],[73,558],[73,572],[73,118],[73,468],[73,90],[73,79],[73,185],[73,454],[73,498],[73,214],[73,452],[73,266],[73,534],[73,507],[73,623],[73,150],[73,479],[73,269],[73,81],[73,212],[73,188],[73,631],[73,100],[73,82],[73,632],[73,575],[73,493],[73,377],[73,126],[73,561],[73,453],[73,136],[73,114],[73,107],[73,573],[73,743],[73,495],[73,392],[73,519],[73,379],[73,760],[73,583],[73,869],[73,526],[73,605],[73,996],[73,204],[73,497],[73,515],[73,436],[73,663],[73,514],[73,681],[73,189],[73,1115],[73,1060],[73,1129],[73,710],[73,562],[73,1144],[73,385],[73,527],[73,398],[73,391],[73,1170],[73,482],[73,474],[74,293],[74,518],[74,932],[74,526],[74,314],[74,335],[74,663],[74,625],[74,146],[74,187],[74,100],[74,86],[74,118],[74,76],[74,189],[74,527],[74,720],[74,474],[74,879],[75,467],[75,469],[75,183],[75,587],[75,606],[75,88],[75,624],[75,476],[75,472],[75,723],[75,743],[75,470],[75,473],[75,519],[75,506],[75,666],[75,83],[75,407],[75,721],[75,510],[75,923],[75,475],[75,605],[75,254],[75,433],[75,210],[75,468],[75,217],[75,436],[75,575],[75,663],[75,146],[75,187],[75,511],[75,1115],[75,1129],[75,710],[75,80],[75,101],[75,94],[75,193],[75,231],[75,188],[75,186],[75,1164],[75,81],[75,718],[75,726],[75,89],[75,805],[75,574],[75,182],[75,223],[75,221],[75,100],[75,527],[75,391],[75,720],[75,482],[75,86],[75,474],[75,722],[76,469],[76,184],[76,518],[76,587],[76,628],[76,77],[76,491],[76,209],[76,216],[76,473],[76,79],[76,213],[76,472],[76,519],[76,528],[76,407],[76,721],[76,475],[76,90],[76,588],[76,436],[76,575],[76,663],[76,1062],[76,146],[76,187],[76,710],[76,1086],[76,527],[76,1170],[76,532],[76,1171],[76,1191],[76,536],[76,722],[77,215],[77,184],[77,93],[77,145],[77,606],[77,217],[77,284],[77,100],[77,541],[77,218],[77,418],[77,81],[77,491],[77,523],[77,214],[77,146],[77,86],[77,83],[77,377],[77,118],[77,265],[77,94],[77,635],[77,79],[77,427],[77,220],[77,698],[77,519],[77,583],[77,475],[77,216],[77,408],[77,604],[77,663],[77,1067],[77,189],[77,1115],[77,527],[77,182],[77,264],[78,85],[78,604],[78,710],[78,79],[78,453],[78,266],[78,188],[78,264],[78,1166],[78,605],[78,1171],[79,215],[79,270],[79,518],[79,606],[79,662],[79,212],[79,209],[79,573],[79,214],[79,274],[79,730],[79,661],[79,584],[79,528],[79,668],[79,512],[79,498],[79,90],[79,100],[79,527],[79,497],[79,436],[79,604],[79,146],[79,187],[79,1115],[79,557],[79,538],[79,710],[79,1144],[79,86],[79,182],[80,104],[80,122],[80,518],[80,209],[80,88],[80,698],[80,742],[80,743],[80,766],[80,797],[80,190],[80,760],[80,583],[80,874],[80,668],[80,84],[80,761],[80,475],[80,90],[80,468],[80,497],[80,217],[80,436],[80,1051],[80,575],[80,1027],[80,146],[80,187],[80,1115],[80,1060],[80,316],[80,538],[80,349],[80,123],[80,1129],[80,562],[80,222],[80,116],[80,110],[80,359],[80,118],[80,188],[80,853],[80,869],[80,221],[80,266],[80,805],[80,183],[80,427],[80,81],[80,212],[80,82],[80,89],[80,699],[80,107],[80,136],[80,137],[80,91],[80,165],[80,369],[80,446],[80,1144],[80,631],[80,391],[80,1170],[80,482],[80,139],[80,86],[80,474],[80,182],[81,92],[81,207],[81,268],[81,270],[81,145],[81,587],[81,606],[81,639],[81,220],[81,221],[81,88],[81,689],[81,655],[81,581],[81,596],[81,454],[81,103],[81,504],[81,661],[81,584],[81,481],[81,797],[81,190],[81,83],[81,626],[81,632],[81,583],[81,656],[81,475],[81,605],[81,705],[81,498],[81,90],[81,409],[81,94],[81,497],[81,569],[81,217],[81,436],[81,759],[81,1027],[81,146],[81,187],[81,422],[81,189],[81,393],[81,1115],[81,557],[81,1060],[81,400],[81,1001],[81,710],[81,1020],[81,385],[81,527],[81,398],[81,391],[81,1170],[81,193],[81,532],[81,482],[81,206],[81,86],[81,574],[81,474],[81,182],[81,592],[81,417],[81,1213],[82,179],[82,145],[82,628],[82,427],[82,439],[82,576],[82,100],[82,187],[82,101],[82,467],[82,743],[82,495],[82,632],[82,90],[82,1027],[82,1129],[82,710],[82,474],[83,92],[83,467],[83,93],[83,587],[83,88],[83,457],[83,98],[83,223],[83,269],[83,392],[83,663],[83,436],[83,220],[83,221],[83,187],[83,100],[83,468],[83,190],[83,94],[83,583],[83,505],[83,475],[83,500],[83,497],[83,575],[83,604],[83,644],[83,565],[83,1129],[83,527],[83,398],[83,536],[83,577],[84,104],[84,215],[84,124],[84,112],[84,288],[84,298],[84,259],[84,135],[84,365],[84,247],[84,354],[84,242],[84,248],[84,234],[84,459],[84,117],[84,346],[84,122],[84,159],[84,164],[84,93],[84,582],[84,586],[84,593],[84,612],[84,434],[84,100],[84,218],[84,617],[84,650],[84,694],[84,697],[84,698],[84,581],[84,457],[84,223],[84,103],[84,163],[84,742],[84,750],[84,236],[84,333],[84,347],[84,775],[84,154],[84,783],[84,798],[84,190],[84,379],[84,626],[84,652],[84,852],[84,855],[84,138],[84,245],[84,856],[84,168],[84,874],[84,767],[84,880],[84,447],[84,358],[84,548],[84,91],[84,158],[84,116],[84,130],[84,107],[84,427],[84,108],[84,237],[84,85],[84,110],[84,133],[84,111],[84,161],[84,136],[84,115],[84,171],[84,356],[84,250],[84,86],[84,87],[84,119],[84,114],[84,174],[84,254],[84,123],[84,125],[84,239],[84,128],[84,761],[84,922],[84,929],[84,933],[84,331],[84,943],[84,951],[84,165],[84,302],[84,425],[84,371],[84,132],[84,109],[84,351],[84,162],[84,127],[84,105],[84,140],[84,1022],[84,816],[84,1026],[84,323],[84,324],[84,313],[84,113],[84,233],[84,256],[84,335],[84,235],[84,322],[84,1039],[84,355],[84,292],[84,252],[84,160],[84,674],[84,257],[84,217],[84,1029],[84,244],[84,436],[84,258],[84,341],[84,604],[84,648],[84,1028],[84,368],[84,255],[84,187],[84,312],[84,1002],[84,462],[84,1008],[84,422],[84,364],[84,1113],[84,565],[84,1115],[84,321],[84,618],[84,169],[84,1120],[84,842],[84,316],[84,349],[84,155],[84,167],[84,1004],[84,1140],[84,421],[84,670],[84,398],[84,391],[84,904],[84,332],[84,972],[84,720],[84,1168],[84,290],[84,448],[84,289],[84,1017],[84,1011],[84,482],[84,1042],[84,1132],[84,139],[84,804],[84,375],[84,737],[84,991],[84,1198],[84,549],[84,911],[84,474],[84,1199],[84,981],[84,1202],[84,310],[84,374],[84,1073],[84,1213],[85,104],[85,215],[85,124],[85,112],[85,259],[85,247],[85,354],[85,242],[85,248],[85,117],[85,383],[85,122],[85,639],[85,617],[85,650],[85,694],[85,707],[85,741],[85,742],[85,333],[85,154],[85,171],[85,779],[85,892],[85,548],[85,922],[85,130],[85,87],[85,425],[85,371],[85,132],[85,109],[85,351],[85,161],[85,306],[85,235],[85,136],[85,324],[85,106],[85,107],[85,234],[85,110],[85,133],[85,91],[85,114],[85,138],[85,116],[85,140],[85,356],[85,332],[85,86],[85,190],[85,282],[85,245],[85,454],[85,323],[85,998],[85,128],[85,105],[85,108],[85,115],[85,313],[85,113],[85,256],[85,335],[85,355],[85,292],[85,160],[85,674],[85,257],[85,250],[85,569],[85,1029],[85,436],[85,969],[85,141],[85,648],[85,1005],[85,255],[85,1002],[85,462],[85,422],[85,311],[85,1115],[85,167],[85,1004],[85,1140],[85,421],[85,1144],[85,670],[85,972],[85,720],[85,448],[85,367],[85,1132],[85,139],[85,182],[85,125],[86,215],[86,270],[86,184],[86,183],[86,582],[86,639],[86,220],[86,212],[86,209],[86,221],[86,88],[86,689],[86,655],[86,214],[86,581],[86,223],[86,596],[86,274],[86,730],[86,661],[86,740],[86,521],[86,269],[86,519],[86,791],[86,190],[86,494],[86,827],[86,583],[86,656],[86,721],[86,767],[86,505],[86,533],[86,475],[86,488],[86,512],[86,87],[86,286],[86,433],[86,210],[86,265],[86,1023],[86,249],[86,1038],[86,569],[86,217],[86,436],[86,575],[86,604],[86,834],[86,146],[86,187],[86,189],[86,1115],[86,557],[86,530],[86,400],[86,397],[86,538],[86,1129],[86,356],[86,1004],[86,385],[86,527],[86,391],[86,972],[86,720],[86,193],[86,206],[86,100],[86,172],[86,180],[86,474],[86,89],[86,222],[86,118],[86,94],[86,266],[86,136],[86,91],[86,188],[86,182],[86,704],[87,582],[87,596],[87,190],[87,542],[87,931],[87,94],[87,91],[87,240],[87,188],[87,136],[87,116],[87,124],[87,193],[87,183],[87,225],[87,100],[87,706],[87,569],[87,648],[87,804],[88,92],[88,215],[88,184],[88,93],[88,223],[88,209],[88,686],[88,211],[88,218],[88,475],[88,433],[88,220],[88,89],[88,187],[88,100],[88,146],[88,172],[88,190],[88,152],[88,231],[88,278],[88,95],[88,226],[88,91],[88,723],[88,661],[88,659],[88,605],[88,436],[88,575],[88,189],[88,710],[88,532],[88,1166],[88,482],[88,724],[88,273],[89,215],[89,573],[89,698],[89,743],[89,470],[89,495],[89,190],[89,583],[89,475],[89,433],[89,90],[89,204],[89,217],[89,436],[89,604],[89,1027],[89,397],[89,1129],[89,710],[89,729],[89,1144],[89,1171],[89,206],[89,474],[89,222],[90,401],[90,587],[90,188],[90,118],[90,229],[90,266],[90,222],[91,104],[91,137],[91,215],[91,124],[91,112],[91,261],[91,166],[91,288],[91,298],[91,259],[91,135],[91,365],[91,354],[91,242],[91,426],[91,248],[91,234],[91,337],[91,117],[91,469],[91,164],[91,550],[91,361],[91,582],[91,586],[91,593],[91,595],[91,607],[91,315],[91,434],[91,619],[91,299],[91,649],[91,639],[91,567],[91,677],[91,617],[91,687],[91,650],[91,692],[91,694],[91,696],[91,697],[91,707],[91,472],[91,163],[91,741],[91,742],[91,744],[91,745],[91,750],[91,752],[91,236],[91,333],[91,347],[91,329],[91,775],[91,154],[91,777],[91,783],[91,546],[91,171],[91,795],[91,798],[91,800],[91,643],[91,379],[91,817],[91,366],[91,626],[91,652],[91,827],[91,121],[91,839],[91,779],[91,855],[91,138],[91,245],[91,856],[91,407],[91,168],[91,868],[91,874],[91,880],[91,447],[91,893],[91,358],[91,810],[91,548],[91,921],[91,922],[91,475],[91,931],[91,902],[91,939],[91,940],[91,943],[91,403],[91,165],[91,465],[91,958],[91,302],[91,425],[91,371],[91,254],[91,409],[91,132],[91,109],[91,351],[91,162],[91,161],[91,128],[91,105],[91,140],[91,620],[91,1024],[91,323],[91,108],[91,324],[91,249],[91,115],[91,178],[91,359],[91,313],[91,113],[91,233],[91,256],[91,281],[91,335],[91,235],[91,322],[91,345],[91,253],[91,308],[91,355],[91,239],[91,133],[91,292],[91,252],[91,160],[91,674],[91,257],[91,225],[91,250],[91,569],[91,1029],[91,1048],[91,408],[91,244],[91,436],[91,258],[91,177],[91,341],[91,604],[91,969],[91,363],[91,102],[91,915],[91,153],[91,1034],[91,648],[91,625],[91,368],[91,158],[91,990],[91,1069],[91,1070],[91,255],[91,187],[91,1002],[91,462],[91,1080],[91,1008],[91,422],[91,311],[91,1113],[91,565],[91,1115],[91,1117],[91,321],[91,107],[91,169],[91,900],[91,771],[91,400],[91,360],[91,1040],[91,842],[91,316],[91,336],[91,349],[91,156],[91,123],[91,155],[91,106],[91,116],[91,167],[91,356],[91,693],[91,1140],[91,421],[91,1144],[91,353],[91,758],[91,175],[91,332],[91,972],[91,720],[91,1168],[91,290],[91,448],[91,289],[91,532],[91,1017],[91,110],[91,367],[91,1178],[91,1011],[91,136],[91,1183],[91,898],[91,1186],[91,1132],[91,1187],[91,139],[91,804],[91,905],[91,1191],[91,1192],[91,375],[91,737],[91,911],[91,474],[91,1199],[91,241],[91,981],[91,1202],[91,1207],[91,536],[91,1209],[91,310],[91,374],[91,174],[91,125],[91,822],[91,1073],[91,860],[91,1213],[91,908],[91,393],[91,238],[91,282],[91,378],[92,93],[92,94],[92,95],[92,96],[92,97],[92,98],[92,99],[92,100],[92,101],[92,102],[92,103],[92,297],[92,467],[92,626],[92,390],[92,929],[92,686],[92,286],[92,492],[92,146],[93,143],[93,215],[93,297],[93,435],[93,467],[93,95],[93,433],[93,218],[93,393],[93,209],[93,226],[93,187],[93,634],[93,101],[93,220],[93,327],[93,624],[93,394],[93,98],[93,476],[93,97],[93,392],[93,775],[93,519],[93,812],[93,838],[93,840],[93,721],[93,874],[93,767],[93,887],[93,418],[93,668],[93,475],[93,705],[93,600],[93,500],[93,988],[93,216],[93,217],[93,523],[93,436],[93,712],[93,1053],[93,604],[93,889],[93,663],[93,857],[93,1067],[93,1027],[93,146],[93,565],[93,1060],[93,1129],[93,562],[93,1144],[93,631],[93,1162],[93,527],[93,398],[93,1191],[93,574],[93,536],[93,1059],[93,273],[94,104],[94,179],[94,215],[94,270],[94,467],[94,469],[94,518],[94,571],[94,145],[94,587],[94,118],[94,628],[94,634],[94,101],[94,427],[94,100],[94,209],[94,222],[94,576],[94,230],[94,476],[94,269],[94,392],[94,473],[94,519],[94,666],[94,721],[94,542],[94,668],[94,761],[94,475],[94,488],[94,187],[94,110],[94,746],[94,126],[94,572],[94,105],[94,140],[94,239],[94,217],[94,436],[94,575],[94,604],[94,969],[94,663],[94,228],[94,502],[94,1067],[94,146],[94,189],[94,557],[94,1060],[94,400],[94,1129],[94,527],[94,532],[94,474],[94,536],[94,862],[94,417],[94,1213],[94,577],[95,266],[95,218],[95,201],[95,187],[95,100],[95,705],[95,433],[95,604],[96,143],[96,365],[96,270],[96,401],[96,518],[96,145],[96,491],[96,274],[96,756],[96,192],[96,519],[96,271],[96,535],[96,721],[96,922],[96,475],[96,286],[96,498],[96,988],[96,1032],[96,408],[96,857],[96,1067],[96,255],[96,146],[96,422],[96,321],[96,710],[96,527],[96,720],[96,480],[96,794],[96,862],[97,668],[97,204],[97,436],[97,604],[98,587],[98,638],[98,731],[98,755],[98,190],[98,710],[98,474],[98,227],[99,401],[99,267],[99,472],[99,743],[99,269],[99,196],[99,473],[99,192],[99,519],[99,807],[99,271],[99,535],[99,673],[99,632],[99,407],[99,721],[99,510],[99,512],[99,204],[99,1038],[99,408],[99,889],[99,187],[99,565],[99,356],[99,710],[99,1086],[99,1020],[99,415],[99,206],[99,182],[99,509],[100,143],[100,179],[100,191],[100,215],[100,260],[100,268],[100,293],[100,270],[100,401],[100,455],[100,467],[100,469],[100,184],[100,539],[100,395],[100,518],[100,531],[100,571],[100,587],[100,118],[100,606],[100,622],[100,628],[100,636],[100,639],[100,530],[100,220],[100,230],[100,186],[100,227],[100,101],[100,216],[100,187],[100,476],[100,214],[100,107],[100,110],[100,102],[100,655],[100,576],[100,209],[100,589],[100,657],[100,218],[100,221],[100,573],[100,698],[100,624],[100,213],[100,197],[100,223],[100,596],[100,472],[100,103],[100,723],[100,274],[100,623],[100,730],[100,731],[100,521],[100,269],[100,470],[100,603],[100,508],[100,584],[100,392],[100,766],[100,519],[100,789],[100,506],[100,379],[100,640],[100,399],[100,540],[100,535],[100,840],[100,528],[100,583],[100,407],[100,861],[100,869],[100,767],[100,505],[100,418],[100,542],[100,668],[100,925],[100,475],[100,599],[100,932],[100,871],[100,941],[100,488],[100,947],[100,201],[100,705],[100,686],[100,441],[100,965],[100,498],[100,988],[100,265],[100,588],[100,140],[100,468],[100,362],[100,497],[100,1038],[100,355],[100,202],[100,225],[100,768],[100,515],[100,217],[100,523],[100,408],[100,436],[100,575],[100,1053],[100,558],[100,604],[100,492],[100,889],[100,203],[100,663],[100,228],[100,915],[100,857],[100,1067],[100,834],[100,514],[100,146],[100,644],[100,402],[100,189],[100,231],[100,565],[100,557],[100,1060],[100,538],[100,263],[100,1129],[100,710],[100,1145],[100,631],[100,226],[100,833],[100,385],[100,527],[100,398],[100,391],[100,720],[100,415],[100,482],[100,139],[100,1190],[100,1191],[100,480],[100,485],[100,387],[100,1195],[100,534],[100,474],[100,1201],[100,301],[100,536],[100,182],[100,310],[100,1210],[100,704],[100,1059],[100,835],[100,1163],[100,879],[100,273],[101,298],[101,628],[101,644],[101,110],[101,187],[101,218],[101,436],[101,222],[101,216],[101,220],[101,624],[101,103],[101,271],[101,488],[101,130],[101,498],[101,604],[101,492],[101,565],[101,400],[101,1086],[101,301],[101,536],[101,577],[102,215],[102,285],[102,259],[102,243],[102,435],[102,452],[102,455],[102,551],[102,272],[102,188],[102,377],[102,612],[102,677],[102,691],[102,352],[102,495],[102,659],[102,795],[102,721],[102,875],[102,887],[102,334],[102,924],[102,902],[102,331],[102,165],[102,425],[102,441],[102,967],[102,140],[102,178],[102,1032],[102,1036],[102,906],[102,113],[102,256],[102,345],[102,253],[102,308],[102,292],[102,712],[102,341],[102,246],[102,107],[102,110],[102,857],[102,153],[102,466],[102,1067],[102,990],[102,255],[102,909],[102,1008],[102,422],[102,311],[102,1114],[102,342],[102,1116],[102,520],[102,900],[102,771],[102,360],[102,717],[102,538],[102,167],[102,1130],[102,421],[102,1033],[102,758],[102,1165],[102,532],[102,1178],[102,406],[102,1011],[102,206],[102,1189],[102,1191],[102,1194],[102,549],[102,241],[102,503],[102,1074],[102,1203],[102,1206],[102,536],[102,174],[102,125],[102,1204],[102,380],[103,455],[103,722],[103,717],[103,263],[103,144],[103,206],[103,557],[103,222],[103,730],[103,519],[103,528],[103,475],[103,705],[103,515],[103,558],[103,565],[103,729],[103,527],[103,589],[104,105],[104,106],[104,107],[104,108],[104,109],[104,110],[104,111],[104,112],[104,113],[104,114],[104,115],[104,116],[104,117],[104,118],[104,119],[104,120],[104,121],[104,122],[104,123],[104,124],[104,125],[104,126],[104,127],[104,128],[104,129],[104,130],[104,131],[104,132],[104,133],[104,134],[104,135],[104,136],[104,137],[104,138],[104,139],[104,140],[104,141],[104,142],[104,741],[104,323],[104,255],[104,1008],[104,316],[104,167],[105,112],[105,135],[105,117],[105,122],[105,567],[105,694],[105,741],[105,859],[105,863],[105,334],[105,548],[105,351],[105,129],[105,346],[105,128],[105,106],[105,107],[105,110],[105,167],[105,116],[105,119],[105,124],[105,125],[105,142],[105,137],[105,108],[105,134],[105,140],[105,324],[105,123],[105,131],[105,115],[105,133],[105,354],[105,113],[105,126],[105,242],[105,237],[105,138],[105,443],[105,248],[105,174],[105,345],[105,173],[105,141],[105,111],[105,249],[105,328],[105,247],[105,120],[105,251],[105,256],[105,257],[105,1001],[105,352],[105,139],[105,243],[105,165],[105,553],[105,647],[105,1002],[105,121],[105,253],[105,289],[105,244],[105,318],[105,1003],[105,371],[105,320],[105,1004],[105,674],[105,304],[105,301],[105,929],[105,1005],[105,314],[105,323],[105,445],[105,292],[105,1006],[105,1007],[105,386],[105,1008],[105,779],[105,176],[105,322],[105,778],[105,815],[105,790],[105,329],[105,1009],[105,972],[105,922],[105,1010],[105,321],[105,1011],[105,276],[105,544],[105,842],[105,296],[105,1012],[105,940],[105,864],[105,361],[105,368],[105,1013],[105,325],[105,444],[105,1014],[105,1015],[105,360],[105,372],[105,169],[105,1016],[105,422],[105,915],[105,912],[105,357],[105,350],[105,381],[105,1017],[105,1018],[105,342],[105,239],[105,175],[105,159],[105,893],[105,362],[105,235],[105,127],[105,1029],[105,255],[105,316],[105,999],[105,114],[105,421],[106,112],[106,135],[106,242],[106,117],[106,346],[106,122],[106,343],[106,650],[106,694],[106,696],[106,741],[106,333],[106,154],[106,379],[106,366],[106,447],[106,893],[106,548],[106,302],[106,425],[106,371],[106,109],[106,134],[106,351],[106,161],[106,127],[106,323],[106,108],[106,314],[106,362],[106,178],[106,256],[106,335],[106,322],[106,345],[106,355],[106,133],[106,674],[106,250],[106,1029],[106,244],[106,153],[106,255],[106,311],[106,107],[106,169],[106,316],[106,129],[106,119],[106,245],[106,131],[106,116],[106,125],[106,124],[106,241],[106,174],[106,130],[106,136],[106,249],[106,141],[106,167],[106,139],[106,120],[106,123],[106,121],[106,252],[106,251],[106,156],[106,142],[106,320],[106,1128],[106,1140],[106,421],[106,1159],[106,972],[106,448],[106,1011],[106,375],[106,549],[107,137],[107,215],[107,124],[107,112],[107,166],[107,288],[107,298],[107,259],[107,135],[107,365],[107,354],[107,242],[107,426],[107,248],[107,428],[107,243],[107,337],[107,452],[107,117],[107,464],[107,122],[107,159],[107,164],[107,343],[107,550],[107,361],[107,582],[107,586],[107,607],[107,434],[107,299],[107,629],[107,176],[107,642],[107,348],[107,649],[107,639],[107,669],[107,567],[107,650],[107,694],[107,696],[107,697],[107,708],[107,457],[107,711],[107,472],[107,163],[107,352],[107,741],[107,750],[107,333],[107,347],[107,329],[107,775],[107,154],[107,777],[107,251],[107,783],[107,546],[107,786],[107,171],[107,791],[107,795],[107,802],[107,494],[107,379],[107,817],[107,366],[107,626],[107,652],[107,827],[107,399],[107,121],[107,839],[107,844],[107,779],[107,855],[107,138],[107,245],[107,856],[107,583],[107,168],[107,880],[107,445],[107,447],[107,893],[107,358],[107,548],[107,921],[107,142],[107,922],[107,902],[107,940],[107,331],[107,943],[107,951],[107,130],[107,165],[107,465],[107,302],[107,425],[107,913],[107,371],[107,254],[107,982],[107,987],[107,903],[107,132],[107,109],[107,134],[107,351],[107,162],[107,161],[107,128],[107,444],[107,127],[107,140],[107,620],[107,323],[107,108],[107,314],[107,324],[107,115],[107,362],[107,178],[107,359],[107,313],[107,113],[107,233],[107,256],[107,335],[107,322],[107,253],[107,308],[107,355],[107,239],[107,133],[107,292],[107,252],[107,160],[107,674],[107,257],[107,225],[107,250],[107,244],[107,436],[107,258],[107,695],[107,372],[107,969],[107,915],[107,153],[107,648],[107,594],[107,1005],[107,625],[107,990],[107,255],[107,312],[107,1002],[107,462],[107,1008],[107,422],[107,291],[107,364],[107,311],[107,1113],[107,1115],[107,1117],[107,321],[107,618],[107,131],[107,241],[107,232],[107,119],[107,116],[107,110],[107,125],[107,123],[107,111],[107,114],[107,139],[107,129],[107,175],[107,120],[107,167],[107,346],[107,289],[107,247],[107,304],[107,156],[107,328],[107,249],[107,345],[107,141],[107,174],[107,317],[107,169],[107,900],[107,771],[107,400],[107,360],[107,842],[107,316],[107,336],[107,349],[107,373],[107,155],[107,356],[107,1004],[107,1140],[107,421],[107,1144],[107,1147],[107,670],[107,319],[107,758],[107,385],[107,391],[107,332],[107,972],[107,1168],[107,350],[107,290],[107,448],[107,532],[107,1173],[107,367],[107,614],[107,344],[107,1178],[107,1011],[107,136],[107,898],[107,1042],[107,1186],[107,1132],[107,1189],[107,905],[107,1191],[107,375],[107,339],[107,991],[107,911],[107,474],[107,981],[107,301],[107,1074],[107,1207],[107,310],[107,374],[107,1073],[107,860],[107,378],[107,1220],[108,112],[108,259],[108,135],[108,248],[108,337],[108,117],[108,346],[108,122],[108,434],[108,694],[108,696],[108,741],[108,333],[108,306],[108,331],[108,425],[108,254],[108,132],[108,109],[108,134],[108,351],[108,128],[108,127],[108,237],[108,110],[108,133],[108,113],[108,167],[108,114],[108,138],[108,116],[108,119],[108,120],[108,242],[108,174],[108,123],[108,125],[108,142],[108,249],[108,347],[108,153],[108,130],[108,245],[108,1028],[108,277],[108,289],[108,256],[108,324],[108,111],[108,252],[108,129],[108,311],[108,165],[108,445],[108,251],[108,354],[108,136],[108,304],[108,335],[108,139],[108,250],[108,173],[108,140],[108,448],[108,314],[108,141],[108,463],[108,1029],[108,121],[108,244],[108,291],[108,239],[108,225],[108,362],[108,124],[108,257],[108,255],[108,1002],[108,316],[108,1143],[109,112],[109,259],[109,135],[109,365],[109,242],[109,248],[109,337],[109,117],[109,299],[109,348],[109,696],[109,741],[109,333],[109,347],[109,329],[109,154],[109,379],[109,366],[109,168],[109,447],[109,548],[109,331],[109,130],[109,302],[109,425],[109,254],[109,110],[109,289],[109,139],[109,140],[109,324],[109,323],[109,178],[109,241],[109,131],[109,282],[109,247],[109,153],[109,245],[109,328],[109,277],[109,156],[109,157],[109,349],[109,288],[109,280],[109,614],[109,939],[109,237],[109,162],[109,133],[109,334],[109,175],[109,111],[109,321],[109,301],[109,990],[109,252],[109,134],[109,136],[109,165],[109,445],[109,352],[109,167],[109,251],[109,336],[109,138],[109,355],[109,335],[109,171],[109,991],[109,250],[109,419],[109,116],[109,256],[109,448],[109,119],[109,305],[109,359],[109,120],[109,314],[109,360],[109,141],[109,174],[109,232],[109,121],[109,122],[109,123],[109,124],[109,125],[109,549],[109,142],[109,244],[109,249],[109,127],[109,992],[109,129],[109,225],[109,362],[109,922],[109,345],[109,128],[109,318],[109,322],[109,292],[109,674],[109,257],[109,594],[109,255],[109,1002],[109,462],[109,1008],[109,1113],[109,618],[109,316],[109,421],[109,1159],[109,972],[110,148],[110,137],[110,215],[110,124],[110,112],[110,261],[110,166],[110,288],[110,295],[110,298],[110,259],[110,135],[110,365],[110,247],[110,270],[110,354],[110,242],[110,426],[110,248],[110,234],[110,243],[110,337],[110,459],[110,117],[110,464],[110,122],[110,159],[110,164],[110,272],[110,361],[110,568],[110,582],[110,586],[110,595],[110,118],[110,607],[110,611],[110,612],[110,315],[110,434],[110,619],[110,299],[110,629],[110,176],[110,634],[110,642],[110,646],[110,348],[110,649],[110,662],[110,449],[110,222],[110,567],[110,677],[110,617],[110,685],[110,687],[110,650],[110,692],[110,694],[110,696],[110,697],[110,706],[110,707],[110,457],[110,711],[110,714],[110,223],[110,163],[110,352],[110,741],[110,750],[110,752],[110,236],[110,333],[110,347],[110,329],[110,775],[110,154],[110,777],[110,783],[110,546],[110,786],[110,171],[110,791],[110,795],[110,797],[110,802],[110,494],[110,811],[110,379],[110,817],[110,818],[110,366],[110,626],[110,652],[110,826],[110,827],[110,121],[110,839],[110,846],[110,528],[110,855],[110,138],[110,856],[110,859],[110,168],[110,866],[110,868],[110,873],[110,875],[110,876],[110,880],[110,447],[110,892],[110,893],[110,897],[110,358],[110,461],[110,810],[110,548],[110,921],[110,922],[110,933],[110,902],[110,939],[110,331],[110,943],[110,950],[110,951],[110,403],[110,130],[110,600],[110,165],[110,465],[110,302],[110,309],[110,425],[110,441],[110,967],[110,913],[110,968],[110,371],[110,254],[110,976],[110,409],[110,903],[110,129],[110,132],[110,134],[110,351],[110,376],[110,162],[110,161],[110,128],[110,444],[110,127],[110,140],[110,620],[110,816],[110,1026],[110,323],[110,178],[110,1032],[110,359],[110,906],[110,313],[110,113],[110,233],[110,256],[110,281],[110,335],[110,235],[110,322],[110,345],[110,253],[110,308],[110,355],[110,239],[110,133],[110,292],[110,252],[110,160],[110,674],[110,257],[110,225],[110,250],[110,998],[110,1029],[110,244],[110,258],[110,341],[110,460],[110,969],[110,915],[110,153],[110,1034],[110,141],[110,648],[110,1005],[110,276],[110,625],[110,1067],[110,1028],[110,368],[110,990],[110,1070],[110,255],[110,909],[110,312],[110,1002],[110,462],[110,1080],[110,1008],[110,914],[110,422],[110,291],[110,311],[110,1113],[110,1114],[110,565],[110,1115],[110,321],[110,157],[110,618],[110,900],[110,771],[110,400],[110,360],[110,1121],[110,1040],[110,842],[110,316],[110,336],[110,349],[110,373],[110,1124],[110,156],[110,123],[110,155],[110,116],[110,167],[110,356],[110,693],[110,1140],[110,421],[110,320],[110,1144],[110,631],[110,1147],[110,1155],[110,670],[110,319],[110,758],[110,1165],[110,916],[110,175],[110,904],[110,332],[110,972],[110,1168],[110,350],[110,290],[110,448],[110,289],[110,532],[110,1173],[110,820],[110,1017],[110,362],[110,328],[110,131],[110,282],[110,111],[110,114],[110,139],[110,115],[110,446],[110,172],[110,119],[110,174],[110,125],[110,142],[110,170],[110,232],[110,367],[110,614],[110,703],[110,344],[110,1075],[110,633],[110,1178],[110,406],[110,1011],[110,482],[110,136],[110,1184],[110,898],[110,1042],[110,1186],[110,1132],[110,1187],[110,1189],[110,905],[110,1191],[110,1192],[110,375],[110,737],[110,1193],[110,549],[110,911],[110,474],[110,1199],[110,241],[110,981],[110,301],[110,1074],[110,1203],[110,1206],[110,1207],[110,536],[110,310],[110,822],[110,1073],[110,860],[110,1213],[110,424],[110,1204],[110,637],[111,112],[111,298],[111,135],[111,242],[111,248],[111,117],[111,122],[111,361],[111,434],[111,299],[111,348],[111,617],[111,650],[111,696],[111,741],[111,333],[111,329],[111,778],[111,821],[111,548],[111,425],[111,161],[111,444],[111,246],[111,232],[111,123],[111,131],[111,243],[111,282],[111,419],[111,241],[111,142],[111,172],[111,140],[111,323],[111,113],[111,120],[111,256],[111,322],[111,345],[111,253],[111,133],[111,292],[111,160],[111,250],[111,258],[111,1061],[111,1041],[111,255],[111,321],[111,169],[111,842],[111,316],[111,167],[111,421],[111,139],[111,374],[111,174],[111,125],[111,860],[112,137],[112,124],[112,241],[112,232],[112,131],[112,242],[112,243],[112,130],[112,123],[112,162],[112,121],[112,244],[112,117],[112,116],[112,125],[112,175],[112,128],[112,165],[112,245],[112,140],[112,142],[112,141],[112,119],[112,167],[112,246],[112,135],[112,136],[112,138],[112,247],[112,237],[112,133],[112,113],[112,120],[112,174],[112,248],[112,249],[112,250],[112,169],[112,251],[112,171],[112,252],[112,122],[112,139],[112,253],[112,254],[112,255],[112,132],[112,256],[112,257],[112,114],[112,258],[112,259],[112,127],[112,288],[112,298],[112,426],[112,337],[112,346],[112,164],[112,361],[112,299],[112,669],[112,696],[112,457],[112,163],[112,741],[112,750],[112,333],[112,347],[112,329],[112,775],[112,154],[112,652],[112,827],[112,306],[112,779],[112,880],[112,447],[112,892],[112,548],[112,331],[112,941],[112,302],[112,129],[112,134],[112,323],[112,178],[112,359],[112,313],[112,233],[112,335],[112,322],[112,345],[112,355],[112,239],[112,292],[112,160],[112,674],[112,460],[112,1005],[112,368],[112,990],[112,1002],[112,462],[112,1008],[112,422],[112,291],[112,311],[112,321],[112,618],[112,360],[112,842],[112,316],[112,349],[112,373],[112,999],[112,1140],[112,1144],[112,391],[112,972],[112,448],[112,375],[112,339],[112,549],[112,310],[112,860],[113,135],[113,247],[113,248],[113,117],[113,122],[113,361],[113,723],[113,163],[113,741],[113,154],[113,652],[113,779],[113,358],[113,254],[113,128],[113,127],[113,277],[113,153],[113,258],[113,130],[113,131],[113,446],[113,124],[113,252],[113,167],[113,136],[113,114],[113,137],[113,138],[113,335],[113,115],[113,116],[113,173],[113,119],[113,120],[113,141],[113,174],[113,123],[113,125],[113,249],[113,256],[113,292],[113,160],[113,436],[113,255],[113,321],[113,316],[113,356],[113,421],[113,1144],[113,972],[113,193],[113,182],[114,137],[114,124],[114,288],[114,298],[114,135],[114,247],[114,248],[114,117],[114,649],[114,617],[114,694],[114,698],[114,741],[114,742],[114,333],[114,347],[114,154],[114,777],[114,546],[114,779],[114,168],[114,922],[114,928],[114,165],[114,302],[114,254],[114,132],[114,351],[114,161],[114,128],[114,127],[114,140],[114,323],[114,233],[114,256],[114,335],[114,322],[114,355],[114,133],[114,292],[114,252],[114,160],[114,1029],[114,258],[114,648],[114,1070],[114,255],[114,1002],[114,462],[114,422],[114,311],[114,1115],[114,321],[114,842],[114,316],[114,116],[114,167],[114,356],[114,1004],[114,1140],[114,130],[114,245],[114,421],[114,1144],[114,670],[114,332],[114,972],[114,448],[114,289],[114,136],[114,139],[114,1191],[114,182],[114,174],[114,125],[114,860],[115,137],[115,124],[115,135],[115,248],[115,117],[115,122],[115,434],[115,628],[115,650],[115,163],[115,741],[115,333],[115,154],[115,777],[115,779],[115,548],[115,161],[115,127],[115,158],[115,1030],[115,155],[115,292],[115,123],[115,160],[115,238],[115,162],[115,172],[115,446],[115,170],[115,250],[115,335],[115,253],[115,133],[115,674],[115,257],[115,217],[115,1005],[115,255],[115,1115],[115,321],[115,316],[115,116],[115,167],[115,332],[115,972],[115,139],[115,273],[116,137],[116,124],[116,166],[116,288],[116,259],[116,135],[116,247],[116,242],[116,248],[116,117],[116,383],[116,346],[116,122],[116,164],[116,361],[116,593],[116,642],[116,348],[116,649],[116,650],[116,694],[116,708],[116,163],[116,741],[116,333],[116,584],[116,329],[116,154],[116,546],[116,171],[116,379],[116,652],[116,836],[116,779],[116,855],[116,880],[116,447],[116,893],[116,810],[116,548],[116,331],[116,951],[116,465],[116,302],[116,425],[116,371],[116,254],[116,317],[116,132],[116,351],[116,161],[116,128],[116,127],[116,140],[116,1022],[116,323],[116,314],[116,324],[116,362],[116,313],[116,233],[116,256],[116,335],[116,253],[116,355],[116,133],[116,292],[116,252],[116,160],[116,674],[116,258],[116,695],[116,372],[116,648],[116,1005],[116,368],[116,255],[116,312],[116,1002],[116,462],[116,311],[116,1113],[116,321],[116,618],[116,557],[116,842],[116,336],[116,123],[116,282],[116,130],[116,280],[116,235],[116,129],[116,172],[116,125],[116,142],[116,237],[116,251],[116,139],[116,119],[116,120],[116,157],[116,138],[116,225],[116,439],[116,167],[116,356],[116,1140],[116,391],[116,972],[116,1168],[116,448],[116,289],[116,344],[116,1011],[116,1132],[116,375],[116,991],[116,549],[116,374],[116,174],[116,860],[117,137],[117,166],[117,365],[117,247],[117,165],[117,233],[117,142],[117,419],[117,369],[117,163],[117,460],[117,161],[117,160],[117,257],[117,258],[117,130],[117,245],[117,156],[117,384],[117,157],[117,155],[117,237],[117,131],[117,235],[117,253],[117,256],[117,164],[117,421],[117,282],[117,168],[117,167],[117,136],[117,323],[117,304],[117,138],[117,250],[117,139],[117,461],[117,446],[117,119],[117,277],[117,248],[117,120],[117,462],[117,141],[117,463],[117,174],[117,254],[117,175],[117,121],[117,123],[117,124],[117,125],[117,292],[117,244],[117,127],[117,239],[117,364],[117,128],[117,225],[117,249],[117,291],[117,318],[117,434],[117,741],[117,754],[117,154],[117,779],[117,358],[117,902],[117,331],[117,444],[117,1026],[117,322],[117,133],[117,255],[117,321],[117,1004],[117,1144],[117,332],[117,374],[118,207],[118,215],[118,270],[118,395],[118,518],[118,576],[118,209],[118,123],[118,178],[118,214],[118,606],[118,622],[118,220],[118,222],[118,221],[118,573],[118,327],[118,698],[118,230],[118,213],[118,476],[118,596],[118,208],[118,730],[118,269],[118,584],[118,399],[118,767],[118,542],[118,475],[118,265],[118,216],[118,335],[118,699],[118,1038],[118,217],[118,604],[118,514],[118,681],[118,146],[118,189],[118,336],[118,710],[118,631],[118,278],[118,385],[118,527],[118,398],[118,720],[118,532],[118,794],[119,259],[119,135],[119,365],[119,247],[119,354],[119,242],[119,248],[119,346],[119,122],[119,343],[119,299],[119,741],[119,333],[119,154],[119,171],[119,138],[119,168],[119,548],[119,165],[119,302],[119,134],[119,128],[119,127],[119,323],[119,314],[119,362],[119,178],[119,233],[119,256],[119,253],[119,355],[119,133],[119,292],[119,160],[119,674],[119,225],[119,250],[119,1029],[119,244],[119,258],[119,153],[119,1005],[119,990],[119,255],[119,1002],[119,462],[119,1115],[119,169],[119,316],[119,167],[119,1004],[119,421],[119,319],[119,758],[119,289],[119,136],[119,139],[119,375],[119,140],[119,125],[119,130],[119,174],[120,135],[120,122],[120,650],[120,696],[120,741],[120,333],[120,922],[120,130],[120,254],[120,161],[120,444],[120,127],[120,323],[120,167],[120,243],[120,419],[120,246],[120,174],[120,225],[120,256],[120,322],[120,160],[120,250],[120,258],[120,255],[120,316],[120,336],[120,1004],[120,421],[120,139],[120,375],[120,125],[121,298],[121,259],[121,135],[121,365],[121,248],[121,346],[121,567],[121,650],[121,741],[121,333],[121,246],[121,168],[121,830],[121,359],[121,127],[121,123],[121,225],[121,448],[121,131],[121,306],[121,447],[121,548],[121,605],[121,130],[121,165],[121,302],[121,371],[121,254],[121,161],[121,816],[121,323],[121,362],[121,313],[121,233],[121,256],[121,335],[121,322],[121,253],[121,355],[121,239],[121,133],[121,252],[121,160],[121,990],[121,255],[121,462],[121,169],[121,842],[121,316],[121,999],[121,167],[121,1140],[121,421],[121,972],[121,375],[121,339],[121,549],[121,125],[122,137],[122,135],[122,354],[122,248],[122,257],[122,247],[122,130],[122,245],[122,221],[122,133],[122,419],[122,252],[122,167],[122,139],[122,256],[122,140],[122,141],[122,174],[122,254],[122,123],[122,124],[122,125],[122,142],[122,249],[122,127],[122,128],[122,741],[122,333],[122,171],[122,561],[122,323],[122,160],[122,885],[122,255],[122,316],[122,1191],[122,474],[123,137],[123,124],[123,166],[123,298],[123,135],[123,365],[123,247],[123,354],[123,242],[123,248],[123,243],[123,464],[123,164],[123,361],[123,582],[123,434],[123,299],[123,649],[123,220],[123,567],[123,689],[123,650],[123,694],[123,696],[123,698],[123,741],[123,333],[123,584],[123,347],[123,329],[123,154],[123,251],[123,546],[123,791],[123,798],[123,379],[123,138],[123,168],[123,445],[123,447],[123,892],[123,893],[123,810],[123,548],[123,922],[123,130],[123,165],[123,958],[123,302],[123,425],[123,371],[123,254],[123,317],[123,982],[123,132],[123,351],[123,161],[123,128],[123,444],[123,127],[123,140],[123,323],[123,324],[123,178],[123,335],[123,322],[123,355],[123,133],[123,292],[123,160],[123,674],[123,257],[123,244],[123,258],[123,460],[123,885],[123,153],[123,141],[123,648],[123,625],[123,368],[123,1070],[123,255],[123,1002],[123,462],[123,422],[123,321],[123,169],[123,842],[123,316],[123,336],[123,172],[123,225],[123,175],[123,167],[123,356],[123,1004],[123,1140],[123,1142],[123,421],[123,1144],[123,1165],[123,391],[123,904],[123,332],[123,350],[123,448],[123,289],[123,367],[123,139],[123,1188],[123,804],[123,1191],[123,375],[123,737],[123,474],[123,310],[123,174],[123,125],[123,577],[123,380],[124,137],[124,215],[124,234],[124,235],[124,236],[124,156],[124,159],[124,237],[124,129],[124,238],[124,239],[124,136],[124,132],[124,138],[124,133],[124,162],[124,172],[124,142],[124,166],[124,288],[124,298],[124,259],[124,135],[124,247],[124,242],[124,248],[124,464],[124,383],[124,346],[124,164],[124,593],[124,617],[124,650],[124,741],[124,333],[124,329],[124,154],[124,171],[124,643],[124,379],[124,640],[124,366],[124,839],[124,168],[124,868],[124,548],[124,922],[124,130],[124,165],[124,302],[124,425],[124,254],[124,161],[124,128],[124,127],[124,140],[124,620],[124,323],[124,324],[124,178],[124,335],[124,322],[124,355],[124,292],[124,674],[124,225],[124,250],[124,1029],[124,244],[124,436],[124,258],[124,304],[124,153],[124,648],[124,1069],[124,255],[124,1002],[124,1115],[124,1117],[124,321],[124,618],[124,842],[124,316],[124,349],[124,155],[124,167],[124,356],[124,1135],[124,1140],[124,421],[124,175],[124,972],[124,720],[124,350],[124,290],[124,289],[124,139],[124,804],[124,981],[124,174],[124,1220],[125,298],[125,259],[125,365],[125,242],[125,426],[125,248],[125,346],[125,299],[125,650],[125,694],[125,696],[125,741],[125,333],[125,329],[125,171],[125,839],[125,306],[125,168],[125,447],[125,548],[125,922],[125,130],[125,302],[125,425],[125,254],[125,134],[125,351],[125,161],[125,128],[125,127],[125,140],[125,323],[125,314],[125,362],[125,178],[125,313],[125,233],[125,256],[125,335],[125,322],[125,355],[125,133],[125,292],[125,252],[125,160],[125,674],[125,250],[125,153],[125,625],[125,255],[125,1002],[125,462],[125,1008],[125,311],[125,321],[125,842],[125,349],[125,167],[125,1140],[125,320],[125,1144],[125,175],[125,972],[125,448],[125,289],[125,1011],[125,136],[125,139],[125,375],[125,174],[125,247],[125,246],[125,157],[125,131],[125,235],[125,403],[125,129],[125,445],[125,137],[125,138],[125,304],[125,141],[125,165],[125,251],[125,232],[125,142],[125,249],[125,225],[125,144],[126,262],[126,427],[126,209],[126,698],[126,130],[126,213],[126,269],[126,475],[126,600],[126,216],[126,497],[126,1060],[126,532],[126,536],[126,667],[127,137],[127,288],[127,298],[127,135],[127,365],[127,247],[127,248],[127,650],[127,741],[127,333],[127,347],[127,329],[127,154],[127,779],[127,856],[127,880],[127,445],[127,447],[127,358],[127,810],[127,548],[127,902],[127,331],[127,130],[127,165],[127,302],[127,371],[127,254],[127,317],[127,134],[127,161],[127,128],[127,167],[127,178],[127,245],[127,133],[127,354],[127,136],[127,138],[127,250],[127,139],[127,256],[127,172],[127,277],[127,141],[127,174],[127,142],[127,323],[127,335],[127,322],[127,355],[127,239],[127,292],[127,252],[127,368],[127,255],[127,462],[127,422],[127,311],[127,360],[127,316],[127,349],[127,1128],[127,1004],[127,1140],[127,332],[127,344],[127,860],[128,298],[128,259],[128,135],[128,242],[128,248],[128,243],[128,346],[128,434],[128,649],[128,650],[128,696],[128,333],[128,347],[128,546],[128,171],[128,798],[128,817],[128,839],[128,855],[128,880],[128,943],[128,165],[128,302],[128,425],[128,254],[128,132],[128,134],[128,161],[128,178],[128,365],[128,257],[128,130],[128,245],[128,277],[128,779],[128,323],[128,158],[128,131],[128,235],[128,289],[128,162],[128,133],[128,324],[128,175],[128,237],[128,457],[128,252],[128,168],[128,445],[128,167],[128,137],[128,304],[128,355],[128,335],[128,250],[128,320],[128,139],[128,256],[128,446],[128,276],[128,448],[128,314],[128,141],[128,174],[128,176],[128,999],[128,142],[128,249],[128,291],[128,239],[128,362],[128,922],[128,313],[128,233],[128,322],[128,292],[128,160],[128,1029],[128,969],[128,368],[128,255],[128,1113],[128,618],[128,316],[128,336],[128,1140],[128,421],[128,391],[128,972],[128,1011],[128,136],[128,905],[128,474],[129,288],[129,298],[129,259],[129,135],[129,242],[129,248],[129,346],[129,427],[129,669],[129,211],[129,707],[129,708],[129,741],[129,333],[129,769],[129,775],[129,154],[129,379],[129,626],[129,652],[129,839],[129,306],[129,528],[129,168],[129,358],[129,810],[129,548],[129,933],[129,902],[129,331],[129,165],[129,302],[129,309],[129,254],[129,175],[129,323],[129,314],[129,324],[129,313],[129,233],[129,335],[129,355],[129,239],[129,133],[129,252],[129,674],[129,225],[129,1070],[129,255],[129,1002],[129,462],[129,1008],[129,311],[129,321],[129,400],[129,316],[129,167],[129,1140],[129,1145],[129,319],[129,448],[129,289],[129,1187],[129,301],[129,310],[129,1073],[129,860],[130,298],[130,259],[130,135],[130,365],[130,464],[130,346],[130,164],[130,361],[130,586],[130,434],[130,299],[130,628],[130,348],[130,649],[130,696],[130,698],[130,352],[130,741],[130,333],[130,154],[130,357],[130,546],[130,171],[130,306],[130,855],[130,168],[130,892],[130,548],[130,922],[130,950],[130,324],[130,174],[130,301],[130,384],[130,139],[130,237],[130,137],[130,157],[130,320],[130,138],[130,335],[130,362],[130,233],[130,134],[130,165],[130,323],[130,254],[130,245],[130,167],[130,371],[130,136],[130,372],[130,322],[130,302],[130,425],[130,286],[130,132],[130,314],[130,256],[130,253],[130,355],[130,133],[130,252],[130,160],[130,674],[130,250],[130,436],[130,695],[130,141],[130,648],[130,255],[130,462],[130,360],[130,842],[130,316],[130,336],[130,156],[130,999],[130,421],[130,670],[130,1168],[130,350],[130,448],[130,1192],[130,374],[130,860],[131,137],[131,215],[131,288],[131,298],[131,259],[131,242],[131,426],[131,243],[131,343],[131,361],[131,593],[131,595],[131,434],[131,299],[131,694],[131,697],[131,741],[131,333],[131,777],[131,827],[131,245],[131,447],[131,548],[131,165],[131,302],[131,425],[131,371],[131,132],[131,351],[131,140],[131,323],[131,324],[131,178],[131,313],[131,233],[131,253],[131,355],[131,292],[131,674],[131,250],[131,1029],[131,244],[131,436],[131,153],[131,255],[131,1002],[131,462],[131,422],[131,311],[131,342],[131,321],[131,169],[131,167],[131,1140],[131,421],[131,1144],[131,670],[131,1165],[131,904],[131,332],[131,972],[131,448],[131,289],[131,1178],[131,139],[131,241],[132,288],[132,426],[132,741],[132,333],[132,379],[132,626],[132,358],[132,475],[132,902],[132,165],[132,302],[132,234],[132,139],[132,133],[132,136],[132,304],[132,157],[132,225],[132,323],[132,324],[132,235],[132,463],[132,140],[132,335],[132,239],[132,255],[132,167],[132,356],[132,1140],[132,1192],[132,1220],[133,135],[133,247],[133,354],[133,248],[133,697],[133,741],[133,333],[133,652],[133,168],[133,902],[133,302],[133,425],[133,254],[133,140],[133,323],[133,324],[133,256],[133,322],[133,257],[133,277],[133,139],[133,239],[133,234],[133,136],[133,138],[133,164],[133,166],[133,252],[133,161],[133,167],[133,236],[133,172],[133,160],[133,290],[133,178],[133,175],[133,165],[133,355],[133,174],[133,142],[133,249],[133,225],[133,171],[133,141],[133,463],[133,258],[133,255],[133,316],[133,972],[134,259],[134,451],[134,434],[134,299],[134,696],[134,741],[134,744],[134,333],[134,347],[134,154],[134,777],[134,811],[134,306],[134,880],[134,447],[134,358],[134,548],[134,943],[134,465],[134,425],[134,371],[134,254],[134,335],[134,178],[134,145],[134,277],[134,165],[134,247],[134,245],[134,328],[134,365],[134,324],[134,167],[134,323],[134,139],[134,140],[134,256],[134,359],[134,314],[134,174],[134,830],[134,362],[134,816],[134,1032],[134,313],[134,322],[134,345],[134,292],[134,252],[134,153],[134,255],[134,1002],[134,422],[134,311],[134,618],[134,336],[134,999],[134,1138],[134,1140],[134,758],[134,972],[134,448],[134,1075],[134,991],[134,860],[134,380],[135,268],[135,298],[135,316],[135,317],[135,318],[135,302],[135,319],[135,174],[135,172],[135,178],[135,320],[135,167],[135,321],[135,158],[135,245],[135,322],[135,323],[135,301],[135,324],[135,225],[135,234],[135,325],[135,244],[135,142],[135,157],[135,326],[135,154],[135,273],[135,140],[135,306],[135,327],[135,292],[135,137],[135,256],[135,282],[135,328],[135,241],[135,246],[135,329],[135,330],[135,331],[135,332],[135,333],[135,289],[135,334],[135,335],[135,336],[135,138],[135,337],[135,250],[135,255],[135,338],[135,339],[135,340],[135,341],[135,305],[135,342],[135,141],[135,254],[135,343],[135,232],[135,160],[135,344],[135,345],[135,346],[135,347],[135,348],[135,349],[135,155],[135,288],[135,350],[135,166],[135,175],[135,351],[135,168],[135,161],[135,352],[135,353],[135,233],[135,354],[135,236],[135,355],[135,356],[135,357],[135,259],[135,358],[135,359],[135,314],[135,360],[135,291],[135,361],[135,362],[135,363],[135,364],[135,551],[135,741],[135,548],[135,258],[136,288],[136,259],[136,365],[136,247],[136,354],[136,242],[136,248],[136,333],[136,791],[136,652],[136,447],[136,921],[136,302],[136,425],[136,371],[136,254],[136,140],[136,323],[136,256],[136,322],[136,252],[136,160],[136,674],[136,1029],[136,255],[136,1113],[136,842],[136,167],[136,356],[136,1140],[136,1144],[136,367],[136,138],[136,250],[136,139],[136,245],[136,225],[137,153],[137,154],[137,155],[137,156],[137,157],[137,158],[137,159],[137,160],[137,161],[137,162],[137,163],[137,164],[137,165],[137,166],[137,167],[137,168],[137,169],[137,170],[137,138],[137,171],[137,172],[137,173],[137,141],[137,174],[137,175],[137,176],[137,142],[137,177],[137,178],[137,298],[137,365],[137,248],[137,337],[137,568],[137,582],[137,642],[137,649],[137,706],[137,741],[137,775],[137,783],[137,546],[137,626],[137,652],[137,846],[137,852],[137,447],[137,892],[137,893],[137,461],[137,548],[137,902],[137,465],[137,425],[137,913],[137,317],[137,313],[137,233],[137,335],[137,322],[137,253],[137,355],[137,252],[137,674],[137,244],[137,436],[137,1018],[137,363],[137,1034],[137,625],[137,255],[137,1002],[137,291],[137,1110],[137,311],[137,618],[137,900],[137,771],[137,400],[137,360],[137,1121],[137,1040],[137,842],[137,316],[137,349],[137,999],[137,1140],[137,384],[137,421],[137,1144],[137,1159],[137,758],[137,904],[137,332],[137,972],[137,290],[137,1173],[137,367],[137,898],[137,1191],[137,1192],[137,375],[137,737],[137,991],[137,981],[137,1203],[137,1207],[137,822],[137,1073],[137,424],[137,1204],[138,365],[138,247],[138,354],[138,242],[138,248],[138,443],[138,343],[138,434],[138,650],[138,696],[138,163],[138,741],[138,333],[138,171],[138,790],[138,779],[138,167],[138,141],[138,289],[138,235],[138,168],[138,548],[138,425],[138,371],[138,161],[138,140],[138,345],[138,355],[138,292],[138,252],[138,160],[138,258],[138,368],[138,255],[138,321],[138,842],[138,316],[138,1004],[138,1142],[138,421],[138,391],[138,139],[138,174],[139,298],[139,259],[139,365],[139,247],[139,354],[139,242],[139,426],[139,248],[139,337],[139,361],[139,434],[139,649],[139,650],[139,694],[139,741],[139,333],[139,546],[139,790],[139,798],[139,494],[139,379],[139,652],[139,306],[139,779],[139,447],[139,548],[139,922],[139,950],[139,165],[139,302],[139,425],[139,254],[139,317],[139,351],[139,161],[139,444],[139,140],[139,816],[139,323],[139,362],[139,233],[139,256],[139,335],[139,322],[139,355],[139,252],[139,160],[139,674],[139,250],[139,1029],[139,244],[139,258],[139,460],[139,648],[139,255],[139,1002],[139,462],[139,311],[139,321],[139,618],[139,900],[139,349],[139,167],[139,356],[139,1140],[139,421],[139,1144],[139,319],[139,972],[139,448],[139,1017],[139,614],[139,1011],[139,238],[139,166],[139,157],[139,237],[139,142],[139,245],[139,174],[139,141],[139,144],[139,282],[139,375],[139,549],[139,911],[139,310],[139,374],[139,822],[139,380],[140,259],[140,248],[140,337],[140,346],[140,299],[140,694],[140,698],[140,741],[140,742],[140,171],[140,652],[140,880],[140,893],[140,902],[140,302],[140,351],[140,234],[140,153],[140,335],[140,172],[140,426],[140,245],[140,324],[140,185],[140,354],[140,356],[140,314],[140,463],[140,323],[140,362],[140,313],[140,256],[140,355],[140,292],[140,252],[140,258],[140,695],[140,864],[140,915],[140,255],[140,312],[140,1002],[140,349],[140,167],[140,1140],[140,916],[140,972],[140,448],[140,289],[140,1178],[140,860],[140,424],[141,288],[141,242],[141,434],[141,348],[141,649],[141,163],[141,741],[141,333],[141,329],[141,154],[141,546],[141,652],[141,827],[141,306],[141,779],[141,168],[141,892],[141,896],[141,358],[141,810],[141,922],[141,941],[141,165],[141,302],[141,371],[141,254],[141,323],[141,256],[141,335],[141,322],[141,253],[141,355],[141,292],[141,252],[141,160],[141,250],[141,1029],[141,258],[141,969],[141,232],[141,255],[141,321],[141,618],[141,316],[141,167],[141,1004],[141,1140],[141,421],[141,320],[141,972],[141,367],[141,339],[141,549],[141,174],[141,822],[141,860],[142,298],[142,365],[142,242],[142,248],[142,243],[142,543],[142,343],[142,361],[142,382],[142,612],[142,434],[142,299],[142,348],[142,650],[142,692],[142,696],[142,163],[142,741],[142,744],[142,750],[142,333],[142,329],[142,154],[142,777],[142,171],[142,795],[142,839],[142,306],[142,854],[142,445],[142,292],[142,922],[142,165],[142,302],[142,425],[142,967],[142,968],[142,254],[142,161],[142,323],[142,256],[142,335],[142,322],[142,253],[142,308],[142,355],[142,160],[142,674],[142,250],[142,1029],[142,244],[142,258],[142,864],[142,255],[142,1002],[142,462],[142,1008],[142,311],[142,1115],[142,321],[142,169],[142,360],[142,316],[142,336],[142,167],[142,693],[142,1004],[142,421],[142,1144],[142,1165],[142,904],[142,332],[142,448],[142,289],[142,367],[142,344],[142,1011],[142,1192],[142,911],[142,474],[142,822],[143,144],[143,145],[143,146],[143,147],[143,988],[144,215],[144,268],[144,293],[144,297],[144,242],[144,426],[144,435],[144,450],[144,469],[144,209],[144,624],[144,731],[144,521],[144,529],[144,777],[144,519],[144,506],[144,666],[144,793],[144,801],[144,640],[144,673],[144,656],[144,865],[144,542],[144,923],[144,832],[144,871],[144,941],[144,944],[144,201],[144,953],[144,441],[144,965],[144,210],[144,982],[144,988],[144,265],[144,1000],[144,362],[144,178],[144,355],[144,202],[144,768],[144,1047],[144,712],[144,604],[144,492],[144,648],[144,834],[144,681],[144,255],[144,511],[144,231],[144,1148],[144,527],[144,720],[144,289],[144,532],[144,344],[144,1178],[144,1180],[144,1191],[144,534],[144,1201],[144,503],[144,227],[144,704],[144,738],[144,637],[145,179],[145,285],[145,477],[145,518],[145,287],[145,576],[145,180],[145,222],[145,587],[145,185],[145,628],[145,427],[145,220],[145,491],[145,209],[145,664],[145,537],[145,519],[145,797],[145,640],[145,528],[145,583],[145,869],[145,721],[145,876],[145,501],[145,668],[145,965],[145,498],[145,988],[145,1023],[145,216],[145,1035],[145,1047],[145,408],[145,436],[145,635],[145,857],[145,1067],[145,1027],[145,146],[145,601],[145,393],[145,1112],[145,831],[145,1060],[145,360],[145,1129],[145,710],[145,1086],[145,729],[145,631],[145,719],[145,720],[145,532],[145,480],[145,1194],[145,574],[145,474],[145,301],[145,862],[145,182],[145,1209],[146,467],[146,531],[146,190],[146,721],[146,510],[146,475],[146,705],[146,433],[146,210],[146,988],[146,468],[146,523],[146,408],[146,436],[146,230],[146,172],[146,413],[146,596],[146,221],[146,217],[146,218],[146,184],[146,219],[146,220],[146,656],[146,722],[146,183],[146,835],[146,152],[146,278],[146,223],[146,377],[146,560],[146,266],[146,558],[146,188],[146,222],[146,185],[146,287],[146,393],[146,710],[146,227],[146,589],[147,531],[147,535],[147,510],[147,988],[147,1038],[147,225],[147,206],[147,196],[147,222],[147,663],[147,200],[147,193],[147,482],[147,417],[149,150],[149,151],[149,152],[149,531],[149,436],[149,189],[150,270],[150,518],[150,531],[150,636],[150,657],[150,152],[150,698],[150,504],[150,743],[150,753],[150,508],[150,793],[150,673],[150,686],[150,996],[150,588],[150,555],[150,436],[150,857],[150,482],[150,474],[150,1201],[150,1208],[150,722],[151,184],[151,518],[151,392],[151,936],[151,441],[151,996],[151,1134],[151,480],[151,503],[151,417],[152,215],[152,285],[152,184],[152,518],[152,587],[152,197],[152,661],[152,801],[152,583],[152,510],[152,965],[152,433],[152,204],[152,497],[152,523],[152,436],[152,575],[152,857],[152,1060],[152,538],[152,710],[152,385],[152,1166],[152,482],[152,206],[152,474],[152,1201],[152,1209],[153,299],[153,741],[153,333],[153,795],[153,331],[153,465],[153,317],[153,178],[153,1032],[153,359],[153,233],[153,345],[153,252],[153,341],[153,157],[153,1031],[153,308],[153,166],[153,775],[153,175],[153,225],[153,362],[153,255],[153,422],[153,311],[153,771],[153,842],[153,349],[153,999],[153,167],[153,1140],[153,375],[153,1193],[153,1074],[153,307],[154,242],[154,649],[154,697],[154,163],[154,333],[154,446],[154,161],[154,160],[154,650],[154,156],[154,290],[154,170],[154,236],[154,776],[154,172],[154,159],[154,245],[154,324],[154,166],[154,167],[154,225],[154,165],[154,171],[154,250],[154,155],[154,254],[154,546],[154,371],[154,620],[154,372],[154,368],[154,321],[154,618],[154,999],[154,972],[154,301],[155,164],[155,343],[155,434],[155,649],[155,650],[155,777],[155,783],[155,546],[155,652],[155,826],[155,779],[155,358],[155,548],[155,425],[155,371],[155,444],[155,335],[155,252],[155,160],[155,674],[155,250],[155,217],[155,177],[155,341],[155,1005],[155,321],[155,316],[155,156],[155,161],[155,163],[155,446],[155,625],[155,253],[155,318],[155,474],[155,726],[155,238],[155,166],[155,170],[155,172],[155,290],[155,1030],[155,167],[155,1144],[155,972],[155,344],[155,633],[155,374],[156,166],[156,365],[156,248],[156,164],[156,434],[156,619],[156,642],[156,646],[156,348],[156,649],[156,617],[156,687],[156,163],[156,741],[156,333],[156,783],[156,546],[156,785],[156,795],[156,811],[156,813],[156,846],[156,854],[156,893],[156,331],[156,371],[156,620],[156,322],[156,292],[156,252],[156,160],[156,695],[156,969],[156,1058],[156,625],[156,255],[156,422],[156,400],[156,842],[156,349],[156,1015],[156,1125],[156,328],[156,650],[156,161],[156,171],[156,172],[156,332],[156,633],[156,905],[156,1192],[156,549],[156,1073],[156,424],[157,166],[157,298],[157,248],[157,164],[157,582],[157,348],[157,649],[157,694],[157,163],[157,754],[157,333],[157,546],[157,652],[157,306],[157,855],[157,893],[157,358],[157,548],[157,165],[157,302],[157,425],[157,371],[157,976],[157,351],[157,178],[157,233],[157,256],[157,335],[157,322],[157,160],[157,177],[157,648],[157,1070],[157,255],[157,422],[157,321],[157,172],[157,235],[157,383],[157,618],[157,900],[157,842],[157,316],[157,167],[157,1140],[157,421],[157,319],[157,972],[157,448],[157,367],[157,344],[157,991],[157,173],[158,288],[158,459],[158,581],[158,236],[158,783],[158,379],[158,868],[158,810],[158,162],[158,324],[158,249],[158,252],[158,1034],[158,1070],[158,255],[158,771],[158,316],[158,1140],[158,758],[158,332],[158,290],[158,1173],[159,242],[159,299],[159,163],[159,741],[159,333],[159,775],[159,810],[159,302],[159,323],[159,252],[159,160],[159,674],[159,255],[159,422],[159,842],[159,373],[159,167],[159,1004],[159,330],[159,972],[159,737],[160,288],[160,365],[160,337],[160,348],[160,649],[160,617],[160,650],[160,163],[160,352],[160,741],[160,546],[160,892],[160,548],[160,921],[160,165],[160,161],[160,620],[160,233],[160,322],[160,355],[160,252],[160,245],[160,368],[160,1031],[160,237],[160,289],[160,1041],[160,162],[160,324],[160,369],[160,311],[160,168],[160,167],[160,354],[160,171],[160,250],[160,446],[160,172],[160,1042],[160,242],[160,343],[160,232],[160,254],[160,175],[160,257],[160,249],[160,291],[160,225],[160,177],[160,372],[160,771],[160,316],[160,999],[160,1142],[160,332],[160,1192],[160,981],[160,301],[160,310],[160,822],[161,288],[161,365],[161,242],[161,248],[161,429],[161,619],[161,299],[161,348],[161,649],[161,617],[161,650],[161,694],[161,163],[161,741],[161,333],[161,777],[161,546],[161,171],[161,855],[161,168],[161,548],[161,921],[161,165],[161,425],[161,254],[161,351],[161,258],[161,282],[161,167],[161,251],[161,250],[161,172],[161,292],[161,249],[161,620],[161,1032],[161,322],[161,252],[161,674],[161,431],[161,663],[161,255],[161,1002],[161,422],[161,311],[161,771],[161,316],[161,1127],[161,1144],[161,1033],[161,905],[161,1192],[161,981],[161,301],[162,298],[162,607],[162,434],[162,348],[162,617],[162,706],[162,236],[162,333],[162,795],[162,379],[162,825],[162,779],[162,868],[162,447],[162,893],[162,950],[162,403],[162,165],[162,302],[162,371],[162,254],[162,290],[162,282],[162,245],[162,238],[162,235],[162,167],[162,170],[162,446],[162,255],[162,172],[162,176],[162,239],[162,234],[162,323],[162,335],[162,322],[162,292],[162,252],[162,250],[162,436],[162,258],[162,341],[162,1034],[162,1070],[162,462],[162,422],[162,321],[162,900],[162,771],[162,400],[162,842],[162,999],[162,1140],[162,421],[162,1033],[162,332],[162,972],[162,1168],[162,614],[162,1178],[162,1042],[162,1132],[162,1187],[162,981],[162,374],[162,1073],[163,348],[163,649],[163,617],[163,650],[163,235],[163,371],[163,175],[163,737],[163,177],[163,165],[163,166],[163,168],[163,171],[163,250],[163,446],[163,172],[163,176],[163,741],[163,333],[163,546],[163,826],[163,921],[163,620],[163,322],[163,648],[163,255],[163,167],[163,1192],[163,981],[164,166],[164,464],[164,172],[164,225],[164,546],[164,547],[164,320],[164,257],[164,369],[164,188],[164,582],[164,619],[164,299],[164,649],[164,333],[164,341],[164,342],[164,720],[164,350],[164,1178],[164,905],[165,298],[165,248],[165,464],[165,434],[165,299],[165,567],[165,694],[165,706],[165,741],[165,744],[165,333],[165,347],[165,775],[165,652],[165,827],[165,779],[165,852],[165,855],[165,873],[165,880],[165,892],[165,358],[165,461],[165,548],[165,902],[165,951],[165,172],[165,167],[165,175],[165,235],[165,258],[165,324],[165,252],[165,168],[165,233],[165,323],[165,169],[165,304],[165,250],[165,255],[165,277],[165,291],[165,239],[165,246],[165,282],[165,425],[165,351],[165,249],[165,256],[165,335],[165,322],[165,648],[165,625],[165,422],[165,400],[165,316],[165,349],[165,325],[165,1001],[165,1130],[165,1140],[165,320],[165,670],[165,385],[165,350],[165,290],[165,375],[165,173],[165,981],[165,374],[166,279],[166,178],[166,280],[166,281],[166,282],[166,283],[166,167],[166,170],[166,171],[166,172],[166,284],[166,365],[166,333],[166,922],[166,1026],[166,322],[166,842],[166,316],[166,353],[166,301],[167,259],[167,248],[167,337],[167,452],[167,550],[167,361],[167,582],[167,612],[167,315],[167,299],[167,649],[167,567],[167,650],[167,696],[167,352],[167,741],[167,333],[167,546],[167,171],[167,626],[167,652],[167,306],[167,778],[167,779],[167,866],[167,447],[167,893],[167,821],[167,358],[167,548],[167,958],[167,425],[167,254],[167,444],[167,1024],[167,323],[167,233],[167,256],[167,322],[167,253],[167,355],[167,292],[167,252],[167,250],[167,244],[167,258],[167,969],[167,1005],[167,1070],[167,255],[167,1008],[167,422],[167,311],[167,321],[167,618],[167,360],[167,316],[167,349],[167,373],[167,999],[167,1001],[167,282],[167,246],[167,172],[167,239],[167,234],[167,243],[167,174],[167,257],[167,232],[167,245],[167,277],[167,276],[167,308],[167,419],[167,366],[167,247],[167,251],[167,175],[167,1128],[167,384],[167,177],[167,173],[167,178],[167,176],[167,320],[167,449],[167,1055],[167,1039],[167,346],[167,347],[167,959],[167,1028],[167,328],[167,758],[167,235],[167,289],[167,168],[167,354],[167,170],[167,960],[167,844],[167,314],[167,249],[167,421],[167,1144],[167,1165],[167,332],[167,290],[167,344],[167,905],[167,1198],[167,1073],[168,426],[168,694],[168,306],[168,175],[168,354],[168,250],[168,257],[168,880],[168,358],[168,902],[168,371],[168,351],[168,323],[168,233],[168,256],[168,335],[168,322],[168,372],[168,368],[168,618],[168,316],[168,301],[169,242],[169,922],[169,403],[169,254],[169,258],[169,402],[169,355],[169,243],[169,277],[170,333],[170,252],[170,255],[170,1140],[170,290],[170,1132],[170,1073],[171,649],[171,650],[171,741],[171,333],[171,546],[171,258],[171,446],[171,172],[171,355],[171,292],[171,245],[171,289],[171,249],[171,839],[171,779],[171,252],[171,674],[171,250],[171,255],[171,972],[172,288],[172,298],[172,365],[172,242],[172,284],[172,464],[172,477],[172,568],[172,578],[172,582],[172,434],[172,619],[172,176],[172,649],[172,662],[172,650],[172,706],[172,394],[172,584],[172,546],[172,358],[172,548],[172,924],[172,950],[172,403],[172,425],[172,371],[172,433],[172,1022],[172,324],[172,313],[172,322],[172,253],[172,355],[172,292],[172,252],[172,674],[172,225],[172,244],[172,177],[172,341],[172,604],[172,363],[172,1069],[172,255],[172,1008],[172,422],[172,291],[172,321],[172,618],[172,771],[172,842],[172,316],[172,538],[172,1129],[172,1004],[172,421],[172,972],[172,1168],[172,350],[172,290],[172,1173],[172,344],[172,1075],[172,1183],[172,1187],[172,1192],[172,737],[172,301],[172,536],[172,182],[172,310],[172,1073],[173,652],[173,445],[173,358],[173,322],[173,421],[173,284],[173,174],[174,298],[174,365],[174,242],[174,248],[174,434],[174,299],[174,696],[174,714],[174,741],[174,333],[174,357],[174,306],[174,445],[174,447],[174,475],[174,425],[174,371],[174,254],[174,376],[174,444],[174,323],[174,314],[174,256],[174,322],[174,355],[174,252],[174,250],[174,258],[174,372],[174,1061],[174,368],[174,255],[174,462],[174,321],[174,618],[174,316],[174,336],[174,1128],[174,421],[174,320],[174,332],[174,367],[174,339],[174,549],[174,241],[174,301],[174,362],[174,178],[174,249],[174,232],[174,247],[174,328],[174,352],[174,251],[174,304],[174,345],[174,243],[174,225],[175,298],[175,259],[175,365],[175,354],[175,248],[175,434],[175,299],[175,669],[175,696],[175,741],[175,333],[175,347],[175,652],[175,855],[175,447],[175,331],[175,943],[175,302],[175,210],[175,323],[175,178],[175,233],[175,256],[175,335],[175,322],[175,355],[175,252],[175,255],[175,312],[175,900],[175,771],[175,360],[175,316],[175,349],[175,1140],[175,421],[175,670],[175,249],[175,308],[175,277],[175,225],[175,257],[175,448],[175,344],[175,375],[175,241],[175,822],[176,259],[176,633],[176,826],[176,846],[176,893],[176,371],[176,322],[176,252],[176,436],[176,372],[176,368],[176,321],[176,618],[176,758],[176,290],[176,301],[177,652],[177,358],[177,1034],[177,1128],[178,259],[178,365],[178,242],[178,234],[178,243],[178,337],[178,551],[178,361],[178,642],[178,457],[178,454],[178,725],[178,741],[178,269],[178,347],[178,775],[178,802],[178,652],[178,844],[178,856],[178,893],[178,358],[178,548],[178,302],[178,362],[178,1031],[178,308],[178,359],[178,232],[178,225],[178,344],[178,830],[178,241],[178,313],[178,256],[178,345],[178,355],[178,292],[178,252],[178,594],[178,625],[178,1070],[178,1113],[178,1114],[178,321],[178,900],[178,316],[178,349],[178,373],[178,1071],[178,1140],[178,421],[178,758],[178,332],[178,720],[178,1178],[178,991],[178,860],[178,307],[179,180],[179,181],[179,182],[179,183],[179,184],[179,185],[179,186],[179,187],[179,188],[179,189],[179,190],[179,510],[179,710],[180,215],[180,467],[180,634],[180,209],[180,541],[180,211],[180,386],[180,519],[180,789],[180,797],[180,640],[180,824],[180,528],[180,583],[180,861],[180,668],[180,947],[180,216],[180,768],[180,523],[180,604],[180,492],[180,502],[180,187],[180,189],[180,278],[180,527],[180,525],[180,182],[180,1210],[181,587],[181,377],[181,209],[181,624],[181,274],[181,730],[181,731],[181,495],[181,519],[181,800],[181,528],[181,505],[181,668],[181,475],[181,488],[181,425],[181,204],[181,468],[181,497],[181,217],[181,408],[181,436],[181,712],[181,1027],[181,189],[181,1086],[181,391],[181,1170],[181,532],[181,206],[181,485],[181,474],[181,704],[181,1059],[182,220],[182,655],[182,476],[182,197],[182,723],[182,743],[182,495],[182,540],[182,673],[182,656],[182,510],[182,254],[182,204],[182,216],[182,575],[182,492],[182,889],[182,663],[182,1027],[182,681],[182,189],[182,557],[182,717],[182,538],[182,710],[182,278],[182,527],[182,474],[182,396],[182,433],[182,724],[182,266],[182,572],[182,470],[182,391],[182,831],[182,452],[182,605],[182,446],[182,265],[182,188],[182,558],[182,573],[182,427],[182,185],[182,699],[182,453],[182,211],[182,1172],[182,183],[182,1101],[182,576],[182,222],[182,819],[183,215],[183,435],[183,455],[183,469],[183,518],[183,254],[183,563],[183,606],[183,662],[183,581],[183,723],[183,470],[183,481],[183,535],[183,583],[183,923],[183,475],[183,605],[183,433],[183,497],[183,217],[183,436],[183,1053],[183,604],[183,889],[183,648],[183,597],[183,1069],[183,187],[183,189],[183,400],[183,1129],[183,356],[183,710],[183,385],[183,527],[183,193],[183,532],[183,474],[184,206],[184,209],[184,211],[184,474],[184,475],[184,436],[184,476],[184,274],[184,199],[184,187],[184,214],[184,229],[184,468],[184,265],[184,208],[184,231],[184,385],[184,452],[184,223],[184,433],[184,518],[184,197],[184,495],[184,673],[184,721],[184,510],[184,936],[184,216],[184,889],[184,710],[184,480],[185,261],[185,275],[185,285],[185,435],[185,550],[185,518],[185,587],[185,277],[185,287],[185,276],[185,576],[185,434],[185,628],[185,491],[185,655],[185,488],[185,705],[185,961],[185,1032],[185,408],[185,1067],[185,1027],[185,1070],[185,601],[185,397],[185,336],[185,1129],[185,356],[185,710],[185,1033],[185,720],[185,532],[185,482],[185,474],[186,455],[186,518],[186,571],[186,628],[186,214],[186,731],[186,392],[186,519],[186,528],[186,849],[186,583],[186,501],[186,599],[186,871],[186,941],[186,600],[186,686],[186,569],[186,768],[186,1047],[186,523],[186,436],[186,524],[186,604],[186,492],[186,663],[186,189],[186,538],[186,710],[186,562],[186,833],[186,385],[186,527],[186,720],[186,485],[186,474],[186,1088],[186,862],[186,704],[186,1163],[187,191],[187,215],[187,262],[187,268],[187,285],[187,270],[187,467],[187,469],[187,518],[187,531],[187,587],[187,563],[187,606],[187,220],[187,218],[187,491],[187,624],[187,560],[187,476],[187,197],[187,223],[187,596],[187,472],[187,274],[187,731],[187,661],[187,269],[187,470],[187,603],[187,266],[187,584],[187,519],[187,506],[187,812],[187,626],[187,535],[187,673],[187,528],[187,583],[187,656],[187,761],[187,475],[187,599],[187,932],[187,488],[187,201],[187,605],[187,705],[187,686],[187,965],[187,286],[187,265],[187,216],[187,1038],[187,768],[187,436],[187,759],[187,1053],[187,203],[187,663],[187,231],[187,400],[187,581],[187,230],[187,226],[187,193],[187,196],[187,699],[187,377],[187,453],[187,221],[187,474],[187,709],[187,558],[187,188],[187,264],[187,273],[187,396],[187,410],[187,580],[187,1023],[187,391],[187,439],[187,440],[187,667],[187,486],[187,492],[187,634],[187,953],[187,1077],[187,559],[187,414],[187,229],[187,468],[187,485],[187,525],[187,675],[187,483],[187,797],[187,208],[187,493],[187,520],[187,592],[187,644],[187,189],[187,511],[187,1114],[187,565],[187,1129],[187,416],[187,710],[187,833],[187,385],[187,527],[188,215],[188,270],[188,469],[188,286],[188,222],[188,587],[188,606],[188,573],[188,504],[188,470],[188,529],[188,205],[188,519],[188,190],[188,475],[188,871],[188,498],[188,217],[188,408],[188,575],[188,604],[188,663],[188,1067],[188,479],[188,189],[188,1116],[188,557],[188,1060],[188,1129],[188,710],[188,527],[188,398],[188,391],[188,1170],[188,193],[188,532],[188,1186],[188,474],[188,1209],[188,264],[189,208],[189,526],[189,1053],[189,604],[189,226],[189,432],[189,476],[189,220],[189,214],[189,560],[189,209],[189,388],[189,230],[189,221],[189,266],[189,217],[189,327],[189,511],[189,427],[189,229],[189,474],[189,491],[189,573],[189,780],[189,557],[189,720],[189,231],[189,452],[189,387],[189,823],[189,194],[189,340],[189,941],[189,433],[189,439],[189,1137],[190,518],[190,606],[190,221],[190,214],[190,581],[190,274],[190,730],[190,740],[190,584],[190,347],[190,223],[190,748],[190,218],[190,635],[190,535],[190,396],[190,803],[190,475],[190,436],[190,685],[190,193],[190,580],[190,391],[190,667],[190,492],[190,804],[190,356],[190,377],[190,485],[190,604],[190,560],[190,675],[190,726],[190,278],[190,558],[190,231],[190,493],[190,805],[190,712],[190,806],[190,786],[190,476],[190,583],[190,767],[190,433],[190,385],[190,720],[190,367],[190,589],[191,192],[191,193],[191,194],[191,195],[191,196],[191,197],[191,198],[191,199],[191,200],[191,201],[191,202],[191,203],[191,204],[191,205],[191,206],[191,535],[192,214],[192,504],[192,297],[192,209],[192,266],[192,439],[192,656],[193,467],[193,469],[193,518],[193,563],[193,491],[193,224],[193,196],[193,661],[193,470],[193,495],[193,529],[193,666],[193,271],[193,760],[193,535],[193,478],[193,528],[193,656],[193,407],[193,510],[193,533],[193,475],[193,599],[193,201],[193,600],[193,512],[193,966],[193,433],[193,500],[193,359],[193,1038],[193,436],[193,759],[193,604],[193,834],[193,597],[193,565],[193,538],[193,833],[193,385],[193,527],[193,391],[193,215],[193,223],[193,480],[193,221],[193,211],[193,1108],[193,638],[193,639],[193,709],[193,540],[193,831],[193,1163],[193,634],[193,517],[193,953],[193,194],[193,482],[193,222],[193,229],[193,195],[193,835],[193,485],[193,377],[193,1171],[193,511],[193,558],[193,231],[193,805],[193,1172],[193,644],[193,679],[193,717],[193,212],[194,267],[194,721],[194,294],[194,206],[194,782],[194,503],[195,268],[195,600],[195,512],[195,596],[195,739],[195,527],[195,417],[196,215],[196,247],[196,476],[196,521],[196,271],[196,535],[196,478],[196,528],[196,501],[196,599],[196,526],[196,201],[196,600],[196,512],[196,500],[196,517],[196,202],[196,203],[196,834],[196,597],[196,511],[196,565],[196,831],[196,440],[196,538],[196,562],[196,833],[196,527],[196,720],[196,835],[196,722],[197,377],[197,209],[197,211],[197,214],[197,457],[197,213],[197,208],[197,265],[197,395],[197,557],[197,452],[197,505],[197,896],[197,475],[197,605],[197,527],[197,474],[197,417],[198,535],[198,835],[199,207],[199,432],[199,208],[199,294],[199,206],[199,1057],[199,340],[199,440],[199,503],[199,700],[200,721],[200,505],[200,475],[200,436],[200,435],[200,340],[200,439],[200,503],[201,766],[201,529],[201,535],[201,925],[201,1038],[202,535],[202,377],[202,214],[202,710],[202,794],[203,531],[203,267],[203,623],[203,661],[203,519],[203,535],[203,505],[203,512],[203,1038],[203,214],[203,294],[203,206],[203,576],[203,219],[203,833],[204,477],[204,667],[204,1020],[204,210],[204,452],[204,1021],[204,206],[204,222],[204,294],[205,405],[205,410],[205,526],[205,286],[206,563],[206,209],[206,661],[206,521],[206,537],[206,519],[206,789],[206,793],[206,869],[206,533],[206,475],[206,599],[206,957],[206,512],[206,441],[206,762],[206,265],[206,517],[206,575],[206,524],[206,340],[206,557],[206,739],[206,416],[206,710],[206,385],[206,532],[206,782],[206,564],[206,747],[206,480],[206,639],[206,273],[206,387],[206,974],[206,1088],[206,516],[206,794],[206,468],[206,503],[206,231],[206,525],[206,805],[206,391],[206,453],[206,427],[206,396],[206,474],[207,208],[207,209],[207,210],[207,211],[207,212],[207,213],[207,214],[207,537],[208,377],[208,211],[208,214],[208,213],[208,557],[208,476],[208,395],[208,584],[208,819],[208,600],[208,686],[208,210],[208,514],[208,440],[208,1060],[208,385],[208,474],[209,215],[209,262],[209,268],[209,297],[209,270],[209,467],[209,518],[209,606],[209,220],[209,453],[209,664],[209,230],[209,211],[209,218],[209,427],[209,532],[209,514],[209,219],[209,667],[209,269],[209,495],[209,488],[209,377],[209,624],[209,392],[209,584],[209,557],[209,231],[209,452],[209,605],[209,388],[209,668],[209,433],[209,541],[209,217],[209,225],[209,221],[209,698],[209,394],[209,214],[209,213],[209,223],[209,596],[209,623],[209,661],[209,765],[209,519],[209,673],[209,583],[209,656],[209,407],[209,876],[209,505],[209,475],[209,600],[209,762],[209,468],[209,523],[209,436],[209,759],[209,663],[209,520],[209,710],[209,631],[209,385],[209,527],[209,398],[209,1170],[209,704],[210,270],[210,214],[210,392],[210,624],[210,221],[210,532],[210,476],[210,656],[210,530],[210,391],[210,794],[210,1221],[211,297],[211,270],[211,452],[211,395],[211,377],[211,606],[211,220],[211,698],[211,230],[211,219],[211,213],[211,596],[211,454],[211,584],[211,789],[211,659],[211,600],[211,686],[211,216],[211,514],[211,1114],[211,527],[211,417],[212,247],[212,270],[212,401],[212,469],[212,606],[212,427],[212,558],[212,225],[212,223],[212,698],[212,274],[212,623],[212,766],[212,386],[212,789],[212,561],[212,475],[212,952],[212,436],[212,575],[212,356],[212,417],[213,214],[213,220],[213,600],[213,709],[213,452],[213,527],[213,476],[213,417],[213,534],[213,504],[213,519],[213,710],[213,457],[213,454],[213,557],[213,427],[213,377],[213,584],[213,663],[214,377],[214,220],[214,221],[214,698],[214,655],[214,476],[214,265],[214,596],[214,557],[214,530],[214,274],[214,231],[214,223],[214,452],[214,532],[214,468],[214,457],[214,417],[214,454],[214,453],[214,623],[214,661],[214,269],[214,508],[214,584],[214,765],[214,583],[214,876],[214,475],[214,946],[214,686],[214,498],[214,408],[214,759],[214,514],[214,1078],[214,739],[214,527],[214,474],[215,216],[215,217],[215,218],[215,219],[215,220],[215,221],[215,222],[215,223],[215,224],[215,225],[215,226],[215,227],[215,228],[215,229],[215,230],[215,231],[215,232],[215,233],[216,286],[216,219],[216,706],[216,474],[216,573],[216,230],[216,452],[216,1027],[216,222],[216,535],[216,536],[216,604],[217,624],[217,519],[217,668],[217,475],[217,674],[217,266],[217,218],[217,220],[217,222],[217,663],[217,377],[217,427],[217,225],[217,436],[217,604],[217,502],[217,1060],[217,532],[218,536],[218,661],[218,766],[218,519],[218,668],[218,475],[218,433],[218,768],[218,436],[218,604],[218,393],[218,1115],[218,1129],[218,527],[218,574],[218,534],[218,722],[219,262],[219,467],[219,518],[219,571],[219,606],[219,622],[219,628],[219,664],[219,584],[219,763],[219,519],[219,780],[219,407],[219,686],[219,498],[219,1000],[219,569],[219,604],[219,228],[219,1062],[219,502],[219,1060],[219,598],[219,710],[219,729],[219,1160],[219,278],[219,527],[219,720],[219,1170],[219,1088],[219,704],[219,417],[219,1059],[220,467],[220,469],[220,518],[220,582],[220,377],[220,622],[220,655],[220,454],[220,225],[220,327],[220,221],[220,556],[220,457],[220,624],[220,557],[220,286],[220,541],[220,623],[220,453],[220,470],[220,656],[220,680],[220,573],[220,698],[220,596],[220,661],[220,269],[220,584],[220,519],[220,506],[220,399],[220,583],[220,874],[220,767],[220,475],[220,599],[220,600],[220,498],[220,523],[220,436],[220,663],[220,502],[220,710],[220,1086],[220,1020],[220,385],[220,972],[220,720],[220,1170],[220,482],[220,474],[220,704],[220,1213],[221,432],[221,222],[221,683],[221,476],[221,474],[221,439],[221,596],[221,274],[221,730],[221,392],[221,765],[221,583],[221,656],[221,475],[221,762],[221,433],[221,498],[221,468],[221,408],[221,575],[221,604],[221,663],[221,834],[221,557],[221,538],[221,1133],[221,385],[221,1162],[221,527],[221,391],[221,532],[221,1166],[221,536],[221,704],[222,432],[222,405],[222,452],[222,469],[222,587],[222,563],[222,606],[222,266],[222,540],[222,286],[222,576],[222,638],[222,679],[222,661],[222,584],[222,495],[222,392],[222,519],[222,666],[222,656],[222,861],[222,721],[222,505],[222,501],[222,488],[222,600],[222,498],[222,517],[222,436],[222,1049],[222,604],[222,857],[222,479],[222,565],[222,1129],[222,710],[222,562],[222,406],[222,862],[222,1218],[222,667],[223,270],[223,398],[223,225],[223,241],[223,535],[223,475],[223,265],[223,604],[223,663],[223,1129],[223,710],[223,385],[223,391],[224,435],[224,469],[224,587],[224,427],[224,491],[224,472],[224,470],[224,519],[224,666],[224,760],[224,478],[224,407],[224,965],[224,468],[224,497],[224,436],[224,604],[224,710],[224,496],[225,298],[225,259],[225,270],[225,242],[225,243],[225,435],[225,451],[225,346],[225,582],[225,377],[225,607],[225,612],[225,619],[225,642],[225,427],[225,677],[225,696],[225,697],[225,698],[225,394],[225,707],[225,750],[225,333],[225,519],[225,788],[225,789],[225,795],[225,798],[225,876],[225,548],[225,921],[225,929],[225,902],[225,950],[225,958],[225,425],[225,441],[225,254],[225,977],[225,620],[225,323],[225,313],[225,256],[225,292],[225,252],[225,362],[225,258],[225,341],[225,1034],[225,466],[225,255],[225,1002],[225,1110],[225,1115],[225,1117],[225,618],[225,1126],[225,356],[225,1144],[225,398],[225,972],[225,720],[225,1168],[225,448],[225,289],[225,532],[225,1178],[225,898],[225,1042],[225,1132],[225,1187],[225,1192],[225,737],[225,1198],[225,549],[225,474],[225,1200],[225,1074],[225,1209],[225,374],[225,822],[225,1220],[226,432],[226,469],[226,539],[226,628],[226,634],[226,491],[226,624],[226,473],[226,519],[226,666],[226,797],[226,528],[226,407],[226,947],[226,949],[226,686],[226,569],[226,604],[226,492],[226,1062],[226,502],[226,710],[226,230],[226,1059],[226,879],[227,409],[227,581],[228,1059],[228,240],[229,262],[229,587],[229,266],[229,766],[229,528],[229,475],[229,265],[229,663],[229,264],[229,509],[230,606],[230,491],[230,580],[230,581],[230,493],[230,498],[230,604],[230,663],[230,502],[230,356],[230,278],[230,485],[230,417],[231,270],[231,531],[231,582],[231,587],[231,698],[231,664],[231,392],[231,765],[231,399],[231,501],[231,475],[231,762],[231,468],[231,436],[231,604],[231,530],[231,391],[231,710],[231,527],[231,482],[231,1213],[232,298],[232,248],[232,243],[232,337],[232,550],[232,434],[232,689],[232,696],[232,741],[232,329],[232,792],[232,379],[232,866],[232,829],[232,892],[232,358],[232,461],[232,548],[232,922],[232,331],[232,958],[232,302],[232,425],[232,968],[232,323],[232,359],[232,256],[232,322],[232,345],[232,292],[232,990],[232,255],[232,422],[232,311],[232,342],[232,771],[232,316],[232,421],[232,1007],[232,972],[232,344],[232,1178],[233,298],[233,365],[233,242],[233,434],[233,299],[233,642],[233,696],[233,481],[233,791],[233,652],[233,306],[233,880],[233,893],[233,358],[233,302],[233,425],[233,371],[233,987],[233,258],[233,250],[233,860],[233,324],[233,246],[233,289],[233,355],[233,320],[233,256],[233,249],[233,364],[233,362],[233,345],[233,322],[233,244],[233,368],[233,842],[233,316],[233,349],[233,972],[233,301],[233,374],[234,288],[234,247],[234,617],[234,769],[234,379],[234,827],[234,987],[234,235],[234,239],[234,1008],[234,422],[234,842],[234,316],[234,356],[234,1140],[234,720],[234,290],[234,1017],[234,1132],[234,991],[235,586],[235,692],[235,694],[235,236],[235,779],[235,855],[235,922],[235,351],[235,324],[235,290],[235,245],[235,252],[235,244],[235,462],[235,693],[235,1140],[235,332],[235,448],[235,374],[236,288],[236,459],[236,694],[236,282],[236,868],[236,358],[236,902],[236,351],[236,239],[236,252],[236,915],[236,1034],[236,316],[236,290],[236,1173],[236,1132],[236,1073],[237,298],[237,343],[237,607],[237,650],[237,698],[237,333],[237,855],[237,951],[237,958],[237,425],[237,323],[237,292],[237,250],[237,258],[237,648],[237,255],[237,1113],[237,618],[237,1143],[237,391],[238,694],[238,817],[238,893],[238,351],[238,256],[238,253],[238,290],[239,275],[239,248],[239,707],[239,741],[239,744],[239,333],[239,852],[239,447],[239,403],[239,302],[239,982],[239,324],[239,256],[239,355],[239,304],[239,282],[239,569],[239,255],[239,842],[239,421],[239,290],[239,1017],[239,1011],[239,375],[239,310],[240,245],[240,488],[240,468],[240,1029],[240,663],[240,1069],[240,1060],[240,1132],[240,474],[241,242],[241,434],[241,707],[241,741],[241,246],[241,875],[241,331],[241,302],[241,308],[241,316],[241,421],[241,1165],[241,1178],[241,419],[241,369],[242,259],[242,419],[242,282],[242,420],[242,421],[242,422],[242,289],[242,345],[242,251],[242,359],[242,308],[242,323],[242,256],[242,290],[242,257],[242,311],[242,355],[242,423],[242,424],[242,342],[242,314],[242,425],[242,254],[242,366],[242,696],[242,741],[242,333],[242,255],[242,349],[243,426],[243,246],[243,419],[243,306],[243,362],[243,551],[243,610],[243,434],[243,645],[243,687],[243,352],[243,741],[243,791],[243,548],[243,254],[243,256],[243,281],[243,345],[243,255],[243,422],[243,311],[243,360],[243,1138],[243,905],[244,434],[244,333],[244,347],[244,323],[244,256],[244,255],[244,426],[244,282],[244,318],[244,549],[245,298],[245,247],[245,426],[245,617],[245,736],[245,741],[245,333],[245,798],[245,306],[245,855],[245,447],[245,548],[245,928],[245,254],[245,323],[245,1032],[245,313],[245,335],[245,322],[245,292],[245,674],[245,255],[245,462],[245,311],[245,316],[245,1140],[245,421],[245,972],[245,290],[246,298],[246,337],[246,551],[246,434],[246,630],[246,802],[246,866],[246,548],[246,958],[246,425],[246,444],[246,345],[246,253],[246,292],[246,250],[246,258],[246,1002],[246,311],[246,342],[246,1165],[246,344],[246,1075],[247,306],[247,364],[247,256],[247,382],[247,383],[247,354],[247,384],[247,248],[247,346],[247,343],[247,741],[247,329],[247,795],[247,548],[247,425],[247,444],[247,323],[247,322],[247,292],[247,250],[247,915],[247,255],[247,1002],[247,336],[248,427],[248,257],[248,323],[248,250],[248,354],[248,419],[248,346],[248,299],[248,741],[248,333],[248,306],[248,779],[248,302],[248,256],[248,335],[248,322],[248,1029],[248,258],[248,255],[248,842],[248,1075],[248,375],[249,259],[249,365],[249,650],[249,696],[249,741],[249,333],[249,329],[249,447],[249,548],[249,302],[249,425],[249,371],[249,254],[249,323],[249,314],[249,335],[249,322],[249,355],[249,292],[249,252],[249,250],[249,1029],[249,255],[249,312],[249,1002],[249,311],[249,618],[249,336],[249,999],[249,421],[249,289],[249,1192],[250,365],[250,348],[250,650],[250,744],[250,333],[250,817],[250,779],[250,334],[250,425],[250,371],[250,254],[250,305],[250,256],[250,322],[250,345],[250,355],[250,292],[250,446],[250,346],[250,308],[250,419],[250,277],[250,252],[250,479],[250,255],[250,316],[250,421],[250,1192],[251,288],[251,687],[251,741],[251,744],[251,333],[251,329],[251,779],[251,1061],[251,255],[251,462],[251,373],[251,1128],[251,289],[251,1220],[252,696],[252,741],[252,744],[252,333],[252,791],[252,652],[252,358],[252,548],[252,425],[252,371],[252,254],[252,323],[252,322],[252,355],[252,384],[252,921],[252,296],[252,282],[252,336],[252,304],[252,359],[252,463],[252,362],[252,255],[252,311],[252,421],[253,777],[253,779],[253,425],[253,582],[253,301],[254,426],[254,348],[254,694],[254,581],[254,723],[254,741],[254,757],[254,333],[254,306],[254,779],[254,258],[254,320],[254,322],[254,403],[254,345],[254,304],[254,335],[254,972],[254,674],[254,360],[254,256],[254,575],[254,724],[254,973],[254,835],[254,317],[254,351],[254,313],[254,255],[254,316],[255,365],[255,548],[255,256],[255,345],[255,864],[255,257],[255,282],[255,346],[255,328],[255,362],[255,426],[255,289],[255,324],[255,419],[255,343],[255,308],[255,323],[255,304],[255,354],[255,355],[255,339],[255,314],[255,292],[255,427],[255,439],[255,424],[255,322],[255,1073],[255,773],[255,428],[255,466],[255,422],[255,301],[255,457],[255,770],[255,744],[255,302],[255,434],[255,352],[255,374],[255,678],[255,915],[255,1074],[255,357],[255,259],[255,306],[255,360],[255,601],[255,1075],[255,643],[255,882],[255,325],[255,344],[255,364],[255,316],[255,290],[255,1011],[256,365],[256,346],[256,741],[256,333],[256,329],[256,379],[256,306],[256,548],[256,331],[256,302],[256,323],[256,334],[256,348],[256,305],[256,301],[256,345],[256,352],[256,335],[256,366],[256,360],[256,318],[256,308],[256,364],[256,277],[256,322],[256,362],[256,354],[256,421],[256,445],[256,419],[256,276],[256,320],[256,339],[256,324],[256,356],[256,316],[256,257],[256,282],[256,1060],[257,259],[257,354],[257,377],[257,434],[257,750],[257,786],[257,358],[257,922],[257,902],[257,943],[257,302],[257,309],[257,324],[257,335],[257,322],[257,292],[257,341],[257,1005],[257,601],[257,462],[257,321],[257,360],[257,421],[257,391],[257,972],[257,367],[257,1186],[257,822],[258,261],[258,298],[258,259],[258,595],[258,299],[258,617],[258,650],[258,694],[258,775],[258,366],[258,893],[258,897],[258,902],[258,943],[258,351],[258,323],[258,1029],[258,864],[258,648],[258,1067],[258,1070],[258,462],[258,1008],[258,422],[258,1113],[258,842],[258,349],[258,1147],[258,904],[258,972],[258,1168],[258,448],[258,1178],[258,1187],[258,536],[258,374],[258,1073],[258,860],[259,277],[259,310],[259,311],[259,312],[259,304],[259,313],[259,314],[259,315],[259,434],[259,299],[259,741],[259,333],[259,347],[259,379],[259,880],[259,548],[259,302],[259,335],[259,355],[259,316],[259,349],[259,1140],[259,421],[259,367],[259,1178],[259,375],[260,436],[262,263],[262,264],[262,265],[262,266],[262,267],[263,606],[263,636],[263,575],[263,857],[263,576],[263,710],[264,606],[264,266],[264,265],[264,663],[264,527],[264,453],[264,605],[264,272],[265,270],[265,455],[265,377],[265,491],[265,655],[265,623],[265,743],[265,269],[265,470],[265,266],[265,766],[265,399],[265,528],[265,656],[265,475],[265,686],[265,512],[265,936],[265,476],[265,761],[265,581],[265,439],[265,663],[265,453],[265,1038],[265,436],[265,759],[265,1067],[265,710],[265,1162],[265,527],[265,398],[265,391],[265,415],[265,485],[265,592],[266,518],[266,587],[266,606],[266,628],[266,573],[266,698],[266,623],[266,453],[266,766],[266,507],[266,528],[266,656],[266,475],[266,705],[266,433],[266,498],[266,436],[266,663],[266,1067],[266,1027],[266,565],[266,1045],[266,710],[266,527],[266,1170],[267,531],[268,269],[268,270],[268,271],[268,272],[268,273],[268,274],[268,560],[268,661],[269,698],[269,457],[269,730],[269,569],[269,558],[269,377],[269,453],[269,573],[269,468],[269,727],[269,675],[269,410],[269,475],[269,720],[270,385],[270,386],[270,377],[270,387],[270,388],[270,389],[270,390],[270,391],[270,297],[270,392],[270,393],[270,394],[270,395],[270,396],[270,397],[270,398],[270,399],[270,400],[270,518],[270,698],[270,596],[270,626],[270,407],[270,468],[270,538],[270,710],[271,560],[271,535],[271,501],[271,533],[271,512],[271,440],[271,330],[272,556],[272,557],[272,766],[272,797],[272,540],[272,524],[272,604],[272,440],[272,527],[272,536],[273,850],[273,440],[273,573],[273,453],[273,572],[273,286],[273,1133],[273,452],[273,557],[274,476],[274,453],[274,480],[274,726],[274,573],[274,656],[274,727],[274,623],[274,626],[274,475],[274,512],[274,674],[274,1029],[274,514],[274,336],[275,276],[275,277],[276,426],[276,550],[276,434],[276,299],[276,333],[276,329],[276,798],[276,818],[276,306],[276,358],[276,943],[276,277],[276,773],[276,336],[276,1004],[276,1155],[276,820],[277,295],[277,365],[277,426],[277,434],[277,299],[277,348],[277,567],[277,711],[277,745],[277,333],[277,347],[277,777],[277,802],[277,818],[277,827],[277,306],[277,859],[277,445],[277,447],[277,895],[277,461],[277,933],[277,943],[277,770],[277,425],[277,995],[277,816],[277,1032],[277,1061],[277,773],[277,648],[277,1070],[277,909],[277,1002],[277,462],[277,360],[277,336],[277,356],[277,1004],[277,1141],[277,421],[277,1147],[277,820],[277,1186],[277,339],[277,991],[277,822],[277,860],[277,1213],[277,1163],[278,730],[278,604],[279,649],[279,546],[280,547],[280,333],[280,935],[280,1056],[280,301],[281,364],[281,328],[281,419],[282,288],[282,361],[282,593],[282,612],[282,434],[282,650],[282,707],[282,333],[282,357],[282,795],[282,852],[282,821],[282,548],[282,924],[282,950],[282,302],[282,292],[282,674],[282,963],[282,1070],[282,462],[282,422],[282,291],[282,1113],[282,321],[282,618],[282,771],[282,842],[282,1128],[282,972],[282,290],[282,448],[282,905],[282,860],[283,697],[283,880],[283,371],[283,368],[284,467],[284,568],[284,634],[284,649],[284,754],[284,546],[284,600],[284,362],[284,1038],[284,436],[284,1115],[284,1086],[284,301],[284,310],[285,286],[285,287],[286,582],[286,743],[286,537],[286,528],[286,475],[286,433],[286,498],[286,468],[286,436],[286,1027],[286,681],[286,1144],[286,1162],[286,391],[286,417],[287,587],[287,628],[287,658],[287,491],[287,824],[287,876],[287,505],[287,501],[287,488],[287,705],[287,686],[287,1000],[287,1019],[287,436],[287,1027],[287,601],[287,511],[287,356],[287,710],[287,719],[287,532],[287,474],[287,879],[288,289],[288,290],[288,291],[288,292],[288,548],[288,987],[288,695],[288,316],[288,1140],[289,741],[289,744],[289,333],[289,329],[289,836],[289,444],[289,323],[289,313],[289,335],[289,648],[289,1041],[289,316],[289,972],[289,1168],[289,334],[289,549],[290,548],[290,446],[291,333],[291,358],[291,475],[291,902],[291,371],[291,335],[291,372],[291,368],[291,618],[291,316],[291,301],[292,348],[292,708],[292,741],[292,379],[292,652],[292,880],[292,358],[292,425],[292,371],[292,313],[292,335],[292,322],[292,647],[292,325],[292,446],[292,305],[292,419],[292,695],[292,372],[292,368],[292,618],[292,349],[292,301],[292,1167],[293,294],[293,297],[294,437],[294,405],[294,554],[294,522],[294,713],[294,504],[294,661],[294,766],[294,793],[294,870],[294,510],[294,526],[294,944],[294,957],[294,441],[294,555],[294,408],[294,889],[294,681],[294,340],[294,440],[294,739],[294,416],[294,1134],[294,564],[294,1180],[294,1195],[294,794],[294,503],[294,1205],[294,1208],[294,1212],[295,296],[295,1033],[296,539],[296,882],[296,1032],[296,1016],[296,349],[296,999],[296,1147],[296,916],[296,820],[296,1191],[296,1211],[296,424],[297,394],[297,710],[297,534],[298,299],[298,300],[298,301],[298,302],[298,303],[298,304],[298,305],[298,306],[298,307],[298,308],[298,309],[299,329],[299,304],[299,625],[299,347],[299,419],[299,626],[299,308],[299,356],[299,305],[299,627],[299,902],[299,1033],[301,365],[301,714],[301,826],[301,306],[301,548],[301,331],[301,371],[301,376],[301,322],[301,372],[301,368],[301,618],[301,360],[301,316],[301,339],[301,344],[301,1028],[301,345],[301,386],[301,352],[301,818],[301,328],[301,969],[301,305],[301,547],[301,928],[301,604],[302,714],[302,333],[302,347],[302,329],[302,908],[302,359],[302,316],[302,319],[302,317],[302,318],[302,368],[302,355],[302,314],[302,304],[302,371],[302,419],[302,959],[302,426],[302,346],[302,323],[302,376],[302,322],[302,349],[302,907],[302,364],[302,960],[302,324],[302,354],[302,830],[302,313],[302,339],[302,462],[302,778],[302,999],[302,1004],[302,1140],[303,829],[303,345],[304,426],[304,428],[304,348],[304,352],[304,741],[304,333],[304,358],[304,922],[304,902],[304,313],[304,355],[304,312],[304,336],[304,349],[304,448],[305,365],[305,714],[305,347],[305,358],[305,902],[305,331],[305,425],[305,371],[305,376],[305,860],[305,322],[305,372],[305,368],[305,422],[305,618],[305,360],[305,316],[305,336],[305,999],[305,319],[305,339],[305,991],[305,374],[305,307],[306,365],[306,382],[306,646],[306,348],[306,457],[306,744],[306,766],[306,347],[306,379],[306,367],[306,322],[306,841],[306,331],[306,402],[306,324],[306,371],[306,842],[306,336],[306,323],[306,335],[306,356],[306,320],[306,339],[306,843],[306,548],[306,674],[306,314],[306,364],[306,345],[306,344],[306,830],[306,317],[306,349],[306,999],[306,374],[307,968],[307,353],[307,347],[307,328],[307,549],[308,434],[308,692],[308,741],[308,333],[308,795],[308,425],[308,371],[308,695],[308,1070],[308,321],[308,693],[308,367],[309,358],[309,902],[309,351],[309,860],[309,419],[309,446],[310,362],[310,365],[310,314],[310,335],[311,880],[311,548],[311,425],[311,816],[311,323],[311,422],[311,359],[311,355],[311,314],[312,439],[312,972],[313,880],[313,402],[313,324],[313,335],[313,359],[313,318],[313,322],[313,334],[313,427],[313,341],[313,972],[314,365],[314,346],[314,649],[314,694],[314,741],[314,742],[314,333],[314,347],[314,777],[314,546],[314,779],[314,371],[314,351],[314,323],[314,362],[314,335],[314,355],[314,320],[314,1002],[314,316],[314,1140],[314,972],[314,448],[314,1017],[314,367],[314,344],[314,1011],[314,375],[315,365],[316,361],[316,348],[316,333],[316,347],[316,331],[316,317],[316,322],[316,345],[316,359],[316,319],[316,318],[316,346],[316,329],[316,330],[316,323],[316,332],[316,349],[316,334],[316,335],[316,351],[316,321],[316,352],[316,336],[316,353],[316,354],[316,337],[316,355],[316,356],[316,320],[316,338],[316,339],[316,357],[316,341],[316,358],[316,342],[316,360],[316,343],[316,364],[316,344],[316,362],[316,532],[316,1191],[317,741],[317,652],[317,358],[317,874],[317,556],[318,365],[318,777],[318,779],[318,856],[318,880],[318,334],[318,893],[318,548],[318,902],[318,331],[318,360],[318,972],[319,358],[319,902],[319,341],[319,894],[319,907],[319,375],[319,335],[319,417],[320,365],[320,649],[320,694],[320,333],[320,329],[320,546],[320,652],[320,827],[320,880],[320,937],[320,351],[320,323],[320,324],[320,362],[320,335],[320,322],[320,355],[320,999],[320,1004],[320,972],[320,448],[321,434],[321,567],[321,465],[321,915],[321,342],[321,1118],[321,325],[321,424],[321,373],[321,904],[321,357],[321,344],[321,446],[321,419],[322,365],[322,434],[322,714],[322,826],[322,548],[322,331],[322,371],[322,376],[322,323],[322,614],[322,419],[322,354],[322,368],[322,328],[322,362],[322,345],[322,446],[322,880],[322,364],[322,374],[322,339],[322,734],[322,613],[322,369],[322,372],[322,618],[322,1140],[323,365],[323,741],[323,333],[323,379],[323,447],[323,939],[323,403],[323,359],[323,346],[323,324],[323,354],[323,355],[323,972],[323,356],[323,448],[323,362],[323,335],[323,695],[323,1002],[323,462],[323,1117],[323,349],[323,1004],[323,1140],[323,670],[323,332],[323,367],[324,365],[324,741],[324,742],[324,744],[324,347],[324,379],[324,331],[324,356],[324,1029],[324,354],[324,463],[324,335],[324,998],[324,368],[324,1002],[324,462],[324,1004],[324,1140],[324,332],[324,720],[324,448],[324,367],[324,1132],[324,375],[324,860],[325,694],[325,351],[325,999],[326,680],[326,617],[326,707],[326,422],[327,435],[327,793],[327,668],[327,947],[327,604],[327,889],[327,663],[327,557],[327,397],[327,385],[327,592],[328,741],[328,855],[328,893],[328,548],[328,902],[328,371],[328,345],[328,372],[328,368],[328,618],[328,549],[328,1073],[329,343],[329,361],[329,434],[329,567],[329,770],[329,771],[329,424],[329,360],[329,444],[329,334],[329,352],[329,445],[329,419],[329,772],[329,421],[329,773],[329,357],[329,379],[329,915],[329,618],[329,374],[330,963],[331,359],[331,364],[331,362],[331,348],[331,365],[331,335],[331,366],[331,352],[331,360],[331,334],[331,345],[332,798],[332,447],[332,462],[332,706],[333,346],[333,347],[333,365],[333,419],[333,335],[333,354],[333,355],[333,345],[333,362],[334,365],[334,856],[334,345],[334,648],[334,360],[334,1140],[334,972],[334,1017],[335,365],[335,346],[335,348],[335,379],[335,779],[335,362],[335,364],[335,402],[335,360],[335,380],[335,345],[335,674],[335,368],[335,999],[335,356],[335,421],[335,972],[335,448],[336,608],[336,841],[336,358],[336,548],[336,345],[336,427],[336,402],[336,939],[336,457],[336,676],[336,445],[336,1001],[336,687],[336,374],[336,456],[336,426],[336,364],[336,344],[336,1215],[337,359],[337,434],[338,601],[338,999],[339,854],[339,358],[339,902],[339,371],[339,341],[339,372],[339,368],[339,422],[339,618],[339,860],[340,505],[340,936],[340,1053],[340,557],[340,410],[340,700],[340,441],[340,503],[341,1034],[342,941],[342,904],[342,553],[342,1137],[343,548],[343,374],[343,549],[343,741],[344,457],[344,882],[344,829],[344,945],[344,883],[344,1155],[344,1175],[344,913],[344,402],[344,830],[344,419],[344,1215],[345,741],[345,779],[345,445],[345,548],[345,921],[345,359],[345,1014],[345,365],[345,386],[345,352],[345,366],[345,348],[345,1056],[345,421],[345,822],[346,443],[346,694],[346,741],[346,744],[346,347],[346,967],[346,351],[346,355],[346,1002],[346,1140],[346,448],[346,1017],[347,741],[347,349],[347,426],[347,358],[348,365],[348,369],[348,402],[348,647],[348,648],[348,386],[348,360],[349,859],[349,358],[349,461],[349,902],[349,425],[349,362],[349,708],[349,860],[350,464],[350,465],[350,466],[351,567],[351,694],[351,359],[351,355],[351,352],[351,695],[351,625],[351,915],[352,567],[352,694],[352,876],[352,358],[352,902],[352,360],[352,421],[352,1215],[354,355],[354,606],[354,649],[354,741],[354,546],[354,839],[354,856],[354,548],[354,955],[354,425],[354,1002],[354,1186],[355,426],[355,694],[355,741],[355,744],[355,775],[355,798],[355,425],[355,987],[355,362],[355,419],[355,972],[356,581],[356,839],[356,358],[356,902],[356,569],[356,1080],[356,618],[356,805],[356,1003],[356,972],[356,383],[356,433],[356,1140],[356,375],[357,374],[358,359],[358,898],[358,822],[358,625],[358,465],[358,758],[358,899],[358,783],[358,567],[358,365],[358,614],[358,900],[358,901],[358,902],[358,612],[358,903],[358,904],[358,905],[358,906],[358,907],[358,611],[358,626],[358,434],[358,908],[358,361],[358,909],[358,910],[358,911],[358,912],[358,913],[358,914],[358,915],[358,461],[358,821],[358,447],[358,844],[358,360],[358,860],[358,916],[358,859],[358,708],[359,365],[359,426],[359,694],[359,696],[359,711],[359,839],[359,852],[359,548],[359,933],[359,902],[359,425],[359,816],[359,1056],[359,625],[359,368],[359,1002],[359,402],[359,618],[359,1004],[359,1140],[359,758],[359,367],[359,991],[359,1202],[359,860],[359,380],[360,365],[360,902],[360,983],[360,366],[361,434],[361,465],[361,567],[361,445],[361,902],[362,672],[362,696],[362,741],[362,775],[362,827],[362,548],[362,368],[362,1002],[362,1113],[362,999],[362,1140],[362,972],[362,448],[362,1011],[362,1074],[362,860],[363,880],[363,422],[364,698],[364,707],[364,695],[364,1070],[364,382],[364,842],[365,366],[365,367],[365,368],[365,369],[365,370],[365,371],[365,372],[365,373],[365,374],[365,375],[365,376],[365,377],[365,378],[365,379],[365,380],[365,714],[365,741],[365,652],[365,902],[365,990],[367,669],[367,928],[367,998],[367,1174],[367,463],[368,434],[368,826],[368,902],[368,371],[368,372],[368,969],[368,618],[368,970],[368,648],[368,650],[368,376],[368,697],[369,649],[369,671],[369,617],[369,684],[369,689],[369,546],[369,808],[369,811],[369,958],[369,425],[369,620],[369,1032],[369,1111],[369,842],[369,1192],[371,714],[371,826],[371,446],[371,454],[371,560],[371,969],[371,618],[371,372],[371,434],[371,970],[371,648],[371,650],[371,697],[371,376],[372,826],[372,996],[372,969],[372,618],[372,650],[372,714],[372,972],[373,567],[373,725],[373,548],[373,553],[373,1123],[374,434],[374,707],[374,457],[374,880],[374,421],[374,1074],[374,428],[374,453],[374,1215],[375,643],[376,714],[376,880],[376,618],[377,518],[377,563],[377,558],[377,457],[377,584],[377,606],[377,491],[377,476],[377,731],[377,470],[377,519],[377,561],[377,399],[377,475],[377,433],[377,498],[377,436],[377,759],[377,663],[377,514],[377,422],[377,1060],[377,710],[377,391],[377,720],[377,1170],[377,485],[377,474],[377,536],[377,704],[377,417],[379,741],[379,814],[379,815],[379,816],[379,625],[380,674],[380,1138],[380,402],[383,581],[383,548],[383,937],[383,527],[384,1026],[385,661],[385,475],[385,440],[385,476],[385,557],[385,405],[385,387],[385,631],[385,512],[386,395],[386,399],[386,398],[387,576],[388,676],[388,394],[388,789],[390,929],[390,415],[391,491],[391,623],[391,743],[391,470],[391,519],[391,721],[391,484],[391,475],[391,468],[391,408],[391,663],[391,1060],[391,476],[391,655],[391,686],[391,427],[391,453],[391,494],[391,639],[391,439],[391,396],[391,485],[391,474],[392,396],[392,573],[392,468],[392,710],[392,439],[392,433],[392,528],[392,398],[393,507],[393,1062],[393,681],[393,1129],[394,398],[395,457],[395,623],[395,497],[395,759],[395,604],[395,514],[395,398],[395,482],[395,722],[396,634],[396,491],[396,495],[396,760],[396,484],[396,468],[396,575],[396,759],[396,1116],[396,1166],[396,1087],[397,401],[397,433],[397,410],[397,491],[397,557],[397,655],[397,398],[398,1139],[398,399],[398,400],[399,491],[399,717],[399,663],[400,581],[400,560],[401,402],[401,403],[401,404],[401,405],[401,406],[401,407],[401,408],[401,409],[401,410],[401,411],[401,412],[401,413],[401,414],[401,415],[401,416],[401,417],[401,418],[401,766],[401,782],[401,794],[402,434],[402,711],[402,403],[402,1014],[402,972],[403,880],[403,972],[405,439],[405,440],[405,441],[405,721],[405,475],[406,712],[406,439],[406,474],[406,713],[406,739],[407,469],[407,472],[407,470],[407,473],[407,717],[407,433],[407,446],[407,468],[408,571],[408,510],[408,475],[408,597],[408,417],[408,439],[408,432],[409,475],[409,527],[410,573],[410,661],[410,858],[410,869],[410,510],[410,475],[410,498],[410,996],[410,468],[410,436],[410,1027],[410,1129],[410,710],[410,503],[410,592],[410,722],[411,659],[411,439],[413,493],[415,439],[415,946],[415,558],[416,439],[417,504],[417,861],[417,870],[417,505],[417,510],[417,533],[417,871],[417,600],[417,957],[417,512],[417,936],[417,890],[417,501],[417,495],[417,601],[417,739],[417,572],[417,686],[417,457],[417,1177],[417,836],[418,782],[419,551],[419,612],[419,434],[419,799],[419,802],[419,868],[419,875],[419,821],[419,963],[419,695],[419,1064],[419,1070],[419,422],[419,1115],[419,999],[419,1004],[419,1007],[419,1012],[419,1178],[419,1189],[419,899],[421,434],[421,567],[421,773],[421,820],[421,778],[421,770],[421,445],[421,426],[422,968],[422,457],[422,625],[422,1081],[422,999],[422,1033],[422,1182],[424,882],[424,1137],[425,696],[425,962],[425,963],[425,740],[426,434],[426,728],[426,741],[426,777],[426,626],[426,855],[426,1056],[426,1070],[426,909],[426,842],[426,1004],[426,1147],[426,822],[427,518],[427,587],[427,662],[427,476],[427,600],[427,433],[427,497],[427,569],[427,436],[427,1027],[427,1115],[427,717],[427,538],[427,720],[427,474],[427,536],[427,879],[428,1070],[428,820],[428,1073],[430,431],[432,433],[432,965],[432,879],[433,467],[433,469],[433,550],[433,563],[433,491],[433,706],[433,715],[433,495],[433,800],[433,824],[433,656],[433,921],[433,475],[433,974],[433,468],[433,446],[433,436],[433,1001],[434,444],[434,445],[434,616],[434,617],[434,618],[434,567],[434,902],[434,1215],[435,436],[436,582],[436,491],[436,706],[436,476],[436,656],[436,947],[436,525],[436,663],[436,667],[436,558],[436,475],[436,727],[436,520],[436,584],[436,596],[436,631],[436,1171],[437,438],[437,731],[437,512],[437,739],[439,469],[439,518],[439,607],[439,476],[439,715],[439,470],[439,584],[439,495],[439,774],[439,784],[439,791],[439,506],[439,793],[439,800],[439,494],[439,808],[439,760],[439,824],[439,840],[439,849],[439,867],[439,876],[439,887],[439,510],[439,484],[439,927],[439,938],[439,947],[439,949],[439,605],[439,954],[439,964],[439,516],[439,978],[439,996],[439,1019],[439,555],[439,1044],[439,1049],[439,575],[439,889],[439,594],[439,1078],[439,1008],[439,1093],[439,1116],[439,1060],[439,997],[439,1045],[439,1146],[439,1148],[439,1159],[439,1007],[439,720],[439,782],[439,1166],[439,1042],[439,480],[439,474],[439,525],[439,888],[440,507],[440,686],[440,512],[440,524],[440,794],[440,468],[440,847],[440,525],[440,500],[440,920],[440,510],[441,505],[444,567],[444,617],[445,567],[445,941],[445,1220],[446,469],[446,740],[446,880],[446,548],[446,921],[446,1040],[446,475],[446,674],[446,1115],[446,842],[446,1030],[446,1129],[446,1004],[446,972],[446,1173],[446,1192],[447,744],[447,798],[447,652],[447,462],[448,652],[448,1041],[448,547],[449,567],[449,707],[449,674],[452,453],[452,454],[452,606],[452,698],[452,508],[452,870],[452,505],[452,475],[452,710],[452,631],[452,592],[453,469],[453,563],[453,491],[453,573],[453,789],[453,760],[453,507],[453,948],[453,759],[453,498],[453,468],[453,575],[453,710],[453,1086],[453,485],[453,474],[453,536],[454,707],[454,475],[454,941],[454,558],[454,479],[455,717],[455,527],[456,457],[456,458],[456,882],[456,1070],[457,557],[457,584],[457,583],[457,880],[457,475],[457,999],[457,474],[457,1215],[458,1070],[459,1033],[460,1127],[461,860],[461,902],[463,1004],[463,1140],[464,465],[464,466],[464,879],[465,567],[465,902],[466,583],[467,468],[467,720],[467,1166],[468,476],[468,731],[468,765],[468,789],[468,793],[468,528],[468,721],[468,533],[468,475],[468,512],[468,532],[468,661],[468,480],[468,659],[468,558],[468,563],[468,937],[468,492],[468,890],[468,710],[468,869],[468,585],[468,493],[468,592],[468,499],[468,952],[468,575],[468,1159],[469,470],[469,471],[469,472],[469,473],[469,663],[470,491],[470,472],[470,471],[470,599],[470,558],[470,576],[470,473],[470,760],[470,535],[470,475],[471,473],[472,716],[472,473],[472,717],[472,718],[472,525],[473,716],[474,518],[474,571],[474,584],[474,519],[474,666],[474,801],[474,861],[474,475],[474,686],[474,965],[474,712],[474,1027],[474,565],[474,1060],[474,717],[474,538],[474,1129],[474,710],[474,562],[474,1144],[474,1162],[474,527],[474,720],[474,485],[474,558],[474,572],[474,476],[474,573],[474,718],[474,537],[474,495],[474,661],[474,560],[474,705],[474,1164],[474,1049],[475,476],[475,495],[475,510],[475,668],[475,700],[475,805],[475,597],[475,558],[475,709],[475,540],[475,889],[475,730],[475,749],[475,484],[475,667],[475,492],[475,659],[475,656],[475,482],[475,759],[475,513],[475,556],[475,485],[475,561],[475,584],[475,739],[475,596],[475,585],[475,632],[475,581],[475,488],[475,712],[475,740],[475,641],[475,774],[475,535],[475,663],[476,661],[476,557],[476,705],[476,491],[476,712],[476,492],[476,527],[476,535],[476,583],[476,876],[476,1027],[476,538],[476,710],[477,710],[477,1020],[478,535],[479,576],[479,572],[480,941],[480,663],[480,710],[480,509],[480,673],[480,525],[480,517],[480,858],[480,580],[481,518],[481,573],[482,929],[482,572],[482,573],[483,563],[483,491],[484,526],[484,575],[484,717],[484,710],[484,1166],[485,491],[485,583],[485,488],[485,704],[488,518],[488,519],[488,673],[488,583],[488,862],[488,942],[488,759],[488,557],[488,530],[488,710],[488,527],[491,585],[491,639],[491,663],[491,664],[491,665],[491,492],[491,659],[491,656],[491,666],[491,557],[491,655],[491,635],[491,536],[491,508],[491,519],[491,789],[491,1157],[491,527],[491,720],[491,722],[492,541],[492,624],[493,743],[493,497],[493,663],[493,710],[494,577],[494,1041],[495,531],[495,573],[495,519],[495,889],[496,760],[496,498],[497,699],[497,572],[497,573],[497,605],[498,581],[498,572],[498,624],[498,558],[498,576],[498,710],[500,587],[500,760],[500,535],[501,535],[501,739],[501,512],[501,674],[501,663],[503,564],[503,661],[504,576],[504,533],[504,600],[505,622],[505,512],[505,639],[505,572],[505,879],[505,533],[507,766],[507,534],[508,572],[509,632],[509,576],[510,638],[510,888],[510,803],[510,889],[510,890],[510,891],[510,511],[510,739],[510,557],[511,766],[511,535],[511,526],[511,576],[512,793],[512,535],[512,533],[512,831],[512,576],[512,557],[512,710],[514,698],[515,573],[515,656],[515,759],[516,557],[516,879],[517,535],[517,557],[518,558],[518,559],[518,560],[518,561],[518,541],[518,557],[518,889],[519,638],[519,572],[519,557],[519,584],[519,600],[519,673],[520,1038],[521,557],[521,535],[522,576],[525,661],[525,743],[525,793],[525,797],[525,673],[525,583],[525,555],[525,663],[526,535],[526,936],[526,879],[527,535],[527,663],[527,558],[527,540],[527,530],[527,896],[527,656],[527,529],[527,706],[527,557],[527,581],[528,766],[528,576],[530,600],[530,1221],[531,562],[531,563],[531,564],[531,565],[531,889],[532,765],[532,573],[533,576],[533,557],[534,932],[535,582],[535,538],[535,597],[535,759],[535,831],[535,600],[535,562],[535,722],[535,634],[535,565],[535,832],[535,833],[535,720],[535,834],[536,556],[536,540],[537,576],[537,557],[538,558],[538,667],[538,573],[538,580],[540,797],[540,604],[541,710],[541,1086],[541,722],[543,544],[546,649],[546,754],[546,650],[546,547],[546,651],[546,652],[546,653],[546,654],[546,852],[546,1192],[547,649],[547,678],[547,744],[547,754],[547,935],[547,976],[547,1142],[548,918],[548,614],[548,612],[549,1032],[551,552],[551,553],[553,595],[553,796],[554,555],[556,583],[556,721],[556,637],[557,639],[557,698],[557,655],[557,661],[557,584],[557,793],[557,656],[557,767],[557,941],[557,762],[557,635],[557,759],[557,572],[557,1122],[557,710],[557,564],[558,563],[558,573],[558,698],[558,789],[558,941],[558,965],[558,576],[558,710],[558,1166],[559,1038],[560,639],[560,655],[560,710],[560,1170],[560,574],[563,592],[565,709],[565,572],[567,670],[567,902],[568,569],[569,581],[572,606],[572,573],[572,723],[572,623],[572,743],[572,763],[572,780],[572,632],[572,583],[572,686],[572,1044],[572,575],[572,604],[572,1067],[572,1045],[572,710],[572,1170],[572,724],[573,638],[573,606],[573,605],[573,575],[573,1129],[573,710],[573,1170],[574,604],[574,1115],[575,639],[575,723],[575,699],[575,655],[575,794],[576,666],[576,698],[576,743],[576,793],[576,640],[576,824],[576,599],[576,947],[576,705],[576,712],[576,663],[576,1067],[576,831],[576,833],[577,721],[578,579],[578,1040],[581,639],[581,710],[582,583],[582,584],[582,585],[582,935],[582,1178],[583,857],[584,762],[584,720],[587,588],[587,589],[587,590],[587,710],[589,604],[592,606],[592,1038],[592,889],[593,594],[594,1065],[594,1066],[596,681],[597,812],[597,1053],[598,661],[598,635],[599,600],[599,833],[600,709],[601,1070],[601,1076],[604,661],[604,766],[604,710],[605,656],[611,791],[611,902],[612,613],[612,614],[612,902],[614,652],[616,626],[617,682],[617,620],[618,826],[618,969],[618,648],[618,650],[619,620],[619,621],[620,754],[620,650],[620,704],[622,623],[622,624],[622,631],[623,729],[623,759],[623,710],[624,1053],[624,889],[624,1162],[624,704],[624,722],[625,694],[625,777],[625,921],[625,902],[625,915],[625,908],[626,820],[626,635],[626,656],[626,652],[626,632],[631,710],[632,847],[632,1129],[632,704],[633,650],[633,902],[634,889],[634,663],[635,639],[635,666],[635,857],[635,1129],[635,738],[636,661],[638,673],[638,674],[638,675],[639,710],[639,794],[640,1129],[642,643],[643,1155],[644,1129],[646,1071],[649,650],[649,651],[649,652],[649,653],[649,654],[650,687],[650,783],[650,811],[650,1127],[650,1192],[650,981],[652,821],[652,822],[654,852],[655,656],[655,767],[655,1122],[656,759],[658,659],[659,824],[660,661],[661,673],[661,738],[661,712],[661,739],[661,765],[661,721],[661,932],[661,762],[661,1122],[663,664],[663,766],[663,722],[663,835],[663,710],[663,702],[663,720],[664,719],[664,720],[664,721],[665,1129],[667,947],[667,1129],[667,710],[667,1171],[673,739],[673,710],[674,726],[675,869],[680,681],[681,889],[681,709],[692,693],[694,695],[695,915],[695,905],[697,880],[698,699],[698,700],[699,710],[700,789],[700,717],[700,794],[705,812],[705,1020],[706,1178],[708,902],[708,1130],[710,793],[710,888],[710,794],[710,724],[710,1136],[710,782],[710,760],[717,957],[719,1167],[720,803],[721,871],[721,872],[721,862],[722,760],[723,724],[726,1129],[729,862],[729,1047],[731,732],[731,733],[734,735],[736,738],[739,840],[744,1140],[757,758],[758,902],[758,1033],[759,760],[759,1052],[760,804],[763,764],[764,1043],[766,767],[766,768],[770,820],[771,1073],[773,1063],[773,1033],[774,889],[777,778],[777,779],[779,963],[782,849],[783,902],[786,787],[788,1033],[790,999],[793,794],[798,1140],[803,1129],[804,931],[805,1186],[809,810],[820,846],[820,1220],[825,1016],[828,829],[830,1140],[830,1014],[842,900],[844,902],[844,941],[844,1075],[846,902],[850,851],[852,853],[859,860],[859,902],[859,1017],[860,902],[860,999],[860,1017],[860,991],[861,862],[863,864],[880,881],[882,883],[882,1189],[884,885],[885,1055],[885,1208],[893,894],[893,1132],[899,902],[900,902],[901,902],[902,903],[902,904],[902,905],[902,906],[902,907],[902,908],[902,910],[902,911],[902,912],[902,913],[902,915],[902,916],[906,1032],[906,908],[906,1033],[908,1032],[908,1142],[913,1034],[915,1070],[916,937],[926,1166],[934,1023],[955,956],[956,1178],[980,981],[984,985],[984,986],[984,1076],[985,1076],[991,1075],[992,1012],[996,997],[1005,1025],[1015,1026],[1018,1064],[1020,1139],[1024,1025],[1032,1033],[1032,1034],[1032,1142],[1033,1149],[1033,1150],[1033,1151],[1033,1152],[1033,1153],[1033,1154],[1033,1155],[1039,1198],[1044,1045],[1044,1046],[1045,1046],[1070,1071],[1073,1074],[1131,1132],[1132,1187],[1162,1163],[1176,1177],[1178,1179],[1179,1203],[1179,1206],[1184,1185],[1203,1204],[1209,1210]]}"""



# %% [markdown]
# ## Graphs

# %%
def _find_data_dir():
    for cand in ["/kaggle/input/tpoc-lpa-benchmark-networks", "dataset", "../dataset"]:
        if os.path.isfile(os.path.join(cand, "football.gml")):
            return cand
    hits = glob.glob("/kaggle/input/**/football.gml", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("football.gml not found")

DATA = _find_data_dir()

G_foot = nx.read_gml(f"{DATA}/football.gml", label="id")
G_foot = nx.relabel_nodes(G_foot, {n: G_foot.nodes[n]["label"] for n in G_foot.nodes()})
TRUTH_FOOT = {n: G_foot.nodes[n]["value"] for n in G_foot.nodes()}
HINT_FOOT = ("These are US college football teams from the 2000 season. Teams in the "
             "same group played in the same athletic conference.")

G_pol = nx.parse_gml(POLBOOKS_GML)
TRUTH_POL = {n: G_pol.nodes[n]["value"] for n in G_pol.nodes()}
HINT_POL = ("These are books about US politics that were sold on Amazon.com around the "
            "2004 US presidential election. Books in the same group share the same "
            "political leaning (liberal, neutral or conservative).")
_pb = json.loads(POLBLOGS_JSON)
G_blog = nx.Graph()
G_blog.add_nodes_from(_pb["labels"])
G_blog.add_edges_from((_pb["labels"][a], _pb["labels"][b]) for a, b in _pb["edges"])
TRUTH_BLOG = {l: str(v) for l, v in zip(_pb["labels"], _pb["value"])}
HINT_BLOG = ("These are US political blogs (identified by their web address) from around the "
             "2004 US presidential election. Blogs in the same group share the same political "
             "leaning (liberal or conservative).")
log("football:", G_foot.number_of_nodes(), "nodes | polbooks:", G_pol.number_of_nodes(),
    "nodes | polblogs:", G_blog.number_of_nodes(), "nodes")

# %% [markdown]
# ## GPU check (fail fast -- Kaggle silently gives CPU-only sessions otherwise)

# %%
import torch
n_gpu = torch.cuda.device_count()
log(f"GPUs visible: {n_gpu}", [torch.cuda.get_device_name(i) for i in range(n_gpu)])
total_gpu = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpu))
if n_gpu == 0:
    raise SystemExit("NO GPU ALLOCATED")

from transformers import AutoModelForCausalLM, AutoTokenizer

def find_model_dirs():
    """Group attached checkpoints by a short name; largest first within a family."""
    hits = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    out = {}
    for h in hits:
        d = os.path.dirname(h)
        low = d.lower()
        if "qwen" not in low:
            continue
        if not glob.glob(os.path.join(d, "*.safetensors")):
            continue
        if "qwen-3" in low or "qwen3" in low:
            name = "qwen3-" + re.search(r"transformers/([^/]+)", d).group(1)
        else:
            name = "qwen2.5-" + re.search(r"transformers/([^/]+)", d).group(1)
        out[name] = d
    return out

MODEL_DIRS = find_model_dirs()
log("models found:", {k: v.split("/")[-3:] for k, v in MODEL_DIRS.items()})

def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"           # needed so the LAST position is the answer slot
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", low_cpu_mem_usage=True, dtype=torch.float16)
    m.eval()
    dm = getattr(m, "hf_device_map", {}) or {}
    log("  placement:", sorted(set(map(str, dm.values()))) or "single device")
    if any(str(v) in ("cpu", "disk") for v in dm.values()):
        log("  WARNING: some layers offloaded to CPU/disk -- this will be slow")
    return tok, m


def make_letter_probs_fn(tok, model):
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]

    def fn(prompts, ks):
        texts = []
        for p in prompts:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)   # Qwen3: skip <think>
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(t)
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            try:
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
            except TypeError:
                logits = model(**enc).logits[:, -1, :]
        full = torch.softmax(logits.float(), dim=-1)
        out = []
        for b, k in enumerate(ks):
            mass = float(full[b, letter_ids[:k]].sum())         # how much mass is on valid letters
            sub = full[b, letter_ids[:k]]
            sub = (sub / sub.sum().clamp_min(1e-12)).cpu().tolist()
            out.append((sub, mass))
        return out
    return fn

# %% [markdown]
# ## Run

# %%
ALL = {}
for name, path in sorted(MODEL_DIRS.items()):
    log(f"=== model {name} ===")
    t0 = time.time()
    try:
        tok, model = load_model(path)
    except Exception as e:
        log("  load failed:", type(e).__name__, str(e)[:200])
        continue
    log(f"  loaded in {time.time()-t0:.0f}s")
    fn = make_letter_probs_fn(tok, model)
    res = {}
    for dsname, G, truth, hint, variants in [
            ("football", G_foot, TRUTH_FOOT, HINT_FOOT, (6, 12)),
            ("polbooks", G_pol, TRUTH_POL, HINT_POL, (6, 12)),
            ("polblogs", G_blog, TRUTH_BLOG, HINT_BLOG, (12,))]:
        res[dsname] = collect_dataset(dsname, G, truth, hint, fn,
                                      k_members_variants=variants, log=log)
        with open(f"/kaggle/working/collect_{name}.json", "w") as f:   # save after EVERY dataset
            json.dump(res, f)
        log(f"  saved collect_{name}.json with datasets {sorted(res)}")
    ALL[name] = res
    log(f"  model {name} done in {time.time()-t0:.0f}s")
    # `fn` closes over model+tok, so it must be deleted too -- deleting only
    # model/tok left ~15 GB pinned and made the NEXT model spill to CPU and OOM.
    del fn, model, tok
    gc.collect(); torch.cuda.empty_cache()
    free = sum(torch.cuda.mem_get_info(i)[0] for i in range(n_gpu))
    log(f"  GPU memory free after cleanup: {free/1e9:.1f} GB of {total_gpu/1e9:.1f} GB")

log("DONE. models collected:", sorted(ALL))
