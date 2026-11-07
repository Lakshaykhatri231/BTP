"""
Phase 2, Stage 4: budgeted semantic correction on top of TPOC-LPA.

Adapts Alper's (KDD'26) adaptive signal-selection idea -- which decides
per-record whether cheap structural signal suffices or whether to spend
budget on an LLM query -- from disjoint entity resolution to OVERLAPPING
community detection.

Two adaptations vs. Alper:
  * the query is MULTI-SELECT (a node may legitimately belong to >1
    community), not "pick the single best match";
  * the budget is a query/compute quota, not a dollar amount.

The LLM is behind a `responder` interface so the whole policy can be
validated with mock responders (oracle / noisy oracle / random) before any
GPU is involved -- in particular, the oracle run measures the HEADROOM:
the best any semantic corrector could possibly do on a given graph.
"""
import math

from tpoc_lpa import node_importance


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
    from tpoc_lpa import community_label_filtering
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
