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
