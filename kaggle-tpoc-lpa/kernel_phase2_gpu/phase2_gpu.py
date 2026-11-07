# %% [markdown]
# # Phase 2 (GPU) — real local LLM as the Stage 4 responder
#
# The CPU notebook validated the Stage 4 policy with mock responders and
# established the key requirement: with **hard override**, a responder below
# ~90% accuracy actively HURTS (Football −0.02 at p=0.9, −0.27 at p=0.3),
# while **calibrated Bayesian fusion** degrades gracefully to the structural
# baseline instead.
#
# So the decisive question for a *local, quantized, open-weight* model is a
# single number: **what is its actual reliability p on these membership
# questions?** That is what this notebook measures first — before running any
# full experiment — because p determines whether the approach can pay off.
#
# Model: Qwen2.5-7B-Instruct, 4-bit, attached as a Kaggle Model (no internet).
#
# Experiments:
# 1. **Measured reliability** — ask the model on boundary nodes whose ground
#    truth we know; accuracy = true p.
# 2. **Self-consistency calibration** — ask k times with shuffled candidate
#    order and measure agreement. This is a GROUND-TRUTH-FREE estimate of p,
#    which is what you would actually have in deployment. Does it track true p?
#    (Alper hardcodes delta_LLM = 0.95 instead; this replaces that.)
# 3. **End-to-end** — run Stage 4 with the measured p under hard override vs
#    calibrated fusion, against the structural baseline.

# %%
import os, re, json, glob, math, random, time
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity

def _find_data_dir():
    for cand in ["/kaggle/input/tpoc-lpa-benchmark-networks", "dataset", "../dataset"]:
        if os.path.isfile(os.path.join(cand, "football.gml")):
            return cand
    hits = glob.glob("/kaggle/input/**/football.gml", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("football.gml not found under /kaggle/input")

DATA = _find_data_dir()
print("data dir:", DATA)


# %% [markdown]
# ## Phase 1 core + Stage 4 (inlined, unmodified)


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
# ## Load the model (4-bit)

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _bnb_config():
    """4-bit config, or None when bitsandbytes is unavailable (the Kaggle
    image does not always ship a new enough build, and we have no internet
    to install one)."""
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    except Exception as e:
        print("bitsandbytes unavailable -> falling back to fp16:", type(e).__name__)
        return None


def _find_model_dir():
    """Kaggle's mount point for an attached Model varies (plain slug vs a
    /kaggle/input/models/... prefix), so search for the config.json rather
    than hardcoding a path."""
    """All attached Qwen model dirs, largest first (we try the biggest that
    actually loads). Kaggle's mount point varies, so search for config.json."""
    hits = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    dirs = [os.path.dirname(h) for h in hits if "qwen" in h.lower()]
    # prefer bigger checkpoints: sort by total size of the dir's weight files
    def size_of(d):
        return sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*.safetensors")))
    return sorted(set(dirs), key=size_of, reverse=True)

MODEL_DIRS = _find_model_dir()
print("candidate model dirs (largest first):")
for d in MODEL_DIRS:
    print("   ", d)
if not MODEL_DIRS:
    print("!! nothing found. /kaggle/input top level:", glob.glob("/kaggle/input/*"))
    raise SystemExit("model not mounted")

T_START = time.time()

def log(*a):
    """Progress line with wall-clock stamp, flushed immediately so a live
    log viewer shows it."""
    print(f"[{time.time()-T_START:7.1f}s]", *a, flush=True)

# Fail fast if no GPU was actually allocated. Kaggle silently hands out a
# CPU-only session when GPU access is not enabled on the account (it did:
# the first run reported 0 GPUs and spent ~2h generating on CPU).
n_gpu = torch.cuda.device_count()
log(f"GPUs visible: {n_gpu}", [torch.cuda.get_device_name(i) for i in range(n_gpu)])
if n_gpu == 0:
    raise SystemExit("NO GPU ALLOCATED -- refusing to run a 7B model on CPU. "
                     "Check Kaggle account phone verification / accelerator settings.")

bnb = _bnb_config()
model, tokenizer, MODEL_DIR = None, None, None
for cand in MODEL_DIRS:
    try:
        log(f"loading {cand} ...")
        tok = AutoTokenizer.from_pretrained(cand)
        kw = dict(device_map="auto", low_cpu_mem_usage=True, dtype=torch.float16)
        if bnb is not None:
            kw["quantization_config"] = bnb
        m = AutoModelForCausalLM.from_pretrained(cand, **kw)
        m.eval()
        model, tokenizer, MODEL_DIR = m, tok, cand
        dm = getattr(m, "hf_device_map", {}) or {}
        log("LOADED:", cand, "| 4-bit" if bnb is not None else "| fp16",
            "| placement:", sorted(set(map(str, dm.values()))) or "single device")
        break
    except Exception as e:
        log("  failed:", type(e).__name__, str(e)[:160])
        torch.cuda.empty_cache()
if model is None:
    raise SystemExit("no model could be loaded")

# %% [markdown]
# ## Prompting
#
# The question is deliberately phrased in terms of **which set of members a
# node groups with**, not "name the conference" — the algorithm only needs a
# grouping judgement, and that framing does not require the model to recall
# official labels (which for Football are the year-2000 alignments, now stale).

# %%
GEN_CACHE = {}

def _generate(prompt, max_new_tokens=12, temperature=0.0, seed=0):
    key = (prompt, temperature, seed)
    if key in GEN_CACHE:
        return GEN_CACHE[key]
    messages = [
        {"role": "system", "content": "You are an expert at grouping entities. Answer with option letters only."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        if temperature and temperature > 0:
            torch.manual_seed(seed)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=True, temperature=temperature, top_p=0.9)
        else:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    resp = tokenizer.decode(gen, skip_special_tokens=True).strip()
    GEN_CACHE[key] = resp
    return resp


def describe(G, label, comm, exclude, k=6):
    members = [str(n) for n in G.nodes() if label[n] == comm and n != exclude]
    return ", ".join(members[:k]) if members else "(no other members)"


def build_prompt(G, label, node, candidates, domain_hint):
    letters = "ABCDEFGH"
    lines = [f"{domain_hint}", "", f"Item: {node}", "", "Candidate groups:"]
    for idx, c in enumerate(candidates):
        lines.append(f"  {letters[idx]}: {describe(G, label, c, node)}")
    lines.append("")
    lines.append(f"Which ONE group does '{node}' most naturally belong to? "
                 f"Reply with a single letter ({'/'.join(letters[:len(candidates)])}).")
    return "\n".join(lines)


def parse_letters(resp, candidates):
    letters = "ABCDEFGH"[:len(candidates)]
    found = [ch for ch in resp.upper() if ch in letters]
    if not found:
        return set()
    return {candidates[letters.index(found[0])]}


def make_llm_responder(G, domain_hint, temperature=0.0, seed=0, log=None):
    def responder(node, candidates, context):
        label = context["label"]
        prompt = build_prompt(G, label, node, candidates, domain_hint)
        resp = _generate(prompt, temperature=temperature, seed=seed)
        picked = parse_letters(resp, candidates)
        if log is not None:
            log.append({"node": str(node), "raw": resp,
                        "picked": [str(c) for c in picked]})
        return picked
    return responder



# %% [markdown]
# ## Helpers

# %%
def to_partition(label_dict, nodes):
    groups = {}
    for n in nodes:
        groups.setdefault(label_dict[n], set()).add(n)
    return list(groups.values())


def majority_map(truth, label, nodes):
    from collections import Counter, defaultdict
    counts = defaultdict(Counter)
    for n in nodes:
        counts[truth[n]][label[n]] += 1
    return {gt: c.most_common(1)[0][0] for gt, c in counts.items()}


def boundary_nodes(G, label, NI, top_m=3):
    """The nodes Stage 4 would actually consider, with their candidate sets."""
    out = []
    for i in G.nodes():
        dist = was_distribution(G, NI, label, i)
        if len(dist) < 2:
            continue
        cands = [c for c, _ in sorted(dist.items(), key=lambda kv: -kv[1])][:top_m]
        out.append((i, cands, dist))
    return out


G_foot = nx.read_gml(f"{DATA}/football.gml", label="id")
# use readable names, not integer ids, so the model has something to reason about
G_foot = nx.relabel_nodes(G_foot, {n: G_foot.nodes[n]["label"] for n in G_foot.nodes()})
TRUTH_FOOT = {n: G_foot.nodes[n]["value"] for n in G_foot.nodes()}
FOOT_HINT = ("These are US college football teams from the 2000 season. Teams in the "
             "same group played in the same athletic conference.")

# %% [markdown]
# ## Experiment 1 + 2 — measured reliability, and ground-truth-free calibration

# %%
NI_f = node_importance(G_foot)
base_label_f, _, _, _ = tpoc_lpa(G_foot, alpha=1.0, beta=0.8)
nodes_f = list(G_foot.nodes())
y_true_f = [TRUTH_FOOT[n] for n in nodes_f]
base_nmi_f = normalized_mutual_info_score(y_true_f, [base_label_f[n] for n in nodes_f])
gt_to_comm_f = majority_map(TRUTH_FOOT, base_label_f, nodes_f)
print(f"Football structural baseline NMI = {base_nmi_f:.4f}")

cands_f = boundary_nodes(G_foot, base_label_f, NI_f)
print(f"boundary nodes Stage 4 would consider: {len(cands_f)}")

MAX_EVAL = 60
EVAL_BUDGET_S = 20 * 60   # hard wall-clock cap on the reliability measurement
subset = cands_f[:MAX_EVAL]

t0 = time.time()
correct, total = 0, 0
agree_scores = []
per_node = []
for idx, (node, cands, dist) in enumerate(subset):
    if time.time() - t0 > EVAL_BUDGET_S:
        log(f"TIME BUDGET hit after {total} nodes -- stopping early, reporting partial result")
        break
    if idx % 5 == 0:
        rate = (time.time() - t0) / max(idx, 1)
        log(f"reliability eval: node {idx}/{len(subset)}  "
            f"running acc={correct}/{max(total,1)}  ~{rate:.1f}s/node")
    gt_comm = gt_to_comm_f.get(TRUTH_FOOT[node])
    prompt = build_prompt(G_foot, base_label_f, node, cands, FOOT_HINT)
    greedy = parse_letters(_generate(prompt, temperature=0.0), cands)
    is_right = bool(greedy) and (next(iter(greedy)) == gt_comm)
    correct += int(is_right); total += 1

    # self-consistency: same question, shuffled candidate order, sampled
    votes = []
    for s in range(3):
        shuf = cands[:]
        random.Random(s).shuffle(shuf)
        p2 = build_prompt(G_foot, base_label_f, node, shuf, FOOT_HINT)
        r2 = parse_letters(_generate(p2, temperature=0.7, seed=s), shuf)
        if r2:
            votes.append(next(iter(r2)))
    if votes:
        top = max(set(votes), key=votes.count)
        agree_scores.append(votes.count(top) / len(votes))
    per_node.append({"node": str(node), "correct": is_right,
                     "agreement": agree_scores[-1] if agree_scores else None})

measured_p = correct / max(total, 1)
selfcons_p = sum(agree_scores) / len(agree_scores) if agree_scores else float("nan")
print(f"\n  MEASURED reliability (vs ground truth) p = {measured_p:.3f}  ({correct}/{total})")
print(f"  SELF-CONSISTENCY estimate (no ground truth) = {selfcons_p:.3f}")
print(f"  Alper's hardcoded assumption               = 0.950")
print(f"  ({time.time()-t0:.0f}s for {total} nodes x 4 generations)")

# %% [markdown]
# ## Experiment 3 — end to end with the real model

# %%
log = []
resp_llm = make_llm_responder(G_foot, FOOT_HINT, temperature=0.0, log=log)

print(f"{'mode':>28} | {'NMI':>17} | note")
print("-" * 70)
print(f"{'structural baseline':>28} | {base_nmi_f:.4f} ({0.0:+.4f}) | no LLM")

for tag, kw in [("hard override (Alper-style)", dict(fusion="hard")),
                 ("calibrated fusion (ours)", dict(fusion="soft", trust=measured_p)),
                 ("fusion w/ hardcoded 0.95", dict(fusion="soft", trust=0.95))]:
    lab, ov, st = stage4_semantic_correction(
        G_foot, base_label_f, beta=0.8, budget=MAX_EVAL,
        responder=resp_llm, NI=NI_f, **kw)
    nmi = normalized_mutual_info_score(y_true_f, [lab[n] for n in nodes_f])
    print(f"{tag:>28} | {nmi:.4f} ({nmi-base_nmi_f:+.4f}) | queried={st['queried']}, "
          f"changes={st['label_changes']}, overlap={st['overlap_nodes']}")

with open("/kaggle/working/llm_log.json", "w") as f:
    json.dump({"measured_p": measured_p, "selfcons_p": selfcons_p,
               "per_node": per_node, "sample_responses": log[:40]}, f, indent=2)
print("\nwrote /kaggle/working/llm_log.json")
print("\nsample model responses:")
for entry in log[:8]:
    print("  ", entry)
