"""Generate the Phase 2 GPU notebook (real local LLM) by inlining the
validated modules, same approach as the CPU notebook."""
from build_phase2_notebook import strip_module

HEADER = '''# %% [markdown]
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
'''

LLM = '''
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
    return "\\n".join(lines)


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
'''

EXPERIMENTS = '''
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
print(f"\\n  MEASURED reliability (vs ground truth) p = {measured_p:.3f}  ({correct}/{total})")
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
print("\\nwrote /kaggle/working/llm_log.json")
print("\\nsample model responses:")
for entry in log[:8]:
    print("  ", entry)
'''


def main():
    parts = [HEADER]
    parts.append("# %% [markdown]\n# ## Phase 1 core + Stage 4 (inlined, unmodified)\n")
    parts.append("# %%\n" + strip_module("tpoc_lpa.py"))
    parts.append("# %%\n" + strip_module("stage4.py"))
    parts.append(LLM)
    parts.append(EXPERIMENTS)
    script = "\n\n".join(parts)
    with open("kernel_phase2_gpu/phase2_gpu.py", "w") as f:
        f.write(script)
    print("wrote kernel_phase2_gpu/phase2_gpu.py", len(script), "chars")


if __name__ == '__main__':
    main()
