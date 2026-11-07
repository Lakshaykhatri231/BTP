"""Build the GPU evidence-collection notebook.

Inlines the validated Phase 1 core, Stage 4 helpers and the collection logic,
plus the Polbooks graph (inlined, to avoid a dataset-versioning round trip)."""
from build_phase2_notebook import strip_module

POLBOOKS_GML = open("data/polbooks/polbooks.gml").read()
POLBLOGS_JSON = open("data/polblogs/polblogs_lcc.json").read()

HEADER = '''# %% [markdown]
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
'''

GPU = '''
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
'''


def main():
    parts = [HEADER, "# %% [markdown]\n# ## Phase 1 core + Stage 4 helpers (inlined, unmodified)\n",
             "# %%\n" + strip_module("tpoc_lpa.py"),
             "# %%\n" + strip_module("stage4.py"),
             open("collect_body.py").read(),
             f'\n# %%\nPOLBOOKS_GML = r"""{POLBOOKS_GML}"""\nPOLBLOGS_JSON = r"""{POLBLOGS_JSON}"""\n',
             GPU]
    script = "\n\n".join(parts)
    import os
    os.makedirs("kernel_collect", exist_ok=True)
    with open("kernel_collect/collect.py", "w") as f:
        f.write(script)
    print("wrote kernel_collect/collect.py", len(script), "chars")


if __name__ == '__main__':
    main()
