"""Generate the Phase 2 Kaggle notebook by inlining the local modules, so the
notebook stays in sync with the validated source instead of being retyped."""
import json
import re

HEADER = '''# %% [markdown]
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
'''


def strip_module(path, drop_imports=True):
    src = open(path).read()
    # drop the module docstring and imports; the notebook has its own header
    src = re.sub(r'^""".*?"""\n', '', src, count=1, flags=re.S)
    out = []
    for line in src.splitlines():
        if drop_imports and re.match(r'^\s*(import |from )', line):
            continue
        if line.startswith("if __name__"):
            break
        out.append(line)
    return "\n".join(out).strip("\n")


def main():
    parts = [HEADER]
    parts.append("# %% [markdown]\n# ## Phase 1 core (inlined, unmodified)\n")
    parts.append("# %%\n" + strip_module("tpoc_lpa.py"))
    parts.append("# %% [markdown]\n# ## Stage 4: budgeted semantic correction\n")
    parts.append("# %%\n" + strip_module("stage4.py"))
    parts.append(EXPERIMENTS)
    script = "\n\n".join(parts)
    with open("kernel_phase2/phase2.py", "w") as f:
        f.write(script)
    print("wrote kernel_phase2/phase2.py", len(script), "chars")


if __name__ == '__main__':
    main()
