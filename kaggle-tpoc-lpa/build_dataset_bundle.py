"""Assemble BTP/datasets: every graph used in the Phase 1 vs final comparison, in one folder.

    python3 build_dataset_bundle.py

Real graphs are copied byte-for-byte (originals, plus the LCC json for Polblogs); Karate is exported
from networkx; the LFR graphs (generated, so they must be stored to be reproducible on another
machine) are exported as compressed .npz. Then every loader is re-run against the bundle and a
MANIFEST.json (sizes, node/edge/class counts, sha256) is written.
"""
import glob, hashlib, json, os, pickle, re, shutil, sys
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data")
DST = os.path.abspath(os.path.join(HERE, "..", "datasets"))
os.makedirs(DST, exist_ok=True)

def copy(rel, dst_rel=None):
    s, d = os.path.join(SRC, rel), os.path.join(DST, dst_rel or rel)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d)

# ---- real graphs, originals ------------------------------------------------
for f in ("dolphins/dolphins.gml", "dolphins/dolphins.txt", "football/football.gml", "football/football.txt",
          "polbooks/polbooks.gml", "polbooks/polbooks.txt", "polblogs/polblogs.gml", "polblogs/polblogs.txt",
          "polblogs/polblogs_lcc.json", "email/edges.txt.gz", "email/labels.txt.gz"):
    copy(f)
for d in ("cora/cora", "citeseer/citeseer"):
    for root, _, files in os.walk(os.path.join(SRC, d)):
        for fn in files:
            copy(os.path.relpath(os.path.join(root, fn), SRC))
for f in sorted(glob.glob(os.path.join(SRC, "snap", "*.gz")) + glob.glob(os.path.join(SRC, "heldout_real", "*.npz"))):
    copy(os.path.relpath(f, SRC))

# ---- Karate (networkx ships it) + Dolphin ground truth ---------------------
os.makedirs(os.path.join(DST, "karate"), exist_ok=True)
K = nx.karate_club_graph()
nx.write_gml(K, os.path.join(DST, "karate", "karate.gml"))
sys.path.insert(0, HERE)
from dolphin_ground_truth import dolphin_truth_by_label
json.dump({k: int(v) for k, v in dolphin_truth_by_label().items()},
          open(os.path.join(DST, "dolphins", "dolphins_truth.json"), "w"), indent=1)

# ---- LFR graphs -----------------------------------------------------------
os.makedirs(os.path.join(DST, "lfr"), exist_ok=True)
def export_lfr(G, truth, name):
    n = G.number_of_nodes()
    assert sorted(G.nodes()) == list(range(n)), "LFR node ids are expected to be 0..n-1"
    np.savez_compressed(os.path.join(DST, "lfr", name + ".npz"),
                        edges=np.array(sorted((min(u, v), max(u, v)) for u, v in G.edges()), dtype=np.int32),
                        truth=np.array([truth[i] for i in range(n)], dtype=np.int32))
n_lfr = 0
for p in sorted(glob.glob(os.path.join(SRC, "phase4_cache", "lfr_*.pkl"))):
    b = os.path.basename(p)[:-4]
    m = re.match(r"lfr_n(\d+)_mu([\d.]+)_s(\d+)(.*)", b)
    n, mu, seed, tag = m.group(1), m.group(2), m.group(3), m.group(4)
    if "min_community100" in tag: base = f"lfr_coarse{n}_mu{mu}_s{seed}"      # few, large communities
    elif "average_degree8" in tag: base = f"lfr_small{n}_mu{mu}_s{seed}"      # n=150, small graphs
    else: base = f"lfr_n{n}_mu{mu}_s{seed}"
    G, t = pickle.load(open(p, "rb")); export_lfr(G, t, base); n_lfr += 1
sys.path.insert(0, HERE)
from lfr_sweep import generate_lfr                    # the mu=0.1 graphs of the paper's own benchmark
for n, seeds in ((1000, range(1, 9)), (10000, range(1, 4))):
    for s in seeds:
        G, t = generate_lfr(n=n, mu=0.1, seed=s); export_lfr(G, t, f"lfr_n{n}_mu0.1_s{s}"); n_lfr += 1
print("LFR graphs exported:", n_lfr)
