"""Verify BTP/datasets by loading every graph THROUGH THE BUNDLE and writing MANIFEST.json."""
import hashlib, json, os, sys, glob
BUNDLE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
os.environ["TPOC_DATA"] = BUNDLE
import networkx as nx
import datasets, phase4_ablation as P
from sklearn.metrics import normalized_mutual_info_score as nmi
from tpoc_lpa import tpoc_lpa

def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()[:16]

def _snap(name):
    from load_snap import load_snap_network
    G, t, _ = load_snap_network(name)
    return G, {n: str(v) for n, v in t.items()}

loaders = {"karate": datasets.karate, "dolphin": datasets.dolphin, "football": datasets.football,
           "polbooks": datasets.polbooks, "polblogs": datasets.polblogs, "email-Eu": datasets.email_eu,
           "cora": datasets.cora, "citeseer": datasets.citeseer,
           "amazon_photo": lambda: P.load_npz_graph("amazon_photo"), "amazon_computers": lambda: P.load_npz_graph("amazon_computers"),
           "coauthor_cs": lambda: P.load_npz_graph("coauthor_cs"), "coauthor_physics": lambda: P.load_npz_graph("coauthor_physics"),
           "cora_ml": lambda: P.load_npz_graph("cora_ml"), "pubmed": lambda: P.load_npz_graph("pubmed"),
           "dblp_authors": lambda: P.load_npz_graph("dblp_authors"),
           # SNAP: the SAME loader the Phase 1 tables use (induced subgraph on the top-5000 communities; NOT reduced to the largest component)
           "snap_amazon": lambda: _snap("amazon"), "snap_dblp": lambda: _snap("dblp"), "snap_youtube": lambda: _snap("youtube")}
manifest = {"root": "BTP/datasets", "graphs": {}, "lfr": {}}
print(f"{'graph':<18}{'nodes':>8}{'edges':>9}{'classes':>8}   (largest connected component, as used in every table)")
for name, ld in loaders.items():
    G, t = ld()
    manifest["graphs"][name] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "classes": len(set(t.values()))}
    print(f"{name:<18}{G.number_of_nodes():>8}{G.number_of_edges():>9}{len(set(t.values())):>8}", flush=True)
for p in sorted(glob.glob(os.path.join(BUNDLE, "lfr", "*.npz"))):
    nm = os.path.basename(p)[:-4]; G, t = datasets.lfr_npz(nm)
    manifest["lfr"][nm] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "communities": len(set(t.values()))}
print("LFR graphs:", len(manifest["lfr"]))
# the exported LFR graphs must reproduce the scores recorded when they were generated
checks = [("lfr_n1000_mu0.6_s1", 0.9291, 0.9421), ("lfr_n1000_mu0.6_s2", 0.9022, 0.9365), ("lfr_n1000_mu0.1_s5", 0.9711, 0.9827)]
for nm, want_base, want_p3 in checks:
    G, t = datasets.lfr_npz(nm); nodes = list(G.nodes()); y = [t[n] for n in nodes]
    b = nmi(y, [tpoc_lpa(G)[0][n] for n in nodes]); p3 = nmi(y, [tpoc_lpa(G, sim="tri_overlap", guard=True)[0][n] for n in nodes])
    ok = abs(b - want_base) < 5e-4 and abs(p3 - want_p3) < 5e-4
    print(f"  reproduce {nm}: base {b:.4f} (recorded {want_base}) | motif+guard {p3:.4f} (recorded {want_p3})  {'OK' if ok else 'MISMATCH'}")
    assert ok
manifest["files"] = {}
for root, _, files in os.walk(BUNDLE):
    for fn in sorted(files):
        p = os.path.join(root, fn); rel = os.path.relpath(p, BUNDLE)
        if rel in ("MANIFEST.json",) or fn == ".DS_Store": continue
        manifest["files"][rel] = {"bytes": os.path.getsize(p), "sha256_16": sha(p)}
json.dump(manifest, open(os.path.join(BUNDLE, "MANIFEST.json"), "w"), indent=1)
print("wrote MANIFEST.json:", len(manifest["files"]), "files")
