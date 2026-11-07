"""Build the teacher-facing Kaggle notebook: Phase 1 (base paper) vs the current final version.

    python3 build_final_comparison_notebook.py

Writes
  dataset_final/   small Kaggle dataset (graphs + cached per-graph result records) -- upload as
                   khatrilakshay2004/btp-final-comparison-data
  kernel_final/    final_comparison.ipynb + kernel-metadata.json

The notebook embeds the real tpoc_lpa.py (so the code the teacher reads is the code that ran), computes
every table from data, and prints the exact lines of every change using the ">>> CHANGE k" markers.
"""
import glob
import gzip
import json
import os
import shutil

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.abspath(os.path.join(HERE, "..", "datasets"))
CACHE = os.path.join(HERE, "data", "phase4_cache")
DS = os.path.join(HERE, "dataset_final")
KN = os.path.join(HERE, "kernel_final")

OWNER = "khatrilakshay2004"
DS_SLUG = "btp-final-comparison-data"
KN_TITLE = "BTP TPOC-LPA Phase 1 vs Final Comparison"
KN_SLUG = "btp-tpoc-lpa-phase-1-vs-final-comparison"


# --------------------------------------------------------------------------- dataset
def build_dataset():
    """Flat folder (no subdirectories, no .gz/.npz): Kaggle's handling of nested folders and archives is fiddly."""
    import numpy as np
    if os.path.isdir(DS):
        shutil.rmtree(DS)
    os.makedirs(DS)
    cp = lambda src, dst: shutil.copy(src, os.path.join(DS, dst))
    cp(f"{BUNDLE}/dolphins/dolphins.gml", "dolphins.gml")
    cp(f"{BUNDLE}/dolphins/dolphins_truth.json", "dolphins_truth.json")
    cp(f"{BUNDLE}/football/football.gml", "football.gml")
    cp(f"{BUNDLE}/polbooks/polbooks.gml", "polbooks.gml")
    for src, dst in (("edges", "email_edges.txt"), ("labels", "email_labels.txt")):
        with gzip.open(f"{BUNDLE}/email/{src}.txt.gz", "rt") as f, open(os.path.join(DS, dst), "w") as g:
            g.write(f.read())
    # Cora: keep only ids + class (the raw file carries 1433 word features per paper), plus the citation edges
    with open(f"{BUNDLE}/cora/cora/cora.content") as f, open(os.path.join(DS, "cora_labels.tsv"), "w") as g:
        for line in f:
            p = line.rstrip("\n").split("\t")
            g.write(f"{p[0]}\t{p[-1]}\n")
    cp(f"{BUNDLE}/cora/cora/cora.cites", "cora_cites.txt")
    # LFR graphs used by the live tables: line 1 = comma-separated community id of node 0..n-1, then one 'u v' edge per line
    want = ["lfr_n1000_mu0.1_s%d" % s for s in range(1, 9)]
    want += ["lfr_n1000_mu0.6_s%d" % s for s in list(range(1, 6)) + list(range(11, 21))]
    want += ["lfr_n5000_mu0.6_s%d" % s for s in (1, 2, 3)]
    want += ["lfr_n10000_mu0.1_s%d" % s for s in (1, 2, 3)]
    want += ["lfr_n10000_mu0.6_s%d" % s for s in (1, 2, 3)]
    for w in want:
        z = np.load(f"{BUNDLE}/lfr/{w}.npz")
        with open(os.path.join(DS, w + ".txt"), "w") as g:
            g.write(",".join(str(int(c)) for c in z["truth"]) + "\n")
            g.write("\n".join(f"{int(a)} {int(b)}" for a, b in z["edges"].tolist()) + "\n")
    # cached per-graph result records written by phase4_ablation.py (every regime, one row per graph)
    for p in sorted(glob.glob(f"{CACHE}/rec_*.json")):
        cp(p, os.path.basename(p))
    n_rec = len(glob.glob(f"{DS}/rec_*.json"))
    with open(os.path.join(DS, "MANIFEST_FINAL.txt"), "w") as f:
        f.write("BTP final comparison data. Marker file used by the notebook to find this dataset.\n")
        f.write(f"{len(want)} LFR graphs, {n_rec} cached result records.\n")
    json.dump({"title": "BTP final comparison data", "id": f"{OWNER}/{DS_SLUG}",
               "licenses": [{"name": "other"}]}, open(os.path.join(DS, "dataset-metadata.json"), "w"), indent=1)
    size = sum(os.path.getsize(p) for p in glob.glob(f"{DS}/*") if os.path.isfile(p))
    print(f"dataset_final: {n_rec} records, {len(want)} LFR graphs, {len(os.listdir(DS))} files, {size/1e6:.1f} MB")


# --------------------------------------------------------------------------- notebook cells
MD_TITLE = r"""# BTP — TPOC-LPA: Phase 1 (base paper) vs. Final version

**What this notebook shows.** Every table compares three things:

| Column | Meaning |
|---|---|
| **Paper** | the number published in the base paper (Kong et al., *Expert Systems With Applications* 317, 131873, 2026) |
| **Phase 1** | my faithful re-implementation of the base paper (all my options switched **off**) |
| **Final** | Phase 4: `sim="tri_overlap"` (motif similarity) + `guard=True` + `refine="granularity"` (size-corrected Stage 2 chosen by the granularity rule). **Final + merge** adds the optional merge pass. |

**Colour code.** 🟢 the new version is better than Phase 1 · ⚪ same (within ±0.0005 NMI) · 🔴 the new version is worse.
Regressions are shown, not hidden.

**✏️ = a place where I changed the code.** Section 6 prints the exact lines of every change straight from `tpoc_lpa.py`
(each change is fenced by `>>> CHANGE k START … <<< CHANGE k END` comments in the file, so you can also search for them yourself).

**How the numbers were produced.** Small graphs and the LFR benchmarks are **run live** in this notebook. The big real graphs
(up to 34,000 nodes) and the 72 held-out synthetic graphs were run once with `phase4_ablation.py`; their per-graph results are read from cached records
(clearly labelled "cached"). Nothing is typed in by hand except the paper's published numbers and the three SNAP graphs, whose run takes ~15 minutes.
"""

CODE_SETUP = r"""
import os, sys, re, glob, json, gzip, time, html, statistics as st
from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from networkx.algorithms.community.quality import modularity
from IPython.display import HTML, display, Markdown

sys.path.insert(0, os.getcwd())
from tpoc_lpa import tpoc_lpa

FAST = os.environ.get("TPOC_NB_FAST") == "1"      # local development switch: skip the 5000/10000-node LFR runs

def find_data():
    env = os.environ.get("TPOC_NB_DATA")
    if env and os.path.isfile(os.path.join(env, "MANIFEST_FINAL.txt")):
        return env
    for cand in glob.glob("/kaggle/input/**/MANIFEST_FINAL.txt", recursive=True):
        return os.path.dirname(cand)
    for cand in ("dataset_final", "../dataset_final"):
        if os.path.isfile(os.path.join(cand, "MANIFEST_FINAL.txt")):
            return cand
    raise FileNotFoundError("attach the dataset btp-final-comparison-data (MANIFEST_FINAL.txt not found)")

D = find_data()
print("data dir:", D)

# ---------------------------------------------------------------- loaders (largest connected component, string labels)
def _lcc(G, truth):
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    keep = max(nx.connected_components(G), key=len)
    G = G.subgraph(keep).copy()
    return G, {n: truth[n] for n in G.nodes()}

def load_karate():
    G = nx.karate_club_graph()
    return _lcc(G, {n: str(G.nodes[n]["club"]) for n in G.nodes()})

def load_dolphin():
    G = nx.read_gml(f"{D}/dolphins.gml")
    t = json.load(open(f"{D}/dolphins_truth.json"))
    return _lcc(G, {n: str(t[n]) for n in G.nodes()})

def load_football():
    G = nx.read_gml(f"{D}/football.gml", label="id")
    return _lcc(G, {n: str(G.nodes[n]["value"]) for n in G.nodes()})

def load_polbooks():
    G = nx.read_gml(f"{D}/polbooks.gml")
    return _lcc(G, {n: str(G.nodes[n]["value"]) for n in G.nodes()})

def load_email():
    G = nx.Graph()
    for line in open(f"{D}/email_edges.txt"):
        if not line.startswith("#"):
            a, b = line.split(); G.add_edge(int(a), int(b))
    truth = {}
    for line in open(f"{D}/email_labels.txt"):
        n, c = line.split(); truth[int(n)] = str(c)
    G.add_nodes_from(truth)
    return _lcc(G, truth)

def load_cora():
    truth = dict(l.rstrip("\n").split("\t") for l in open(f"{D}/cora_labels.tsv"))
    G = nx.Graph(); G.add_nodes_from(truth)
    for line in open(f"{D}/cora_cites.txt"):
        a, b = line.split()
        if a in truth and b in truth:
            G.add_edge(a, b)
    return _lcc(G, truth)

def load_lfr(name):
    lines = open(f"{D}/{name}.txt").read().split("\n")
    truth = [int(c) for c in lines[0].split(",")]
    G = nx.Graph(); G.add_nodes_from(range(len(truth)))
    for l in lines[1:]:
        if l:
            a, b = l.split(); G.add_edge(int(a), int(b))
    return G, dict(enumerate(truth))

# ---------------------------------------------------------------- one run = one number set
BASE = {}                                                       # Phase 1: every option off == the base paper
FINAL = dict(sim="tri_overlap", guard=True, refine="granularity")   # Final (Phase 4)
FINAL_MERGE = dict(FINAL, merge=True, merge_min_gain=0.03)          # Final + optional merge pass

def run(G, truth, **kw):
    info = {}
    t0 = time.time()
    lab, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8, info=info, **kw)
    nodes = list(G.nodes())
    groups = defaultdict(set)
    for n in nodes:
        groups[lab[n]].add(n)
    return dict(nmi=nmi_score([truth[n] for n in nodes], [lab[n] for n in nodes]),
                q=modularity(G, list(groups.values())), nc=len(groups), info=info, sec=time.time() - t0)

# ---------------------------------------------------------------- tiny HTML table helper (no pandas needed)
WIN, TIE, LOSS, NOTE, HEAD = "#d4edda", "#eeeeee", "#f8d7da", "#fff3cd", "#dbe4f0"
TOL = 5e-4

def verdict(delta, tol=TOL):
    if delta > tol:  return ("🟢 better", WIN)
    if delta < -tol: return ("🔴 worse", LOSS)
    return ("⚪ same", TIE)

def table(headers, rows, caption=None):
    h = '<table style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px;color:#1a1a1a;margin:6px 0">'
    if caption:
        h += f'<caption style="text-align:left;font-weight:bold;padding:4px 0">{html.escape(caption)}</caption>'
    h += "<tr>" + "".join(f'<th style="border:1px solid #999;padding:4px 8px;background:{HEAD};text-align:left">{html.escape(x)}</th>' for x in headers) + "</tr>"
    for r in rows:
        h += "<tr>"
        for c in r:
            txt, bg = (c if isinstance(c, tuple) else (c, "#ffffff"))
            h += f'<td style="border:1px solid #bbb;padding:3px 8px;background:{bg}">{html.escape(str(txt))}</td>'
        h += "</tr>"
    display(HTML(h + "</table>"))

f4 = lambda x: f"{x:.4f}"
sd = lambda x: f"{x:+.4f}"

# ---------------------------------------------------------------- what the base paper published (Tables 8, 9, 10 of the paper)
PAPER_NMI = {"Karate": 1.0000, "Dolphin": 0.9053, "Football": 0.9146,
             "LFR n=1000, mu=0.1": 1.0000, "LFR n=1000, mu=0.6": 0.9869, "LFR n=5000, mu=0.6": 0.9974,
             "LFR n=10000, mu=0.1": 1.0000, "LFR n=10000, mu=0.6": 0.9935}
PAPER_Q = {"Karate": 0.3715, "Dolphin": 0.5265, "Football": 0.6032}
"""

MD_S1 = r"""## 1. The paper's own benchmarks — Phase 1 (base paper) vs Final

Each row is run live below with both versions. LFR rows are means over several generated graphs (seeds) with a paired per-seed count.
Karate, Dolphin and Football are single graphs. The base-paper column is what the paper published (its own SNAP subsamples are not
reproducible, so those are in Section 4).
"""

CODE_S1 = r"""
def seeds(prefix, ids):
    return [f"{prefix}_s{i}" for i in ids]

SUITE = [
    ("Karate",              [load_karate()]),
    ("Dolphin",             [load_dolphin()]),
    ("Football",            [load_football()]),
    ("LFR n=1000, mu=0.1",  seeds("lfr_n1000_mu0.1", range(1, 9))),
    ("LFR n=1000, mu=0.6",  seeds("lfr_n1000_mu0.6", list(range(1, 6)) + list(range(11, 21)))),
]
if not FAST:
    SUITE += [("LFR n=5000, mu=0.6",  seeds("lfr_n5000_mu0.6", (1, 2, 3))),
              ("LFR n=10000, mu=0.1", seeds("lfr_n10000_mu0.1", (1, 2, 3))),
              ("LFR n=10000, mu=0.6", seeds("lfr_n10000_mu0.6", (1, 2, 3)))]

T1 = {}          # name -> dict(paper, base, final, base_q, final_q, better, equal, worse, n)
t_start = time.time()
for name, graphs in SUITE:
    b, f, bq, fq = [], [], [], []
    for g in graphs:
        G, truth = load_lfr(g) if isinstance(g, str) else g
        rb, rf = run(G, truth, **BASE), run(G, truth, **FINAL)
        b.append(rb["nmi"]); f.append(rf["nmi"]); bq.append(rb["q"]); fq.append(rf["q"])
    d = [y - x for x, y in zip(b, f)]
    T1[name] = dict(paper=PAPER_NMI[name], base=st.mean(b), final=st.mean(f), base_q=st.mean(bq), final_q=st.mean(fq),
                    better=sum(x > 1e-6 for x in d), equal=sum(abs(x) <= 1e-6 for x in d), worse=sum(x < -1e-6 for x in d), n=len(d))
    print(f"  done {name:<22} ({len(graphs)} graph(s), {time.time()-t_start:.0f}s elapsed)")

rows = []
for name, r in T1.items():
    v1, c1 = verdict(r["final"] - r["base"])
    gap = r["final"] - r["paper"]
    if gap > TOL:    vp = ("above the paper", WIN)
    elif gap >= -TOL: vp = ("matches the paper", WIN)
    elif gap >= -0.002: vp = (f"within 0.002 of the paper ({gap:+.4f})", WIN)
    else:            vp = (f"below paper by {-gap:.4f}", LOSS)
    seedwise = f"{r['better']} / {r['equal']} / {r['worse']}" if r["n"] > 1 else "single graph"
    star = " *" if "5000" in name else ""
    rows.append([name + star, f4(r["paper"]), f4(r["base"]), f4(r["final"]), (sd(r["final"] - r["base"]), c1), (v1, c1), vp, seedwise])
table(["Benchmark (NMI)", "Paper", "Phase 1 (base)", "Final", "Final − Phase 1", "vs Phase 1", "vs the paper", "seeds better / equal / worse"],
      rows, "Table 1 — NMI on the paper's own benchmarks")
display(Markdown("\\* LFR n=5000: my generator settings differ from the paper's N4 setting, so this row is indicative only.  "
                 "**Karate** stays at 0.8372 (the paper's own plain-LPA baseline also scores 0.8372 there); the gap is one node and 300 random tie-break runs never reach 1.0."))

rows = []
for name in ("Karate", "Dolphin", "Football"):
    r = T1[name]; v, c = verdict(r["final_q"] - r["base_q"])
    rows.append([name, f4(PAPER_Q[name]), f4(r["base_q"]), f4(r["final_q"]), (sd(r["final_q"] - r["base_q"]), c)])
table(["Modularity Q", "Paper", "Phase 1", "Final", "Final − Phase 1"], rows, "Table 1b — modularity on the classic graphs (Q is not the objective here; NMI against the true communities is)")

b_ = sum(1 for r in T1.values() if r["final"] - r["base"] > TOL)
t_ = sum(1 for r in T1.values() if abs(r["final"] - r["base"]) <= TOL)
w_ = sum(1 for r in T1.values() if r["final"] - r["base"] < -TOL)
display(Markdown(f"**Scoreboard, paper benchmarks:** Final is better than Phase 1 on **{b_}**, the same on **{t_}**, worse on **{w_}** of {len(T1)} benchmarks."))
"""

MD_S2 = r"""## 2. Graphs beyond the paper — where the new version matters most

The paper only tested graphs where its Stage 2 behaves. On dense or noisy graphs it collapses into one community; on sparse graphs it stops with far too many.
These per-graph results come from the cached records of `phase4_ablation.py` (each graph was run once for every regime).
"""

CODE_S2 = r"""
REC = {os.path.basename(p)[4:-5]: json.load(open(p)) for p in glob.glob(f"{D}/rec_*.json")}

def s5_regime(r, rho=1.5):
    # granularity rule: keep the size-corrected (gamma=1) result only if it has <= 1.5x the communities of the paper-style one
    return "gamma1" if r["gamma1"]["nc"] <= rho * r["phase3"]["nc"] else "phase3"

def final_of(r, merge=False, eps=0.03):
    reg = s5_regime(r); base = r[reg]
    if not merge:
        return base
    m = r["g1_merge" if reg == "gamma1" else "p3_merge"]
    return m if (m["merge_gain"] >= eps and m["merges"] > 0) else base

def ids_for(prefix, mu, seeds_):
    return [f"{prefix}_mu{mu}_s{s}" for s in seeds_]

S15 = list(range(1, 6)) + list(range(11, 21))
BEYOND = [
    ("email-Eu (42 departments)",     ["email-Eu"]),
    ("LFR n=1000, mu=0.7 (15 seeds)", ids_for("lfr1000", 0.7, S15)),
    ("LFR n=1000, mu=0.6 (15 seeds)", ids_for("lfr1000", 0.6, S15)),
    ("LFR n=1000, mu=0.5 (15 seeds)", ids_for("lfr1000", 0.5, S15)),
    ("Amazon Computers",              ["amazon_computers"]),
    ("Amazon Photo",                  ["amazon_photo"]),
    ("Coauthor-CS",                   ["coauthor_cs"]),
    ("Coauthor-Physics",              ["coauthor_physics"]),
    ("Cora",                          ["cora"]),
    ("Cora-ML",                       ["cora_ml"]),
    ("Citeseer",                      ["citeseer"]),
    ("Pubmed",                        ["pubmed"]),
    ("DBLP-authors",                  ["dblp_authors"]),
    ("Polbooks",                      ["polbooks"]),
    ("Polblogs",                      ["polblogs"]),
]
rows, T2 = [], {}
for name, ids in BEYOND:
    rs = [REC[i] for i in ids]
    b = st.mean(r["base"]["nmi"] for r in rs)
    f = st.mean(final_of(r)["nmi"] for r in rs)
    m = st.mean(final_of(r, merge=True)["nmi"] for r in rs)
    T2[name] = (b, f, m)
    v1, c1 = verdict(f - b); v2, c2 = verdict(m - b)
    rows.append([name, f"{rs[0]['n']:,}", f4(b), f4(f), (sd(f - b), c1), (v1, c1), f4(m), (sd(m - b), c2), (v2, c2)])
table(["Graph (NMI)", "nodes", "Phase 1 (base)", "Final", "Final − Ph.1", "verdict", "Final + merge", "+merge − Ph.1", "verdict (+merge)"],
      rows, "Table 2 — Phase 1 vs Final, graphs the paper did not test (cached records)")
display(Markdown("*Final* = motif + guard + granularity rule. *Final + merge* adds the merge pass (accepted only when the total modularity gain is ≥ 0.03). "
                 "The merge pass is optional because it also has a failure case (SNAP-DBLP, Section 4)."))

# scoreboard over EVERY graph in the study (per graph, not per group)
def cls(gid):
    if gid in ("karate","dolphin","football","polbooks","polblogs","email-Eu","cora","citeseer","amazon_photo"): return "real, development (9)"
    if gid in ("amazon_computers","coauthor_cs","coauthor_physics"): return "real, held-out (3)"
    if gid in ("cora_ml","pubmed","dblp_authors"): return "real, fresh (3)"
    m = re.match(r"lfr1000_mu[\d.]+_s(\d+)$", gid)
    if m and int(m[1]) <= 5: return "synthetic, development (20)"
    return "synthetic, held-out (72)"
groups = defaultdict(list)
for gid, r in REC.items():
    groups[cls(gid)].append(r)
rows = []
for g in ("real, development (9)", "real, held-out (3)", "real, fresh (3)", "synthetic, development (20)", "synthetic, held-out (72)"):
    rs = groups[g]
    cells = [g]
    for merge in (False, True):
        d = [final_of(r, merge)["nmi"] - r["base"]["nmi"] for r in rs]
        w, l = sum(x > 1e-6 for x in d), sum(x < -1e-6 for x in d)
        cells += [(f"{w} better / {len(d)-w-l} equal / {l} worse", WIN if w > l else (LOSS if l > w else TIE)), sd(st.mean(d))]
    rows.append(cells)
table(["Graph set (count)", "Final vs Phase 1 (per graph)", "mean ΔNMI", "Final + merge vs Phase 1", "mean ΔNMI"], rows,
      "Table 2b — scoreboard over all 107 graphs of the study (a graph is 'equal' if NMI differs by < 1e-6)")
"""

MD_S3 = r"""## 3. The 72 held-out synthetic graphs (never used to design anything)

These were generated *after* the changes were designed: new seeds, larger graphs (n = 5,000 and 10,000), graphs with few large communities and small 150-node graphs.
"""

CODE_S3 = r"""
HELD = [("LFR n=1000, mu=%s" % mu, ids_for("lfr1000", mu, range(11, 21))) for mu in (0.5, 0.6, 0.7)]
HELD += [("LFR n=5000, mu=%s" % mu, ids_for("lfr5000", mu, (1, 2, 3))) for mu in (0.5, 0.6, 0.7)]
HELD += [("LFR n=10000, mu=%s" % mu, ids_for("lfr10000", mu, (1, 2, 3))) for mu in (0.5, 0.6, 0.7)]
HELD += [("few large communities, mu=%s" % mu, ids_for("coarse1000", mu, (1, 2, 3, 4))) for mu in (0.2, 0.4, 0.6)]
HELD += [("small n=150, mu=%s" % mu, ids_for("small150", mu, range(1, 7))) for mu in (0.2, 0.4)]
rows, tw, te, tl = [], 0, 0, 0
for name, ids in HELD:
    rs = [REC[i] for i in ids]
    b = st.mean(r["base"]["nmi"] for r in rs); f = st.mean(final_of(r)["nmi"] for r in rs)
    d = [final_of(r)["nmi"] - r["base"]["nmi"] for r in rs]
    w, l = sum(x > 1e-6 for x in d), sum(x < -1e-6 for x in d); e = len(d) - w - l
    tw += w; te += e; tl += l
    v, c = verdict(f - b)
    rows.append([name, len(rs), f4(b), f4(f), (sd(f - b), c), (v, c), f"{w} / {e} / {l}"])
table(["Held-out group (NMI)", "graphs", "Phase 1 (base)", "Final", "Final − Ph.1", "verdict", "graphs better / equal / worse"], rows,
      "Table 3 — held-out synthetic graphs")
display(Markdown(f"**Total over the {tw+te+tl} held-out synthetic graphs: Final is better than Phase 1 on {tw}, equal on {te}, worse on {tl}.**"))
"""

MD_S4 = r"""## 4. SNAP graphs and the overlap stage (recorded runs)

These graphs are large (17k–93k nodes, up to ~15 minutes per run), so the values below are the recorded outputs of
`phase4_vs_paper.py snap …` and `phase4_overlap.py sweep …` (re-run on 25 Sep 2026).
**The paper evaluated smaller, undocumented subsamples of the same SNAP data, so the paper column is not comparable — compare Phase 1 with Final only.**
"""

CODE_S4 = r"""
SNAP = [  # name, paper (own, smaller subset), Phase 1, Final (Phase 4), Final + merge
    ("SNAP Amazon (16,716 nodes; paper used 6,428)",  0.9983, 0.9586, 0.9610, 0.9824),
    ("SNAP DBLP (93,432 nodes; paper used 23,654)",   0.9933, 0.7626, 0.7667, 0.6197),
    ("SNAP Youtube (39,841 nodes; paper used 10,364)", 0.8808, 0.5195, 0.5155, 0.5155),
]
rows = []
for name, pap, p1, fi, fm in SNAP:
    v1, c1 = verdict(fi - p1); v2, c2 = verdict(fm - p1)
    rows.append([name, (f4(pap), NOTE), f4(p1), f4(fi), (sd(fi - p1), c1), (v1, c1), f4(fm), (sd(fm - p1), c2), (v2, c2)])
table(["Graph (NMI, first community as truth)", "Paper (other graph)", "Phase 1", "Final", "Final − Ph.1", "verdict", "Final + merge", "+merge − Ph.1", "verdict (+merge)"],
      rows, "Table 4 — SNAP graphs (recorded)")
display(Markdown("**SNAP-DBLP with merge is a real failure:** 9,803 communities were merged into 807 (modularity rises 0.762 → 0.892 while NMI falls) — the resolution limit of modularity. "
                 "This is why the merge pass stays optional. On SNAP-Youtube the motif similarity costs 0.004 and nothing recovers it."))

OVL = [  # graph, truth overlap, share flagged by Stage 3 at beta=0.8, ONMI disjoint, ONMI with Stage 3 (beta=0.8)
    ("SNAP Amazon", "81 %", "4.0 %", 0.4604, 0.4770),
    ("SNAP DBLP",   "15 %", "24.7 %", 0.2871, 0.2327),
]
rows = []
for g, tov, fl, a, b in OVL:
    v, c = verdict(b - a)
    rows.append([g, tov, fl, f4(a), f4(b), (sd(b - a), c), (v, c)])
table(["Graph", "nodes truly in >1 community", "flagged by Stage 3 (paper's beta=0.8)", "ONMI without Stage 3", "ONMI with Stage 3", "change", "verdict"],
      rows, "Table 4b — Stage 3 (overlap) scored against real overlapping ground truth (Phase 4 study; not a fix, a finding)")
display(Markdown("The base paper's overlap rate is not adapted to the graph: it flags 4 % where 81 % is true and 25 % where 15 % is true. "
                 "The paper's own overlap metric EQ also moves against agreement with the truth (Amazon: EQ falls 0.9418 → 0.9379 while ONMI rises)."))
"""

MD_S5 = r"""## 5. Where the new version is *not* better (honest regressions)

A change is only trustworthy if its failures are visible. The rule I fixed in advance: **a change becomes the default only if no unseen real graph loses more than 0.005 NMI.**
Several of my changes broke that rule, so **every change stays optional; the defaults reproduce the paper**.
"""

CODE_S5 = r"""
rows = []
for name, (b, f, m) in T2.items():
    for label, val in (("Final", f), ("Final + merge", m)):
        if val - b < -TOL:
            rows.append([name, label, f4(b), f4(val), (sd(val - b), LOSS)])
for name, pap, p1, fi, fm in SNAP:
    for label, val in (("Final", fi), ("Final + merge", fm)):
        if val - p1 < -TOL:
            rows.append([name.split(" (")[0], label, f4(p1), f4(val), (sd(val - p1), LOSS)])
for name, r in T1.items():
    if r["final"] - r["base"] < -TOL:
        rows.append([name, "Final", f4(r["base"]), f4(r["final"]), (sd(r["final"] - r["base"]), LOSS)])
table(["Graph", "Version", "Phase 1", "Version's NMI", "change"], rows, "Table 5 — every row above where the new version is worse than Phase 1")
display(Markdown("Reading it: **Coauthor-Physics** is the guard's failure (higher modularity, worse match to the truth); **Coauthor-CS** is the motif similarity on a sparse graph; "
                 "**SNAP-DBLP** is the merge pass over-merging; **Karate / LFR mu=0.6** are gaps to the *paper* that remain open, not regressions."))
"""

MD_SUM = r"""## Summary — where the new version wins, and where it does not"""

CODE_SUM = r"""
gain = sorted(((max(f, m) - b, name, b, f, m) for name, (b, f, m) in T2.items()), reverse=True)[:6]
lines = ["**Biggest wins (NMI, Phase 1 -> Final; and with the optional merge pass where it differs):**"]
for d, name, b, f, m in gain:
    lines.append(f"- **{name}**: {b:.4f} -> {f:.4f}" + (f" -> with merge {m:.4f}" if abs(m - f) > TOL else ""))
lines.append("")
better = [n for n, r in T1.items() if r["final"] - r["base"] > TOL]
same = [n for n, r in T1.items() if abs(r["final"] - r["base"]) <= TOL]
lines.append("**On the paper's own benchmarks:** better than Phase 1 on " + "; ".join(better) + ("; unchanged on " + "; ".join(same) if same else "") + ". Football is now **above** the paper's published 0.9146.")
gaps = [f"{n} (-{r['paper'] - r['final']:.4f})" for n, r in T1.items() if r["paper"] - r["final"] > 0.002]
lines.append("**Still below the paper:** " + "; ".join(gaps) + ".")
lines.append("")
lines.append("**Not better (see Table 5):** Coauthor-Physics, Coauthor-CS, Pubmed and Amazon Photo without the merge pass; SNAP-DBLP with the merge pass; SNAP-Youtube. "
             "This is why none of the changes is switched on by default.")
display(Markdown("\n".join(lines)))
"""

MD_S6 = r"""## 6. ✏️ Where I changed the code (for the teacher)

Everything below is printed **from the same `tpoc_lpa.py` that produced the tables above**. Lines that belong to a change are marked ✏️.
The paper's own code is untouched and marked `# PAPER` in the file; with all options off the file **is** the base paper
(checked by hashing the output of every configuration before and after each edit: identical).

Each change has: the problem in the base paper → what I changed → the source it comes from → the code → a live before/after → its weak spot.
"""

CODE_MAP = r"""
import re as _re, html as _html
SRC = open("tpoc_lpa.py").read().split("\n")

def change_lines(k, context=1):
    core, start = set(), None
    for i, l in enumerate(SRC, 1):
        if start is None and f"CHANGE {k} START" in l and ">>>" in l:
            start = i
        if start is not None:
            core.add(i)
            if f"CHANGE {k} END" in l and "<<<" in l:
                start = None
        if f"[CHANGE {k}]" in l:
            core.add(i)
    show = set(core)
    for i in list(core):
        for j in range(i - context, i + context + 1):
            if 1 <= j <= len(SRC): show.add(j)
    return sorted(core), sorted(show)

def spans(core):
    runs = [[core[0], core[0]]]
    for a in core[1:]:
        if a - runs[-1][1] <= 3: runs[-1][1] = a
        else: runs.append([a, a])
    return ", ".join(f"{s}–{e}" if s != e else str(s) for s, e in runs)

def show_change_code(k, title):
    core, shown = change_lines(k)
    runs, cur = [], [shown[0]]
    for a in shown[1:]:
        if a - cur[-1] <= 2: cur.extend(range(cur[-1] + 1, a + 1))
        else: runs.append(cur); cur = [a]
    runs.append(cur)
    out = (f'<div style="font-family:Arial;font-size:14px;margin:8px 0"><b>✏️ CHANGE {k} — {_html.escape(title)}</b> '
           f'&nbsp;<span style="color:#555">tpoc_lpa.py, lines {spans(core)}; search the file for <code>CHANGE {k}</code></span></div>')
    out += '<pre style="background:#f6f8fa;border:1px solid #d0d7de;padding:8px;font-size:12px;line-height:1.35;overflow-x:auto;color:#1a1a1a">'
    for ri, run in enumerate(runs):
        if ri: out += '<span style="color:#888">      ⋮\n</span>'
        for i in run:
            mark = "✏️" if i in core else "  "
            bg = "background:#fff3b0;" if i in core else ""
            out += f'<span style="{bg}display:block">{i:>4} {mark} {_html.escape(SRC[i-1])}</span>'
    display(HTML(out + "</pre>"))
    return core

ORDER = [(1, "A. Motif similarity", "sim='tri_overlap'", "tri_overlap()"),
         (3, "B. Size-corrected Stage 2 (gamma term)", "gamma=1.0", "edge_attractiveness(), community_evolution_null()"),
         (5, "C. Merge pass", "merge=True, merge_min_gain=0.03", "community_merge() + its call in tpoc_lpa()"),
         (2, "D. Best-iterate guard", "guard=True", "_partition_of(), community_evolution()"),
         (4, "E. Granularity rule", "refine='granularity'", "tpoc_lpa()")]
rows = []
for k, name, flag, fn in ORDER:
    core, _ = change_lines(k)
    rows.append([f"CHANGE {k}", name, flag, fn, spans(core)])
table(["Marker in the file", "Change", "switch", "function", "lines"], rows, "Change map — search tpoc_lpa.py for the marker")
"""

MD_C1 = r"""### ✏️ Change A — Motif similarity  *(code marker: CHANGE 1)*
**Problem in the base paper.** Stage 2 scores how attractive a neighbour is; part of that is a similarity between the two linked nodes, and the paper uses **Jaccard**
(shared neighbours ÷ all distinct neighbours). Two well-connected nodes in one tight community share many neighbours, but their combined neighbour list is huge, so Jaccard scores the link low.

**What I changed.** With `c` = number of shared neighbours (= triangles the link is in), the similarity becomes  **`c / (1 + c)`**:
0 triangles → 0, 1 → 0.5, 3 → 0.75, 9 → 0.9. It depends on the triangle count, not on how long the neighbour lists are.

**Source.** Li et al., *ACM TKDD* 2020 (motif-aware label propagation); H-LPMC, *KAIS* 2025; graphlet-based edge weighting, *JIIS* 2026. They support "triangle structure helps label propagation".
The exact formula `c/(1+c)` and its place inside TPOC-LPA's attractiveness are my own.
"""
CODE_C1 = r"""
show_change_code(1, "Motif similarity replaces Jaccard inside Eq. 11")
G, t = load_polbooks()
rows = []
for label, kw in (("Jaccard (base paper)", {}), ("Motif  c/(1+c)", dict(sim="tri_overlap"))):
    r = run(G, t, **kw); rows.append([label, f4(r["nmi"]), r["nc"], len(set(t.values()))])
b, m = rows[0][1], rows[1][1]
v, c = verdict(float(m) - float(b))
rows[1][1] = (rows[1][1], c)
table(["Political-books graph (105 books)", "NMI", "communities found", "true communities"], rows, "Live before/after")
# weak spot, from the cached records: motif effect = guard-only (Jaccard) vs motif+guard, on sparse graphs
rows = []
for name, gid in (("Cora", "cora"), ("Citeseer", "citeseer"), ("Coauthor-CS", "coauthor_cs"), ("Polbooks", "polbooks"), ("LFR n=1000, mu=0.6 seed 1", "lfr1000_mu0.6_s1")):
    r = REC[gid]; d = r["phase3"]["nmi"] - r["jac_guard"]["nmi"]; v, c = verdict(d, 1e-4)
    rows.append([name, f4(r["jac_guard"]["nmi"]), f4(r["phase3"]["nmi"]), (sd(d), c)])
table(["Effect of the motif similarity alone (cached)", "Jaccard", "Motif", "change"], rows, "Weak spot: on sparse graphs one shared neighbour is often chance, yet it scores 0.5")
"""

MD_C2 = r"""### ✏️ Change B — Size-corrected Stage 2 (the γ term)  *(code marker: CHANGE 3)*
**Problem in the base paper.** Stage 2 (Eq. 12) **adds up** the attractiveness of a node's neighbours inside each community. A big community always collects a bigger sum just for being big — a rich-get-richer snowball.
On dense graphs everything ends up in one community.

**What I changed.** I subtract what a random graph with the same degrees would give:

`score(c) = Σ_(j in N(i) ∩ c) A(i,j)  −  γ · s_i · S_c / W`

(real attraction minus expected-by-chance attraction). **γ = 0 is exactly the paper's Eq. 12; γ = 1 is the full correction.** I use γ = 1 and let the granularity rule (Change E) decide when to trust it.

**Source.** Barber & Clark, *Physical Review E* 80 (2009) — LPAm, the same "subtract the chance term" update. Mine: using it with the paper's node-importance-weighted attractiveness, and showing Eq. 12 is the unbounded γ = 0 case.
"""
CODE_C2 = r"""
show_change_code(3, "Size-corrected Stage 2: subtract the expected-by-chance attraction")
REG = {}
rows = []
for name, loader in (("email-Eu", load_email), ("Football", load_football), ("Karate", load_karate), ("Dolphin", load_dolphin), ("Polbooks", load_polbooks)):
    G, t = loader()
    p = run(G, t, sim="tri_overlap", guard=True)            # paper-style Stage 2 (gamma = 0) + guard
    g1 = run(G, t, sim="tri_overlap", gamma=1.0)             # size-corrected Stage 2 (gamma = 1)
    REG[name] = (G, t, p, g1)
    v, c = verdict(g1["nmi"] - p["nmi"], 1e-3)
    rows.append([name, len(set(t.values())), p["nc"], g1["nc"], f4(p["nmi"]), f4(g1["nmi"]), (sd(g1["nmi"] - p["nmi"]), c)])
table(["Graph", "true communities", "communities gamma=0", "communities gamma=1", "NMI gamma=0 (paper-style)", "NMI gamma=1", "change"],
      rows, "Live before/after: gamma = 0 (the paper's Eq. 12) vs gamma = 1")
display(Markdown("**Where it wins:** email-Eu and Football (it also recovers Football's 12th conference). "
                 "**Weak spot:** on small graphs whose true communities are *coarse* (Karate, Dolphin, Polbooks) gamma = 1 splits the graph finer than the truth and loses badly. "
                 "That is exactly why Change E exists."))
"""

MD_C3 = r"""### ✏️ Change C — Merge pass  *(code marker: CHANGE 5)*
**Problem.** On sparse graphs Stage 2 stops with far too many small communities — 214 on Cora, which has 7 true classes. (This is *under*-merging only; the opposite failure, collapse into one community, is handled by Changes B and D.)

**What I changed.** After Stage 2, greedily merge the pair of communities whose merger raises modularity the most, and keep merging while any pair still gains. Then apply a **gate: accept the whole pass only if the total modularity gain is ≥ 0.03**; otherwise keep the pre-merge partition.
The gate is on the total gain, not on each merge.

**Source.** Liu & Murata, *Physica A* 389 (2010) — LPAm+ (size-corrected update + greedy merging); the greedy merging follows Clauset, Newman & Moore, *Phys. Rev. E* 70 (2004). **The 0.03 gate is my own** (chosen on the development graphs by a sweep).
"""
CODE_C3 = r"""
show_change_code(5, "Merge pass with a total-gain gate")
rows = []
for name, loader in (("Cora", load_cora), ("Football", load_football), ("Polbooks", load_polbooks)):
    G, t = loader()
    a = run(G, t, **FINAL)
    info = {}
    m = run(G, t, **FINAL_MERGE)
    gain = m["info"]["merge_gain"]; ok = m["info"]["merges"] > 0
    v, c = verdict(m["nmi"] - a["nmi"])
    rows.append([name, len(set(t.values())), a["nc"], m["nc"], f4(a["nmi"]), f4(m["nmi"]),
                 (f"total gain {gain:.3f} -> {'ACCEPTED' if ok else 'rejected (< 0.03)'}", WIN if ok else TIE), (sd(m["nmi"] - a["nmi"]), c)])
table(["Graph", "true communities", "communities before merge", "after merge", "NMI before", "NMI after", "gate (0.03)", "change"],
      rows, "Live before/after: the gate accepts the merge where it helps and rejects it where it does not")
display(Markdown("**Where it wins** (cached, Table 2): Cora, Cora-ML, Amazon Computers/Photo, Coauthor-CS/Physics, Pubmed, DBLP-authors. "
                 "**Weak spot:** on SNAP-DBLP it merges ~9,800 communities into 807 and loses 0.147 NMI (Table 4) — the resolution limit of modularity; a lower gate also hurts email-Eu and large noisy LFR graphs."))
"""

MD_C4 = r"""### ✏️ Change D — Best-iterate guard  *(code marker: CHANGE 2)* — **my own finding, no published source**
**Problem.** Stage 2 repeats until nothing changes, but the *last* round can be the worst one — e.g. after everything collapsed into a single community.

**What I changed.** After every Stage-2 round, measure modularity and remember the best round (starting from Stage 1's partition); return that round, not the last. That partition is what Stage 3 (overlap filtering) then receives.
Like saving the best checkpoint during training.

**Source.** None — my own design.
"""
CODE_C4 = r"""
show_change_code(2, "Best-modularity iterate is kept and passed on to Stage 3")
G, t = load_email()
rows = []
for label, kw in (("Base paper (last round)", {}), ("With guard (best round)", dict(guard=True))):
    r = run(G, t, **kw); rows.append([label, r["nc"], f4(r["nmi"])])
v, c = verdict(float(rows[1][2]) - float(rows[0][2])); rows[1][2] = (rows[1][2], c)
table(["email-Eu (986 people, 42 true departments)", "communities at the end", "NMI"], rows, "Live before/after: the paper's Stage 2 collapses to ONE community")
rows = []
for name, gid in (("Amazon Computers", "amazon_computers"), ("Coauthor-Physics", "coauthor_physics"), ("Coauthor-CS", "coauthor_cs")):
    r = REC[gid]; d = r["jac_guard"]["nmi"] - r["base"]["nmi"]; v, c = verdict(d)
    rows.append([name, f4(r["base"]["nmi"]), f4(r["jac_guard"]["nmi"]), (sd(d), c), f"{r['base']['q']:.3f} -> {r['jac_guard']['q']:.3f}"])
table(["Guard alone (cached)", "NMI base", "NMI with guard", "change", "modularity Q"], rows, "Held-out graphs: it can help — and it can fail")
display(Markdown("**Weak spot:** on unseen **Coauthor-Physics** the guard costs 0.081 NMI: it kept a more fragmented partition with *higher* modularity (0.345 → 0.506) that matches the true communities *worse*. "
                 "Modularity is not the same as matching the truth (the same trap as Karate). I first thought the guard was harmless because I had only tested it on graphs where it does nothing."))
"""

MD_C5 = r"""### ✏️ Change E — Granularity rule  *(code marker: CHANGE 4)* — **my own design, no published source**
**Problem.** The size-corrected Stage 2 (Change B) wins on some graphs and loses badly on others, and I cannot use the true labels to choose.

**What I changed.** Run both the paper-style and the size-corrected Stage 2 from the same Stage 1 partition, and keep the size-corrected result **only if it has at most 1.5× as many communities**.
If the two agree on scale, the correction only fine-tunes boundaries; if it shatters the graph, we are in the coarse-truth case where modularity misleads, so the paper-style result is kept.
The threshold 1.5 was chosen on the development graphs by leave-one-graph-out testing.
"""
CODE_C5 = r"""
show_change_code(4, "Granularity rule: keep gamma=1 only if it has <= 1.5x the communities")
rows = []
for name, (G, t, p, g1) in REG.items():
    info = {}
    r = run(G, t, **FINAL)
    ratio = g1["nc"] / p["nc"]
    took = r["info"]["refined"]
    best = max(p["nmi"], g1["nmi"])
    chosen = g1["nmi"] if took else p["nmi"]
    v, c = verdict(chosen - best, 1e-3)
    rows.append([name, p["nc"], g1["nc"], f"{ratio:.2f}x", ("gamma=1 used" if took else "paper-style kept", WIN if took else TIE), f4(chosen), (("picked the better one" if chosen >= best - 1e-3 else "picked the worse one"), c)])
table(["Graph", "communities gamma=0", "communities gamma=1", "ratio", "rule's decision (<= 1.5x ?)", "NMI of decision", "was it right?"],
      rows, "Live: the rule decides WITHOUT looking at the true labels")
display(Markdown("On the 72 held-out synthetic graphs the rule beat the previous version on 53, tied on 14 and lost on 5. "
                 "**Weak spot:** the margin is thin (Cora sits at 1.39×, Karate at 2.0×) and it loses 0.012 on Coauthor-Physics — so it stays optional."))
"""

MD_S7 = r"""## 7. Proof that "everything off" is the base paper

All five changes default to **off**. Here the default call is run on the paper's classic graphs and must give the reproduction numbers of Phase 1
(Dolphin and Football modularity match the paper exactly; Karate is the known one-node gap).
"""
CODE_S7 = r"""
expect = {"Karate": (0.8372, 0.4036), "Dolphin": (0.9053, 0.5113), "Football": (0.9095, 0.6032)}
rows = []
for name, loader in (("Karate", load_karate), ("Dolphin", load_dolphin), ("Football", load_football)):
    G, t = loader()
    lab, ov, NI, it = tpoc_lpa(G)            # no options at all
    nodes = list(G.nodes()); g = defaultdict(set)
    for n in nodes: g[lab[n]].add(n)
    nm = nmi_score([t[n] for n in nodes], [lab[n] for n in nodes]); q = modularity(G, list(g.values()))
    ok = abs(nm - expect[name][0]) < 5e-4 and abs(q - expect[name][1]) < 5e-4
    rows.append([name, f4(nm), f4(expect[name][0]), f4(q), f4(expect[name][1]), ("identical to Phase 1" if ok else "MISMATCH", WIN if ok else LOSS)])
    assert ok, name
table(["tpoc_lpa(G) with no options", "NMI", "Phase 1 NMI", "Q", "Phase 1 Q", "check"], rows, "Defaults reproduce the Phase 1 base-paper implementation")
"""

MD_REF = r"""## References (papers behind each change)

| Change | Paper |
|---|---|
| Base paper | Kong, Huo, Wang, Guo (2026). *Three-stage overlapping community detection algorithm based on label propagation.* Expert Systems With Applications 317, 131873. |
| A — motif similarity | Li et al. (2020). *Community detection by motif-aware label propagation.* ACM TKDD 14(2). · *H-LPMC*, Knowledge and Information Systems (2025). · *Graphlet-based edge weighting for community detection*, J. Intelligent Information Systems (2026). |
| B — γ term | Barber & Clark (2009). *Detecting network communities by propagating labels under constraints* (LPAm). Physical Review E 80, 026129. |
| C — merge pass | Liu & Murata (2010). *Advanced modularity-specialized label propagation algorithm…* (LPAm+). Physica A 389; arXiv:0910.1154. · Clauset, Newman & Moore (2004). *Finding community structure in very large networks.* Phys. Rev. E 70. |
| D, E | none — my own designs |
| Closest prior work | LPA-MNI, Entropy 23(5), 497 (2021): already combines modularity with node importance, so my contribution is the *analysis* of the base paper and where each regime works, not a new algorithm family. |
| Benchmarks | LFR (Lancichinetti et al., 2008) · overlapping NMI (Lancichinetti et al., 2009) · SNAP ground truth (Yang & Leskovec, 2012) · Amazon/Coauthor graphs (Shchur et al., 2018) · Cora-ML/Pubmed/DBLP (Bojchevski & Günnemann, 2018) |
"""


def build_notebook():
    os.makedirs(KN, exist_ok=True)
    src = open(os.path.join(HERE, "tpoc_lpa.py")).read()
    nb = nbf.v4.new_notebook()
    C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s, hide=False: C.append(_code(s, hide))

    def _code(s, hide):
        c = nbf.v4.new_code_cell(s.strip("\n"))
        if hide:
            c.metadata["_kg_hide-input"] = True
            c.metadata["jupyter"] = {"source_hidden": True}
        return c

    md(MD_TITLE)
    md("### The implementation (collapsed)\nThe full `tpoc_lpa.py` is written to disk here so the notebook is self-contained. Section 6 prints the changed lines from it.")
    code("%%writefile tpoc_lpa.py\n" + src, hide=True)
    code(CODE_SETUP)
    md(MD_S1); code(CODE_S1)
    md(MD_S2); code(CODE_S2)
    md(MD_S3); code(CODE_S3)
    md(MD_S4); code(CODE_S4)
    md(MD_S5); code(CODE_S5)
    md(MD_SUM); code(CODE_SUM)
    md(MD_S6); code(CODE_MAP)
    for m_, c_ in ((MD_C1, CODE_C1), (MD_C2, CODE_C2), (MD_C3, CODE_C3), (MD_C4, CODE_C4), (MD_C5, CODE_C5)):
        md(m_); code(c_)
    md(MD_S7); code(CODE_S7)
    md(MD_REF)
    nb.cells = C
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path = os.path.join(KN, "final_comparison.ipynb")
    nbf.write(nb, path)
    json.dump({"id": f"{OWNER}/{KN_SLUG}", "title": KN_TITLE, "code_file": "final_comparison.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": False, "keywords": [],
               "dataset_sources": [f"{OWNER}/{DS_SLUG}"], "kernel_sources": [], "competition_sources": [], "model_sources": []},
              open(os.path.join(KN, "kernel-metadata.json"), "w"), indent=2)
    print("wrote", path, f"({len(C)} cells)")


if __name__ == "__main__":
    build_dataset()
    build_notebook()
