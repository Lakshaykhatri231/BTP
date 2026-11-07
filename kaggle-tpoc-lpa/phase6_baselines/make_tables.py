"""Aggregate results/*.json into RESULTS.md (Markdown tables) + results/summary.json.  usage: python3 make_tables.py"""
import json, os, glob, statistics as st
import paper_numbers as P
H = os.path.dirname(os.path.abspath(__file__)); R = f"{H}/results"
ALGS = ["Fastgreedy", "LPA", "Louvain", "Walktrap", "Infomap*", "TPOC-LPA (repro)", "Ours"]
def J(p):
    return json.load(open(p)) if os.path.exists(p) else None
def fmt(x, d=4): return "—" if x is None else f"{x:.{d}f}"
def ranks(vals):   # vals: dict alg->value or None; higher better; competition ranking
    items = {a: v for a, v in vals.items() if v is not None}
    return {a: 1 + sum(1 for b in items.values() if b > v + 1e-12) for a, v in items.items()}
def table(rows, cols, title, note="", bold_best=True):
    """rows: list of (name, {col: val}); adds Avg and mean-rank rows over rows where every col has a value."""
    out = [f"### {title}", ""] + ([note, ""] if note else []) + ["| Network | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    full = [(n, v) for n, v in rows if all(v.get(c) is not None for c in cols)]
    for n, v in rows:
        best = max([x for x in v.values() if x is not None], default=None)
        cells = [("**" + fmt(v.get(c)) + "**") if (bold_best and v.get(c) is not None and best is not None and v[c] >= best - 1e-12) else fmt(v.get(c)) for c in cols]
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    if full:
        avg = {c: st.mean(v[c] for _, v in full) for c in cols}
        rk = {c: st.mean(ranks({a: v[a] for a in cols})[c] for _, v in full) for c in cols}
        out.append("| **Avg** | " + " | ".join(fmt(avg[c]) for c in cols) + " |")
        out.append("| **Mean rank** | " + " | ".join(f"{rk[c]:.2f}" for c in cols) + " |")
    return "\n".join(out) + "\n"

md = ["# Phase 6 — comparison with the TPOC-LPA paper's baselines", "",
      "Every number marked *run* was computed by us on the same graph with the same metric code. Numbers marked *paper* are copied from "
      "Kong et al. (2026) for baselines without public code (NSA, SimCMR, WLC, UMSTMO, GREESE, MDPA) and are NOT re-run. "
      "`Infomap*` is not in the paper (extra baseline). `TPOC-LPA (repro)` = our reimplementation of the paper's algorithm; `Ours` = the best pipeline "
      "(`sim=tri_overlap, guard, refine=granularity, merge=mapeq, moves=synwalk`). NMI = sklearn arithmetic NMI; Q = unweighted modularity.", ""]
summ = {}
# ---------------- real graphs
real_names = [("karate", "Karate"), ("dolphin", "Dolphin"), ("football", "Football"), ("polbooks", "Polbooks"), ("polblogs", "Polblogs"),
              ("email-Eu", "email-Eu"), ("cora", "Cora"), ("citeseer", "Citeseer"), ("amazon", "SNAP-Amazon*"), ("youtube", "SNAP-Youtube*"), ("dblp", "SNAP-DBLP*")]
for metric, key in (("NMI", "NMI"), ("Q", "Q")):
    rows = []
    for g, label in real_names:
        d = J(f"{R}/real_{g}.json")
        if d: rows.append((label, {a: d["algs"].get(a, {}).get(key) for a in ALGS}))
    md.append(table(rows, ALGS, f"Real networks — {metric} (run)", "SNAP-* rows: our own induced subgraphs (3.9x larger than the paper's subsets), truth = first listed community; "
                    "an algorithm that timed out or exceeded memory shows —; avg/rank only over rows where all columns exist.\n"))
    summ[f"real_{metric}"] = {n: v for n, v in rows}
rows = []
for g, label in real_names:
    d = J(f"{R}/real_{g}.json")
    if d: rows.append((label, {a: d["algs"].get(a, {}).get("NMI_best", d["algs"].get(a, {}).get("NMI")) for a in ALGS}))
md.append(table(rows, ALGS, "Real networks — NMI, stochastic baselines (LPA, Louvain, Infomap*) at their BEST run (oracle, run)",
                "Most baseline-friendly reading: the best of 10 runs (3 for SNAP) chosen with the labels. Deterministic columns are unchanged."))
summ["real_NMI_best"] = {n: v for n, v in rows}
paper_cols = ["Fastgreedy", "LPA", "Louvain", "Walktrap", "NSA", "SimCMR", "TPOC-LPA (paper)"]
for tab, label, T in (("NMI", "T8", P.T8), ("Q", "T9", P.T9)):
    rows = []
    for g, nm in (("karate", "Karate"), ("dolphin", "Dolphin"), ("football", "Football")):
        rows.append((nm + " (paper)", dict(zip(paper_cols, T[g]))))
    md.append(table(rows, paper_cols, f"Same graphs, {tab} as published (paper)", "For reference: the paper's own table values for the three graphs identical to ours."))
# ---------------- LFR
for metric in ("NMI", "Q"):
    rows = []
    for net in ("N1", "N2", "N3", "N4", "N5", "N6"):
        d = J(f"{R}/lfr_{net}.json")
        if not d: continue
        seeds = d["per_seed"]; v = {}
        for a in ALGS:
            xs = [s[a][metric] for s in seeds.values() if a in s and metric in s[a]]
            v[a] = st.mean(xs) if len(xs) == len(seeds) else None
        rows.append((f"{net} (n={d['params']['n']}, mu={d['params']['mu']}, {len(seeds)} seeds)", v))
    base = list(rows)
    md.append(table(base, ALGS, f"LFR benchmark N1–N6 — {metric} (run)",
                    "Paper's Table 5 parameters (avg degree 20; cmin 15 instead of 10 for N1/N2); mean over 5 seeds; stochastic algorithms averaged over 3 runs per graph."))
    if len(base) == 6:
        grp = [("Avg-1000 (N1,N2)", (0, 1)), ("Avg-5000 (N3,N4)", (2, 3)), ("Avg-10000 (N5,N6)", (4, 5)), ("Avg-0.1 (N1,N3,N5)", (0, 2, 4)), ("Avg-0.6 (N2,N4,N6)", (1, 3, 5))]
        ex = [(nm, {a: (st.mean(base[i][1][a] for i in ix) if all(base[i][1][a] is not None for i in ix) else None) for a in ALGS}) for nm, ix in grp]
        md.append(table(ex, ALGS, f"LFR grouped averages — {metric} (run)", ""))
    summ[f"lfr_{metric}"] = {n: v for n, v in rows}
rows = []
for net in ("N1", "N2", "N3", "N4", "N5", "N6"):
    d = J(f"{R}/lfr_{net}.json")
    if not d: continue
    seeds = d["per_seed"]
    rows.append((f"{net} (n={d['params']['n']}, mu={d['params']['mu']})", {a: st.mean(s_[a].get("NMI_best", s_[a].get("NMI")) for s_ in seeds.values()) for a in ALGS}))
md.append(table(rows, ALGS, "LFR N1–N6 — NMI, stochastic baselines at their BEST of 3 runs per graph (oracle, run)", ""))
pc = ["Fastgreedy", "LPA", "Louvain", "Walktrap", "NSA", "SimCMR", "TPOC-LPA"]
for lab, T in (("NMI (paper Table 10)", P.T10), ("Q (paper Table 11)", P.T11)):
    rows = [(f"N{i+1}", {a: T[a][i] for a in pc}) for i in range(6)]
    md.append(table(rows, pc, f"LFR N1–N6, {lab}", "As published; the paper's graphs are not available, so this is not directly comparable to the run table above."))
# ---------------- EQ
EQC = ["SLPA", "TPOC-LPA (repro)", "Ours"]
rows = []; eqnames = ["karate", "dolphin", "jazz", "lesmis", "cora", "power", "facebook", "ca-grqc", "lastfm", "ca-hepth", "pgp", "deezer"]
for g in eqnames:
    d = J(f"{R}/eq_{g}.json")
    if not d: continue
    b = "b0.8"
    rows.append((f"{g} (n={d['n']}, m={d['m']})", {"SLPA": d["SLPA"]["EQ"], "TPOC-LPA (repro)": d[f"TPOC-LPA (repro) {b}"]["EQ"], "Ours": d[f"Ours {b}"]["EQ"]}))
md.append(table(rows, EQC, "EQ on the paper's real networks — beta=0.8 (run)", "SLPA: r scanned 0.05–0.5, best EQ kept (label-free choice). Karate/Dolphin at beta=1.0 are in results/eq_*.json."))
eqc = ["GREESE", "MDPA", "SLPA", "WLC", "UMSTMO", "TPOC-LPA"]
md.append(table([(g, dict(zip(eqc, P.T12[g]))) for g in eqnames], eqc, "EQ as published (paper Table 12)"))
summ["eq"] = {n: v for n, v in rows}
# ---------------- overlap
OC = ["SLPA", "TPOC-LPA (repro)", "Ours"]; fam = {"R1": "N-1000-0.1", "R2": "N-1000-0.3", "R3": "N-5000-0.1", "R4": "N-5000-0.3", "R5": "N-10000-0.1", "R6": "N-10000-0.3"}
for metric, T, tname in (("LFK", P.T13, "NMI (LFK) — paper Table 13"), ("NMImax", P.T14, "NMI_max — paper Table 14"), ("Omega", P.T15, "Omega — paper Table 15")):
    rows = []
    for r in ("R1", "R2", "R3", "R4", "R5", "R6"):
        per = {a: [] for a in OC}; nfile = 0
        for s in (1, 2):
            d = J(f"{R}/overlap_{r}_s{s}.json")
            if d and len(d) == 7:
                nfile += 1
                for om, row in d.items():
                    for a in OC: per[a].append(row[a][metric])
        if nfile: rows.append((f"{r} = {fam[r]} ({nfile} seeds x 7 om)", {a: st.mean(per[a]) for a in OC}))
    if len(rows) == 6:
        rows.append(("Avg", {a: st.mean(v[a] for _, v in rows) for a in OC}))
    md.append(table([r for r in rows if r[0] != "Avg"], OC, f"Overlapping LFR R1–R6, {metric} averaged over om=2..8 (run)", "SLPA: best r per graph by LFK-NMI (favours the baseline)."))
    pcols = ["GREESE", "MDPA", "SLPA", "WLC", "UMSTMO", "TPOC-LPA"]
    md.append(table([(f"{r} = {fam[r]}", dict(zip(pcols, T[r]))) for r in ("R1", "R2", "R3", "R4", "R5", "R6")], pcols, f"Same, as published ({tname})"))
    summ[f"ov_{metric}"] = {n: v for n, v in rows}
open(f"{H}/RESULTS.md", "w").write("\n".join(md)); json.dump(summ, open(f"{R}/summary.json", "w"), indent=1)
print("wrote RESULTS.md")
