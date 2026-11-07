"""Builds main.tex (single-file IEEEtran paper). Every table number is read from the experiment outputs
(../kaggle-tpoc-lpa/phase6_baselines/results/summary.json); nothing in the tables is typed by hand except the ablation table (see comment)."""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{H}/../kaggle-tpoc-lpa/phase6_baselines/results/summary.json"))
import sys; sys.path.insert(0, f"{H}/../kaggle-tpoc-lpa/phase6_baselines")
import paper_numbers as PN

ALGS = ["Fastgreedy", "LPA", "Louvain", "Walktrap", "Infomap*", "TPOC-LPA (repro)", "Ours"]
HDR = {"Fastgreedy": "FG", "LPA": "LPA", "Louvain": "Louv.", "Walktrap": "Walk.", "Infomap*": "Info.$^{*}$", "TPOC-LPA (repro)": "TPOC$^{\\dagger}$", "Ours": "SN-LPA"}

def f4(x): return "--" if x is None else f"{x:.4f}"
def tex_table(star, caption, label, rows, cols, hdr, colspec=None, avg=True, rank=True, size="\\footnotesize", bold=True, pos="t"):
    """rows: list of (name, dict). Bold best per row; Avg / mean-rank rows over complete rows."""
    env = "table*" if star else "table"
    colspec = colspec or ("l" + "r" * len(cols))
    out = [f"\\begin{{{env}}}[{pos}]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size, f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
           "Network & " + " & ".join(hdr[c] for c in cols) + " \\\\", "\\midrule"]
    for name, v in rows:
        vals = [v.get(c) for c in cols]; best = max([x for x in vals if x is not None], default=None)
        cells = []
        for x in vals:
            t = f4(x)
            if bold and x is not None and best is not None and x >= best - 1e-12: t = f"\\textbf{{{t}}}"
            cells.append(t)
        out.append(f"{name} & " + " & ".join(cells) + " \\\\")
    full = [v for _, v in rows if all(v.get(c) is not None for c in cols)]
    if full and avg:
        out.append("\\midrule")
        out.append("Mean & " + " & ".join(f4(sum(v[c] for v in full) / len(full)) for c in cols) + " \\\\")
    if full and rank:
        def rk(v, c): return 1 + sum(1 for d in cols if v[d] > v[c] + 1e-12)
        out.append("Mean rank & " + " & ".join(f"{sum(rk(v, c) for v in full) / len(full):.2f}" for c in cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(out)

def clean(n): return n.replace("*", "$^{*}$").replace("_", "\\_")

# ---- tables
real_rows = [(clean(n), v) for n, v in S["real_NMI"].items()]
T_REAL = tex_table(True, "NMI of different algorithms on real networks.", "tab:real", real_rows, ALGS, HDR, size="\\scriptsize")
lfr_rows = [(re.sub(r" \(n=(\d+), mu=([\d.]+), 5 seeds\)", r" ($n{=}\1$, $\\mu{=}\2$)", n), v) for n, v in S["lfr_NMI"].items()]
T_LFR = tex_table(True, "NMI of different algorithms on synthetic networks.", "tab:lfr", lfr_rows, ALGS, HDR, size="\\scriptsize")
EQC = ["SLPA", "TPOC-LPA (repro)", "Ours"]; EQH = {"SLPA": "SLPA", "TPOC-LPA (repro)": "TPOC$^{\\dagger}$", "Ours": "SN-LPA"}
eq_rows = [(re.sub(r" \(n=(\d+), m=(\d+)\)", r" ($n{=}\1$)", n).replace("ca-grqc", "CA-GrQc").replace("ca-hepth", "CA-HepTh").replace("lastfm", "LastFM").replace("pgp", "PGP")
            .replace("deezer", "Deezer").replace("facebook", "Facebook").replace("power", "Power").replace("jazz", "Jazz").replace("lesmis", "Lesmis").replace("karate", "Karate")
            .replace("dolphin", "Dolphin").replace("cora", "Cora"), v) for n, v in S["eq"].items()]
T_EQ = tex_table(False, "EQ of different algorithms on real networks.", "tab:eq", eq_rows, EQC, EQH,
                 size="\\scriptsize", pos="H")
OV = {}
fam = {"R1": "R1 ($n{=}10^3,\\mu{=}0.1$)", "R2": "R2 ($n{=}10^3,\\mu{=}0.3$)", "R3": "R3 ($n{=}5{\\cdot}10^3,\\mu{=}0.1$)", "R4": "R4 ($n{=}5{\\cdot}10^3,\\mu{=}0.3$)",
       "R5": "R5 ($n{=}10^4,\\mu{=}0.1$)", "R6": "R6 ($n{=}10^4,\\mu{=}0.3$)"}
for m in ("LFK", "NMImax", "Omega"):
    OV[m] = {k.split(" ")[0]: v for k, v in S[f"ov_{m}"].items()}
ov_rows = []
for r in ("R1", "R2", "R3", "R4", "R5", "R6"):
    row = {}
    for m, tag in (("LFK", "L"), ("NMImax", "M"), ("Omega", "O")):
        for a in EQC: row[f"{a}|{tag}"] = OV[m][r][a]
    ov_rows.append((fam[r], row))
cols = [f"{a}|{t}" for t in "LMO" for a in EQC]
hdr = {c: EQH[c.split("|")[0]] for c in cols}
body = ["\\begin{table*}[t]", "\\centering", "\\caption{Average LFK-NMI, $\\mathrm{NMI}_{\\max}$ and Omega values of different algorithms on overlapping networks, $om$ from 2 to 8.}", "\\label{tab:ov}", "\\scriptsize",
        "\\begin{tabular}{lccc|ccc|ccc}", "\\toprule", " & \\multicolumn{3}{c|}{LFK-NMI} & \\multicolumn{3}{c|}{$\\mathrm{NMI}_{\\max}$} & \\multicolumn{3}{c}{Omega} \\\\",
        "Family & " + " & ".join(hdr[c] for c in cols) + " \\\\", "\\midrule"]
for name, v in ov_rows:
    cells = []
    for t in "LMO":
        grp = [v[f"{a}|{t}"] for a in EQC]; b = max(grp)
        cells += [("\\textbf{%.4f}" if x >= b - 1e-12 else "%.4f") % x for x in grp]
    body.append(f"{name} & " + " & ".join(cells) + " \\\\")
body.append("\\midrule")
cells = []
for t in "LMO":
    for a in EQC: cells.append("%.4f" % (sum(v[f"{a}|{t}"] for _, v in ov_rows) / 6))
body.append("Mean & " + " & ".join(cells) + " \\\\")
body += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
T_OV = "\n".join(body)

pc = ["Fastgreedy", "LPA", "Louvain", "Walktrap", "NSA", "SimCMR", "TPOC-LPA"]
PH = {"Fastgreedy": "FG", "LPA": "LPA", "Louvain": "Louv.", "Walktrap": "Walk.", "NSA": "NSA", "SimCMR": "SimCMR", "TPOC-LPA": "TPOC-LPA"}
qrows = [("Karate", dict(zip(pc, PN.T8["karate"]))), ("Dolphin", dict(zip(pc, PN.T8["dolphin"]))), ("Football", dict(zip(pc, PN.T8["football"])))]
for i, nm in enumerate(["N1", "N2", "N3", "N4", "N5", "N6"]):
    qrows.append((nm, {a: PN.T10[a][i] for a in pc}))
T_PAPER = tex_table(False, "NMI as published in~\\cite{kong2026} (not re-run).",
                    "tab:paper", qrows, pc, PH, size="\\scriptsize\\setlength{\\tabcolsep}{2.4pt}", avg=False, rank=False, pos="H")

# Ablation: values from the development runs recorded in the project log (15 seeds for LFR; email-Eu single graph). Typed by hand.
ABL = [("TPOC-LPA (re-implemented baseline)", 0.9311, 0.0513, 0.0000),
       ("+ M1--M4 (motif, guard, size-corr., granularity)", 0.9637, 0.8534, 0.6381),
       ("+ M6, map-equation move score", 0.9776, 0.8759, 0.6554),
       ("+ M6, Synwalk move score", 0.9830, 0.8917, 0.6717)]
abl = ["\\begin{table}[H]", "\\centering", "\\caption{Effect of the modifications on NMI.}", "\\label{tab:abl}",
       "\\scriptsize", "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{38mm}rrr}", "\\toprule", "Configuration & $\\mu{=}0.6$ & $\\mu{=}0.7$ & email-Eu \\\\", "\\midrule"]
abl += [f"{n} & {a:.4f} & {b:.4f} & {c:.4f} \\\\" for n, a, b, c in ABL] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
T_ABL = "\n".join(abl)


# ---- dataset table and run-time table (read from the real-graph result files)
R = f"{H}/../kaggle-tpoc-lpa/phase6_baselines/results"
meta = [("karate", "Karate", "two factions"), ("dolphin", "Dolphin", "four groups (reconstructed)"), ("football", "Football", "conferences"), ("polbooks", "Polbooks", "political leaning"),
        ("polblogs", "Polblogs", "political leaning"), ("email-Eu", "email-Eu", "departments"), ("cora", "Cora", "paper classes"), ("citeseer", "Citeseer", "paper classes"),
        ("amazon", "SNAP-Amazon", "first community"), ("youtube", "SNAP-Youtube", "first community"), ("dblp", "SNAP-DBLP", "first community")]
rows = ["\\begin{table}[H]", "\\centering", "\\caption{Real networks used in the experiments.}", "\\label{tab:data}", "\\scriptsize",
        "\\begin{tabular}{lrrr>{\\raggedright\\arraybackslash}p{23mm}}", "\\toprule", "Network & $n$ & $m$ & Classes & Truth \\\\", "\\midrule"]
for g, nm, tr in meta:
    d = json.load(open(f"{R}/real_{g}.json"))
    rows.append(f"{nm} & {d['n']:,} & {d['m']:,} & {d['classes']:,} & {tr} \\\\".replace(",", "{,}"))
rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
T_DATA = "\n".join(rows)

# =================== latency and scalability (from results/latency_*.json)
import numpy as np
LAT = json.load(open(f"{R}/latency_lfr.json")); LREAL = json.load(open(f"{R}/latency_real.json"))
NS = LAT["N"]
PAPER_T16 = {400: 0.281, 800: 0.505, 1600: 0.965, 3200: 2.386, 6400: 5.974, 12800: 13.61, 25600: 24.326}   # Table 16 of Kong et al. (2026), their hardware
LAT_ROWS = [("Fastgreedy", "FG"), ("LPA", "LPA"), ("Louvain", "Louvain"), ("Walktrap", "Walktrap"), ("Infomap*", "Infomap$^{*}$"), ("SLPA", "SLPA"),
            ("TPOC-LPA (repro)", "TPOC$^{\\dagger}$"), ("Ours", "SN-LPA")]
def ftime(v): return f"{v:.3f}" if v < 1 else (f"{v:.2f}" if v < 100 else f"{v:.1f}")
def slope(ts): return float(np.polyfit(np.log(NS), np.log(ts), 1)[0])
tt = {a: [LAT["time"][a][str(n)]["median"] for n in NS] for a, _ in LAT_ROWS}
tt["paper"] = [PAPER_T16[n] for n in NS]
rows = ["\\begin{table*}[t]", "\\centering", "\\caption{Execution time (s) of different algorithms for different $N$.}", "\\label{tab:lat}", "\\scriptsize",
        "\\begin{tabular}{l" + "r" * len(NS) + "r}", "\\toprule", "Algorithm & " + " & ".join(f"{n:,}".replace(",", "{,}") for n in NS) + " & Slope \\\\", "\\midrule"]
for a, lab in LAT_ROWS:
    best = [min(tt[x][i] for x, _ in LAT_ROWS) for i in range(len(NS))]
    cells = [("\\textbf{%s}" if tt[a][i] <= best[i] * 1.0000001 else "%s") % ftime(tt[a][i]) for i in range(len(NS))]
    rows.append(f"{lab} & " + " & ".join(cells) + f" & {slope(tt[a]):.2f} \\\\")
rows += ["\\midrule", "TPOC-LPA~\\cite{kong2026} (published) & " + " & ".join(ftime(v) for v in tt["paper"]) + f" & {slope(tt['paper']):.2f} \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table*}"]
T_LAT = "\n".join(rows)

# running-time figure (pgfplots, log-log)
styles = {"Fastgreedy": "mark=square*,color=gray", "LPA": "mark=triangle*,color=green!50!black", "Louvain": "mark=diamond*,color=teal", "Walktrap": "mark=pentagon*,color=brown",
          "Infomap*": "mark=star,color=magenta", "SLPA": "mark=x,color=purple", "TPOC-LPA (repro)": "mark=o,color=blue", "Ours": "mark=*,color=red,very thick"}
fig = ["\\begin{figure}[t]", "\\centering", "\\begin{tikzpicture}", "\\begin{loglogaxis}[width=0.98\\columnwidth,height=5.6cm,xlabel={Number of nodes $N$},ylabel={Time (s)},",
       "legend style={font=\\tiny,at={(0.5,-0.2)},anchor=north,legend columns=3,cells={anchor=west},draw=none},tick label style={font=\\scriptsize},label style={font=\\small},grid=both,grid style={gray!25}]"]
for a, lab in LAT_ROWS:
    coords = " ".join(f"({n},{tt[a][i]:.5f})" for i, n in enumerate(NS))
    fig += [f"\\addplot[{styles[a]},mark size=1.6pt] coordinates {{{coords}}};", f"\\addlegendentry{{{lab}}}"]
coords = " ".join(f"({n},{tt['paper'][i]})" for i, n in enumerate(NS))
fig += [f"\\addplot[black,dashed,mark=none] coordinates {{{coords}}};", "\\addlegendentry{TPOC-LPA~\\cite{kong2026} (published)}", "\\end{loglogaxis}", "\\end{tikzpicture}",
        "\\caption{Running time versus the number of nodes.}", "\\label{fig:lat}", "\\end{figure}"]
F_LAT = "\n".join(fig)

# per-stage timing of our pipeline
stg = LAT["stages"]
rows = ["\\begin{table}[t]", "\\centering", "\\caption{Execution time (s) of each stage of the proposed pipeline.}", "\\label{tab:stage}", "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{rrrrrrrr}", "\\toprule", "$N$ & $NI$ & Stage 1 & Stage 2 & M4 & M5+M6 & Stage 3 & Total \\\\", "\\midrule"]
for n in NS:
    g = stg[str(n)]
    rows.append(f"{n:,} & ".replace(",", "{,}") + " & ".join(ftime(g[k]) for k in ["t_node_importance", "t_stage1", "t_stage2", "t_refine", "t_merge", "t_stage3", "t_total"]) + " \\\\")
rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
T_STAGE = "\n".join(rows)

# real graphs
LR = [("karate", "Karate"), ("dolphin", "Dolphin"), ("football", "Football"), ("polblogs", "Polblogs"), ("email-Eu", "email-Eu"), ("cora", "Cora"), ("citeseer", "Citeseer"),
      ("amazon", "SNAP-Amazon"), ("youtube", "SNAP-Youtube"), ("dblp", "SNAP-DBLP")]
LALG = ["Fastgreedy", "LPA", "Louvain", "Walktrap", "Infomap*", "TPOC-LPA (repro)", "Ours"]
def ft2(v): return "$<$0.01" if v < 0.005 else (f"{v:.2f}" if v < 100 else f"{v:.0f}")
rows = ["\\begin{table}[t]", "\\centering", "\\caption{Execution time (s) of different algorithms on real networks.}", "\\label{tab:latreal}", "\\scriptsize\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular}{l" + "r" * len(LALG) + "}", "\\toprule", "Network & " + " & ".join(HDR[a] for a in LALG) + " \\\\", "\\midrule"]
for g, nm in LR:
    if g not in LREAL: continue
    cells = [ft2(LREAL[g]["time"][a]["median"]) for a in LALG]
    rows.append(f"{clean(nm)} & " + " & ".join(cells) + " \\\\")
rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
T_LATREAL = "\n".join(rows)

# analysis text, numbers read from the measurements
N0, N1 = NS[0], NS[-1]
t = lambda a, n=N1: LAT["time"][a][str(n)]["median"]
s1share = stg[str(N1)]["t_stage1"] / stg[str(N1)]["t_total"]; s1share0 = stg[str(N0)]["t_stage1"] / stg[str(N0)]["t_total"]
extra = (t("Ours") - t("TPOC-LPA (repro)")) / t("TPOC-LPA (repro)")
LAT_TEXT = (
 f"We repeat the scalability experiment of Sec.~5.3 of~\\cite{{kong2026}}: overlapping LFR graphs with average degree 5, maximum degree 10, $\\mu{{=}}0.1$, community sizes 10 to 20, $om{{=}}2$, $on{{=}}10$ and $N$ from {N0} to {N1:,} nodes, doubling each time. "
 f"Every algorithm runs on the same graph, one after the other, on an otherwise idle machine; the time is the wall-clock time of the call (including, for the igraph methods, the conversion of the graph), and we report the median of {LAT['reps']} runs. "
 f"Table~\\ref{{tab:lat}} and Fig.~\\ref{{fig:lat}} show the result; the last column is the slope of a least-squares line on the logarithms of $N$ and time, that is the empirical growth exponent (1 means linear growth).\n\n"
 "@@T_LAT@@\n\n@@F_LAT@@\n\n"
 f"At $N={N1:,}$ our pipeline needs {t('Ours'):.2f}\\,s, which is {extra * 100:.0f}\\% more than our re-implementation of the baseline ({t('TPOC-LPA (repro)'):.2f}\\,s), about {t('Ours') / t('Infomap*'):.0f} times the time of Infomap ({t('Infomap*'):.2f}\\,s), about {t('Ours') / t('Louvain'):.0f} times that of Louvain ({t('Louvain'):.3f}\\,s) and {t('Ours') / t('Walktrap'):.1f} times that of Walktrap ({t('Walktrap'):.2f}\\,s); it is faster than SLPA ({t('SLPA'):.2f}\\,s). "
 "LPA, Louvain, Fastgreedy, Walktrap and Infomap are implemented in C or C++, whereas TPOC-LPA, our pipeline and SLPA are pure Python, so most of the gap to the first group is the implementation language and not the algorithm; we did not optimise our code. "
 f"The growth exponent of our pipeline is {slope(tt['Ours']):.2f}, against {slope(tt['TPOC-LPA (repro)']):.2f} for our baseline re-implementation and {slope(tt['paper']):.2f} for the times published in~\\cite{{kong2026}}, whose claim is that the algorithm is close to linear. "
 "Our baseline therefore does not reproduce a linear growth; the published times come from a different implementation and machine, so we use them only for their trend.\n\n"
 "Table~\\ref{tab:stage} explains where the time goes. "
 f"Stage~1 takes {s1share * 100:.0f}\\% of the total at $N={N1:,}$ and only {s1share0 * 100:.0f}\\% at $N={N0}$. In our literal reading of Stage~1 the tightness--sparsity value of the growing community is recomputed from its members for every candidate neighbour, so a cost that depends on the community size appears inside the loop and makes this stage grow faster than linearly. "
 f"The modifications themselves (the second Stage~2 run of the granularity rule, the merge pass and the moves) add {stg[str(N1)]['t_refine'] + stg[str(N1)]['t_merge']:.2f}\\,s at the largest size, and none of them dominates. "
 "An incremental update of the community's tightness would reduce the Stage~1 cost; we have not done this.\n\n"
 "@@T_STAGE@@\n\n"
 "Table~\\ref{tab:latreal} gives the time on the real networks, with the same protocol (3 runs, median; one run for the cases that take longer than 20\\,s). "
 f"Our pipeline needs about the same time as the baseline re-implementation, from {LREAL['karate']['time']['Ours']['median'] * 1000:.0f}\\,ms on Karate to {LREAL['dblp']['time']['Ours']['median']:.0f}\\,s on SNAP-DBLP ({LREAL['dblp']['n']:,} nodes), where the baseline needs {LREAL['dblp']['time']['TPOC-LPA (repro)']['median']:.0f}\\,s. "
 f"The C implementations of LPA, Louvain and Infomap need at most {max(LREAL[g]['time'][x]['median'] for g in LREAL for x in ('LPA', 'Louvain', 'Infomap*')):.1f}\\,s on any of these graphs and Fastgreedy at most {max(LREAL[g]['time']['Fastgreedy']['median'] for g in LREAL):.0f}\\,s; Walktrap needs {LREAL['dblp']['time']['Walktrap']['median']:.0f}\\,s on SNAP-DBLP and {LREAL['youtube']['time']['Walktrap']['median']:.0f}\\,s on SNAP-Youtube. On the largest graphs a run of our pipeline is therefore of the order of a minute, which is acceptable for offline analysis but far from the speed of the C methods.\n\n"
 "@@T_LATREAL@@\n")

tpl = open(f"{H}/template.tex").read()
LAT_TEXT = LAT_TEXT.replace("@@T_LAT@@", T_LAT).replace("@@F_LAT@@", F_LAT).replace("@@T_STAGE@@", T_STAGE).replace("@@T_LATREAL@@", T_LATREAL)
for k, v in {"@@T_REAL@@": T_REAL, "@@T_LFR@@": T_LFR, "@@T_EQ@@": T_EQ, "@@T_OV@@": T_OV, "@@T_PAPER@@": T_PAPER, "@@T_ABL@@": T_ABL, "@@T_DATA@@": T_DATA, "@@LAT_SECTION@@": LAT_TEXT}.items():
    tpl = tpl.replace(k, v)
open(f"{H}/main.tex", "w").write(tpl); print("wrote main.tex", len(tpl))
