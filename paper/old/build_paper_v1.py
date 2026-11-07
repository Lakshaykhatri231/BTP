"""Builds main.tex (single-file IEEEtran paper). Every table number is read from the experiment outputs
(../kaggle-tpoc-lpa/phase6_baselines/results/summary.json); nothing in the tables is typed by hand except the ablation table (see comment)."""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{H}/../kaggle-tpoc-lpa/phase6_baselines/results/summary.json"))
import sys; sys.path.insert(0, f"{H}/../kaggle-tpoc-lpa/phase6_baselines")
import paper_numbers as PN

ALGS = ["Fastgreedy", "LPA", "Louvain", "Walktrap", "Infomap*", "TPOC-LPA (repro)", "Ours"]
HDR = {"Fastgreedy": "FG", "LPA": "LPA", "Louvain": "Louv.", "Walktrap": "Walk.", "Infomap*": "Info.$^{*}$", "TPOC-LPA (repro)": "TPOC$^{\\dagger}$", "Ours": "Ours"}

def f4(x): return "--" if x is None else f"{x:.4f}"
def tex_table(star, caption, label, rows, cols, hdr, colspec=None, avg=True, rank=True, size="\\footnotesize", bold=True):
    """rows: list of (name, dict). Bold best per row; Avg / mean-rank rows over complete rows."""
    env = "table*" if star else "table"
    colspec = colspec or ("l" + "r" * len(cols))
    out = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size, f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
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
T_REAL = tex_table(True, "NMI on real networks. All columns re-run by us on the same graphs. FG = Fastgreedy, Walk. = Walktrap, TPOC$^{\\dagger}$ = our "
                   "re-implementation of the baseline TPOC-LPA, Info.$^{*}$ = Infomap (extra baseline, not in~\\cite{kong2026}). Stochastic methods: mean of 10 runs (3 on SNAP). "
                   "SNAP$^{*}$ rows use our own induced subgraphs, truth = first listed community. Mean / rank over the 11 rows.", "tab:real", real_rows, ALGS, HDR, size="\\scriptsize")
lfr_rows = [(re.sub(r" \(n=(\d+), mu=([\d.]+), 5 seeds\)", r" ($n{=}\1$, $\\mu{=}\2$)", n), v) for n, v in S["lfr_NMI"].items()]
T_LFR = tex_table(True, "NMI on the six LFR settings of~\\cite{kong2026} (Table~5 there; average degree 20; mean of 5 graphs per row; stochastic methods averaged over 3 runs). "
                  "Realised mixing is calibrated (Sec.~\\ref{sec:setup}); $c_{\\min}{=}15$ instead of 10 for $n{=}1000$.", "tab:lfr", lfr_rows, ALGS, HDR, size="\\scriptsize")
EQC = ["SLPA", "TPOC-LPA (repro)", "Ours"]; EQH = {"SLPA": "SLPA", "TPOC-LPA (repro)": "TPOC$^{\\dagger}$", "Ours": "Ours"}
eq_rows = [(re.sub(r" \(n=(\d+), m=(\d+)\)", r" ($n{=}\1$)", n).replace("ca-grqc", "CA-GrQc").replace("ca-hepth", "CA-HepTh").replace("lastfm", "LastFM").replace("pgp", "PGP")
            .replace("deezer", "Deezer").replace("facebook", "Facebook").replace("power", "Power").replace("jazz", "Jazz").replace("lesmis", "Lesmis").replace("karate", "Karate")
            .replace("dolphin", "Dolphin").replace("cora", "Cora"), v) for n, v in S["eq"].items()]
T_EQ = tex_table(False, "Extended modularity EQ ($\\beta{=}0.8$) on the networks of~\\cite{kong2026}. SLPA: threshold $r$ scanned in $[0.05,0.5]$, best EQ kept.", "tab:eq", eq_rows, EQC, EQH,
                 size="\\scriptsize")
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
body = ["\\begin{table*}[t]", "\\centering", "\\caption{Overlapping LFR networks R1--R6 of~\\cite{kong2026} (Table~7 there), averaged over $om=2..8$ and 2 generator seeds. "
        "Left to right: LFK-NMI, $\\mathrm{NMI}_{\\max}$, Omega. SLPA: $r\\in\\{0.01,\\dots,0.10\\}$ tuned per graph by LFK-NMI (favours the baseline).}", "\\label{tab:ov}", "\\scriptsize",
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
T_PAPER = tex_table(True, "NMI \\emph{as published} in~\\cite{kong2026} (not re-run). NSA and SimCMR have no public implementation, so they appear only here. "
                    "Our own re-run of the other columns is in Tables~\\ref{tab:real} and~\\ref{tab:lfr}; the two are not directly comparable (different NMI normalisation and graph instances, see Sec.~\\ref{sec:setup}).",
                    "tab:paper", qrows, pc, PH, size="\\scriptsize", avg=False, rank=False)

# Ablation: values from the development runs recorded in the project log (15 seeds for LFR; email-Eu single graph). Typed by hand.
ABL = [("TPOC-LPA (re-implemented baseline)", 0.9311, 0.0513, 0.0000),
       ("+ M1--M4 (motif, guard, size-corr., granularity)", 0.9637, 0.8534, 0.6381),
       ("+ M6, map-equation move score", 0.9776, 0.8759, 0.6554),
       ("+ M6, Synwalk move score", 0.9830, 0.8917, 0.6717)]
abl = ["\\begin{table}[t]", "\\centering", "\\caption{Cumulative effect of the modifications (development runs; M5, the merge pass, is off here). Columns 2--3: LFR, $n{=}1000$, 15 seeds, realised $\\mu$ as shown; column 4: email-Eu. NMI.}", "\\label{tab:abl}",
       "\\scriptsize", "\\begin{tabular}{p{38mm}rrr}", "\\toprule", "Configuration & $\\mu{=}0.6$ & $\\mu{=}0.7$ & email-Eu \\\\", "\\midrule"]
abl += [f"{n} & {a:.4f} & {b:.4f} & {c:.4f} \\\\" for n, a, b, c in ABL] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
T_ABL = "\n".join(abl)

tpl = open(f"{H}/template.tex").read()
for k, v in {"@@T_REAL@@": T_REAL, "@@T_LFR@@": T_LFR, "@@T_EQ@@": T_EQ, "@@T_OV@@": T_OV, "@@T_PAPER@@": T_PAPER, "@@T_ABL@@": T_ABL}.items():
    tpl = tpl.replace(k, v)
open(f"{H}/main.tex", "w").write(tpl); print("wrote main.tex", len(tpl))
