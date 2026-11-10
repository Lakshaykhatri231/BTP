"""Generates the illustration figures of the paper (TikZ) from runs of the real code.

Every partition, score and number drawn here is computed by kaggle-tpoc-lpa/tpoc_lpa.py on the
graph shown (see figlib.py; the step traces are checked against the library). The example graphs
are small planted-partition graphs (networkx.random_partition_graph, parameters below) chosen
because the behaviour discussed in each figure occurs on them; Karate and email-Eu are real data.

Run:  python3 make_figures.py   -> writes fig_*.tex next to this file (inlined by ../build_paper.py)
"""
import json, math, os
from figlib import *
from layoutopt import optimise

OUT = H
# validated categorical palette (dataviz reference palette, light mode): fixed order, never cycled
PAL = ["2A78D6", "EB6834", "1BAF7A", "EDA100", "6250D6", "E87BA4", "008300", "E34948", "9C6B4E", "A6A6A6", "17BECF"]   # last three: only for extra Stage-1 fragments
DARK_TEXT = {"1BAF7A", "EDA100", "E87BA4", "A6A6A6", "17BECF"}           # black numbers on light fills, white otherwise

PREAMBLE = "\n".join([f"\\definecolor{{pc{i}}}{{HTML}}{{{c}}}" for i, c in enumerate(PAL)] + [
    r"\definecolor{edgec}{HTML}{7F93BD}",
    r"\definecolor{arrowc}{HTML}{4A6FB5}",
    r"\definecolor{badc}{HTML}{C0392B}",
    r"\definecolor{goodc}{HTML}{1E7F4F}",
    r"\tikzset{",
    r"  gn/.style={circle, draw=black!85, line width=0.5pt, minimum size=5.2mm, inner sep=0pt, font=\sffamily\fontsize{7}{7}\selectfont},",
    r"  gns/.style={circle, draw=black!85, line width=0.4pt, minimum size=3.2mm, inner sep=0pt},",
    r"  ge/.style={draw=edgec, line width=0.55pt},",
    r"  barrow/.style={single arrow, fill=arrowc, draw=none, minimum height=8mm, inner xsep=0pt, inner ysep=1.25mm, single arrow head extend=1.5mm, single arrow head indent=0pt},",
    r"  ptitle/.style={font=\scriptsize, anchor=south, inner sep=1pt, align=center},",
    r"  pstat/.style={font=\scriptsize, anchor=north, inner sep=1pt, align=center},",
    r"  ring/.style={circle, draw=badc, line width=1.2pt, minimum size=7.2mm, inner sep=0pt},",
    r"  elab/.style={font=\sffamily\fontsize{6.2}{6.6}\selectfont, align=center, fill=white, fill opacity=0.9, text opacity=1, inner sep=0.6pt},",
    r"  note/.style={font=\scriptsize, align=center, inner sep=1.5pt},",
    r"  ibox/.style={font=\scriptsize, align=left, inner sep=4pt, anchor=north west},",
    r"}"])


def f3(x): return f"{x:.3f}"


# ------------------------------------------------------------------ layout and colours
class Layout:
    """Node positions for one graph, optimised for readability (layoutopt.py) once per panel size
    and cached in layout_cache.json, so every panel of a figure shows the graph the same way."""
    def __init__(self, G, small=False, label_edges=(), seed=0):
        self.G, self.small, self.label_edges, self.seed = G, small, tuple(label_edges), seed
    def at(self, W, H):
        key = json.dumps([sorted(map(list, self.G.edges())), round(W, 2), round(H, 2), self.small,
                          sorted(map(list, self.label_edges)), self.seed])
        if key not in _CACHE:
            if self.small:
                P, _ = optimise(self.G, W, H, dmin=0.6, clear=0.24, starts=2, iters=900 * self.G.number_of_nodes(), seed=self.seed)
            else:
                P, _ = optimise(self.G, W, H, dmin=1.05, clear=0.5, label_edges=self.label_edges, starts=4, seed=self.seed)
            _CACHE[key] = {str(v): p for v, p in P.items()}
            json.dump(_CACHE, open(CACHE_FILE, "w"))
        return {v: tuple(_CACHE[key][str(v)]) for v in self.G}


CACHE_FILE = os.path.join(OUT, "layout_cache.json")
_CACHE = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}


class FixedLayout:
    """Hand-placed positions (any units), scaled to fill the panel."""
    def __init__(self, P): self.P = P
    def at(self, W, H):
        xs = [p[0] for p in self.P.values()]; ys = [p[1] for p in self.P.values()]
        return {v: ((x - min(xs)) / (max(xs) - min(xs)) * W, (y - min(ys)) / (max(ys) - min(ys)) * H) for v, (x, y) in self.P.items()}


class ScaledLayout:
    """The same drawing as `base` at size (W0, H0), stretched to another panel size."""
    def __init__(self, base, W0, H0): self.base, self.W0, self.H0 = base, W0, H0
    def at(self, W, H):
        P = self.base.at(self.W0, self.H0)
        return {v: (x * W / self.W0, y * H / self.H0) for v, (x, y) in P.items()}


def layout(G, truth=None, small=False, label_edges=(), seed=0, fixed=None, **_):
    if fixed: return FixedLayout(fixed), (1.0, 1.0)
    return Layout(G, small, label_edges, seed), (1.0, 1.0)


def colours(ref, truth, others=()):
    """Colour per community label, the same in every panel of a figure. Labels of the reference
    partition take the colour of the true community they cover most; other labels get the next
    free colours in order of first appearance."""
    col, used = {}, set()
    groups = {}
    for v, c in ref.items(): groups.setdefault(c, []).append(v)
    for c, mem in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        tc = max(set(truth[v] for v in mem), key=lambda t: sum(truth[v] == t for v in mem))
        idx = sorted(set(truth.values())).index(tc)
        if idx in used: idx = next(i for i in range(len(PAL)) if i not in used)
        col[c] = idx; used.add(idx)
    for lab in others:
        for v in sorted(lab, key=lambda v: (str(lab[v]), v)):
            c = lab[v]
            if c not in col:
                idx = next(i for i in range(len(PAL)) if i not in used); col[c] = idx; used.add(idx)
    return col


# ------------------------------------------------------------------ TikZ pieces
class Fig:
    def __init__(self): self.lines = []; self.n = 0
    def add(self, s): self.lines.append(s)

    def panel(self, G, pos, lab, col, x0, y0, w, h, title=None, stat=None, rings=(), overlap=None,
              elabels=None, numbers=True, frame=None, small=False, cores=(), notes=None, hl_edges=()):
        """Draw one graph panel with lower-left corner (x0, y0) and size w x h (cm)."""
        self.n += 1; p = f"P{self.n}"
        pad = 0.25 if not small else 0.22
        P = pos.at(w - 2 * pad, h - 2 * pad)
        X = lambda v: x0 + pad + P[v][0]
        Y = lambda v: y0 + pad + P[v][1]
        if frame:
            self.add(f"\\draw[{frame}, rounded corners=2pt] ({x0:.3f},{y0:.3f}) rectangle ({x0 + w:.3f},{y0 + h:.3f});")
        hl = {frozenset(e) for e in hl_edges}
        for u, v in G.edges():
            st_ = "ge, line width=1.5pt, draw=black!55" if frozenset((u, v)) in hl else "ge"
            self.add(f"\\draw[{st_}] ({X(u):.3f},{Y(u):.3f}) -- ({X(v):.3f},{Y(v):.3f});")
        import re as _re
        placed = []
        rn = 0.15 if small else 0.40   # node radius plus ring
        def vis(t): return len(_re.sub(r"\\[a-zA-Z]+|[{}$]|\\,", "", t))
        skip = set(); leaders = []
        def clearance(x, y, hw, hh, excl=None):
            c = min((max(abs(x - X(w)) - hw, abs(y - Y(w)) - hh) - rn for w in G if w != excl), default=9)
            for (px, py, pw, ph) in placed:
                c = min(c, max(abs(x - px) - hw - pw, abs(y - py) - hh - ph))
            for (e1, e2) in G.edges():   # and clear of the drawn edges
                if (e1, e2) in skip or (e2, e1) in skip: continue
                ax_, ay_, bx_, by_ = X(e1), Y(e1), X(e2), Y(e2)
                dx_, dy_ = bx_ - ax_, by_ - ay_; L2 = dx_ * dx_ + dy_ * dy_ or 1e-9
                t_ = max(0.0, min(1.0, ((x - ax_) * dx_ + (y - ay_) * dy_) / L2))
                c = min(c, math.hypot(x - ax_ - t_ * dx_, y - ay_ - t_ * dy_) - hh - 0.02)
            # stay inside the panel, clear of its title and statistics lines
            # labels may use the empty gutter beside the panel, but not the band where the arrows are
            side = 0.75 if abs(y - (y0 + h / 2)) > 0.8 else 0.0
            return min(c, (y0 + h - 0.05) - (y + hh), (y - hh) - (y0 + 0.05), (x - hw) - (x0 - side), (x0 + w + side) - (x + hw))
        for (u, v), t in (elabels or {}).items():   # edge label: slide along the edge to the freest spot
            parts = t.split("\\\\")
            hw, hh = 0.045 * max(vis(x_) for x_ in parts) + 0.06, 0.12 * len(parts)
            skip.clear(); skip.add((u, v))
            best = max(((f, X(u) + f * (X(v) - X(u)), Y(u) + f * (Y(v) - Y(u))) for f in (0.5, 0.55, 0.45, 0.6, 0.4, 0.65, 0.35, 0.7, 0.3, 0.75)),
                       key=lambda q: clearance(q[1], q[2], hw, hh))
            placed.append((best[1], best[2], hw, hh))
            # written along its own edge (sloped), so it is clear which edge it belongs to
            self.add(f"\\path ({X(u):.3f},{Y(u):.3f}) -- ({X(v):.3f},{Y(v):.3f}) node[elab, sloped, pos={best[0]:.2f}] {{{t}}};")
        for v in cores:
            self.add(f"\\node[circle, draw=black, line width=1.4pt, minimum size=7.4mm, inner sep=0pt] at ({X(v):.3f},{Y(v):.3f}) {{}};")
        for v in rings:
            self.add(f"\\node[ring{', minimum size=5mm, line width=1pt' if small else ''}] at ({X(v):.3f},{Y(v):.3f}) {{}};")
        style = "gns" if small else "gn"
        r = 0.16 if small else 0.26
        for v in G:
            labs = sorted(overlap[v], key=lambda c: col[c]) if overlap and len(overlap[v]) > 1 else [lab.get(v)]
            if labs[0] is None:   # unlabelled node (Stage 1 in progress)
                self.add(f"\\node[{style}, fill=white, text=black] ({p}n{v}) at ({X(v):.3f},{Y(v):.3f}) {{{v if numbers else ''}}};")
                continue
            if len(labs) == 1:
                c = PAL[col[labs[0]]]
                txt = "black" if c in DARK_TEXT else "white"
                body = str(v) if numbers else ""
                self.add(f"\\node[{style}, fill=pc{col[labs[0]]}, text={txt}] ({p}n{v}) at ({X(v):.3f},{Y(v):.3f}) {{{body}}};")
            else:   # overlapping node: one sector per community, like Fig. 6 of the base paper
                k_ = len(labs)
                for i, c in enumerate(labs):
                    a0, a1 = 90 + 360 * i / k_, 90 + 360 * (i + 1) / k_
                    self.add(f"\\fill[pc{col[c]}] ({X(v):.3f},{Y(v):.3f}) -- ++({a0:.1f}:{r}) arc ({a0:.1f}:{a1:.1f}:{r}) -- cycle;")
                self.add(f"\\node[{style}, fill=none, text=black] ({p}n{v}) at ({X(v):.3f},{Y(v):.3f}) {{{v if numbers else ''}}};")
        for v, t in sorted((notes or {}).items(), key=lambda kv: -sum(1 for w in G if math.hypot(X(w) - X(kv[0]), Y(w) - Y(kv[0])) < 1.2)):   # most crowded first
            hw, hh = 0.05 * vis(t) + 0.06, 0.13
            skip.clear()
            ro = 0.43 if v in set(rings) else r + 0.05   # touch the node (or its red ring)
            cands = [(X(v) + dx * (ro + hw), Y(v) + dy * (ro + hh), 0) for dx, dy in
                     ((1, 0), (-1, 0), (0, 1), (0, -1), (.75, .75), (-.75, .75), (.75, -.75), (-.75, -.75))]
            def seg_d(px_, py_, a_, b_):
                dx_, dy_ = b_[0] - a_[0], b_[1] - a_[1]; L2 = dx_ * dx_ + dy_ * dy_ or 1e-9
                t_ = max(0.0, min(1.0, ((px_ - a_[0]) * dx_ + (py_ - a_[1]) * dy_) / L2))
                return math.hypot(px_ - a_[0] - t_ * dx_, py_ - a_[1] - t_ * dy_)
            def score(q):
                c = clearance(q[0], q[1], hw, hh, excl=v)
                for (a_, b_) in leaders:   # keep clear of the other leader lines
                    c = min(c, seg_d(q[0], q[1], a_, b_) - hh)
                # the label must be clearly nearer to its own node than to any other node
                d_own = math.hypot(q[0] - X(v), q[1] - Y(v))
                d_oth = min((math.hypot(q[0] - X(w_), q[1] - Y(w_)) for w_ in G if w_ != v), default=9)
                return min(c, d_oth - d_own - 0.25) - 0.02 * d_own
            best = max(cands, key=score)
            placed.append((best[0], best[1], hw, hh))
            self.add(f"\\node[elab, font=\\sffamily\\fontsize{{6.5}}{{6.5}}\\selectfont] at ({best[0]:.3f},{best[1]:.3f}) {{{t}}};")
        if title:
            self.add(f"\\node[ptitle] at ({x0 + w / 2:.3f},{y0 + h:.3f}) {{{title}}};")
        if stat:
            self.add(f"\\node[pstat] at ({x0 + w / 2:.3f},{y0:.3f}) {{{stat}}};")
        return X, Y

    def arrow(self, x, y, label=None, rot=0, below=None, length=8, tw=None):
        self.add(f"\\node[barrow, rotate={rot}, minimum height={length}mm] at ({x:.3f},{y:.3f}) {{}};")
        if label:
            self.add(f"\\node[note, anchor=south] at ({x:.3f},{y + 0.28:.3f}) {{{label}}};")
        if below:
            self.add(f"\\node[note, anchor=north{(', text width=' + str(tw) + 'cm') if tw else ''}] at ({x:.3f},{y - 0.28:.3f}) {{{below}}};")

    def tex(self): return "\\begin{tikzpicture}\n" + "\n".join(self.lines) + "\n\\end{tikzpicture}"


def stats(G, lab, truth, q=True):
    s = f"{k(lab)} comm."
    if q: s += f", $Q{{=}}{Q(G, lab):.3f}$"
    return s + f", NMI {NMI(G, lab, truth):.3f}"


def figure(body, caption, label, star=True, pos="H"):
    """Full-width figures float to the top of a page ([!t]), right after the paragraph that introduces them."""
    if star:
        return f"\\begin{{figure*}}[{'b' if pos == '!b' else 't'}]\n\\centering\n{body}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure*}}\n"
    return f"\\begin{{figure}}[{pos}]\n\\centering\n{body}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n"


NUMBERS = {}   # every number quoted in the captions/text, also written to figures/numbers.json


# Full-width figures in the style of Figs. 4-6 of the base paper: graph panels joined by block arrows,
# three per row, rows read in a snake (left-to-right, down, right-to-left, down, ...). No text boxes:
# every explanation is in the paragraph that refers to the figure. All panels of one graph share one layout.
FW = 18.1; AG = 1.3; PWD = (FW - 2 * AG) / 3; PH = 4.0


def colx(c):
    return c * (PWD + AG)


def snake(F, G, pos, col, items, pitch=None, stat_lines=1):
    """items: dicts with lab, title, stat (optional), arrow (label on the arrow INTO this panel), kw.
    Returns the total height used."""
    pitch = pitch or PH + 0.95 + 0.33 * stat_lines
    n = len(items); rows = (n + 2) // 3
    for idx, it in enumerate(items):
        r_, p = divmod(idx, 3)
        c = p if r_ % 2 == 0 else 2 - p
        y = (rows - 1 - r_) * pitch
        F.panel(G, pos, it["lab"], col, colx(c), y, PWD, PH, title=it.get("title"), stat=it.get("stat"), **it.get("kw", {}))
        if idx == 0: continue
        pr, pp = divmod(idx - 1, 3); pc = pp if pr % 2 == 0 else 2 - pp
        if pr == r_:   # same row: horizontal arrow between the two panels
            xm = (colx(min(c, pc)) + PWD + colx(max(c, pc))) / 2
            F.arrow(xm, y + PH / 2, rot=0 if c > pc else 180, label=it.get("arrow"), length=9)
        else:          # next row: down arrow beside the panel column, clear of the statistics line
            xa = colx(c) + (PWD - 0.3 if c == 2 else 0.3)
            ya = y + PH + (pitch - PH) / 2 - 0.05
            F.add(f"\\node[barrow, rotate=-90, minimum height={pitch - PH - 0.25:.2f}cm] at ({xa:.3f},{ya:.3f}) {{}};")
            if it.get("arrow"):
                F.add(f"\\node[note, anchor={'east' if c == 2 else 'west'}] at ({xa + (-0.35 if c == 2 else 0.35):.3f},{ya:.3f}) {{{it['arrow']}}};")
    return rows * pitch


def row(F, G, pos, col, items, y=0.0, centre=False, ph=None):
    """One row of panels with arrows; items as in snake(). arrow_dir 'l' draws the arrow pointing left
    (used for "input in the middle" layouts)."""
    ph = ph or PH; ws = [it.get("w", PWD) for it in items]
    x = (FW - (sum(ws) + (len(items) - 1) * AG)) / 2 if centre else 0.0
    for idx, it in enumerate(items):
        if idx:
            if it.get("arrow_dir", "r") != "none":
                F.arrow(x - AG / 2, y + ph / 2, rot=180 if it.get("arrow_dir") == "l" else 0, label=it.get("arrow"), length=9)
        F.panel(G, it.get("pos", pos), it["lab"], col, x, y, ws[idx], ph, title=it.get("title"), stat=it.get("stat"), **it.get("kw", {}))
        x += ws[idx] + AG


EX_SPEC = ((7, 5, 4), 0.6, 0.08, 21)   # the example network of the three stage figures
FR = "goodc, line width=1.1pt"


def _example():
    G, truth = planted(*EX_SPEC); r = pipeline(G)
    pos, _ = layout(G, truth, seed=3)
    col = colours(r["M6"], truth, [r["L1"], r["g1"], r["M5"], r["base"]])
    return G, truth, r, pos, col


# ================================================================== Stage 1 of SN-LPA (unchanged pre-selection)
def fig_stage1():
    G, truth, r, pos, col = _example(); NI = r["NI"]
    steps = stage1_steps(G, NI)
    items = [dict(lab={}, title="(a)")]
    # first core: shown in three steps (selected, first two joins, last two joins with the rejections)
    s0 = steps[0]; c0 = s0["core"]; acc = [v for v, _, ok in s0["tried"] if ok]
    part = lambda mem: {v: c0 for v in mem}
    items.append(dict(lab=part([c0]), title="(b)", arrow="core " + str(c0), kw=dict(cores=(c0,))))
    items.append(dict(lab=part([c0] + acc[:2]), title="(c)", arrow=", ".join(map(str, acc[:2])), kw=dict(cores=(c0,))))
    items.append(dict(lab=part([c0] + acc), title="(d)", arrow=", ".join(map(str, acc[2:])), kw=dict(cores=(c0,))))
    for n, st in enumerate(steps[1:]):
        items.append(dict(lab=st["label"], title=f"({chr(101 + n)})", arrow=f"core {st['core']}", kw=dict(cores=(st["core"],))))
    assert len(items) == 9
    F = Fig(); snake(F, G, pos, col, items, pitch=PH + 1.25, stat_lines=0)
    NUMBERS["stage1"] = dict(spec=EX_SPEC, order=sorted(G, key=lambda v: -NI[v]),
                             steps=[dict(core=st["core"], tried=[(v, round(c, 3), ok) for v, c, ok in st["tried"]]) for st in steps])
    cap = ("Community pre-selection (Stage~1) of SN-LPA on an example network with three planted communities (nodes 1--7, 8--12 and 13--16). The black ring marks the current core; "
           "white nodes are not yet labelled.")
    return figure(F.tex(), cap, "fig:stage1")


# ================================================================== Stage 2 of SN-LPA (modified evolution)
def fig_stage2():
    G, truth, r, pos, col = _example(); NI = r["NI"]
    b = r["base"]; assert r["g0"] == r["L1"] and r["refined"]
    mv = moves_trace(G, NI, r["M5"]); assert len(mv) == 2
    mid = dict(r["M5"]); mid[mv[0][1]] = mv[0][3]
    items = [
        dict(lab=r["L1"], title="(a) Stage 1 result", stat=stats(G, r["L1"], truth)),
        dict(lab=b, title="(b) plain-sum run, $\\gamma{=}0$ (TPOC-LPA rule)", stat=stats(G, b, truth) + "\\\\\\textit{lower $Q$ than (a): rejected}", arrow="plain sum"),
        dict(lab=r["g1"], title="(c) size-normalised run, $\\gamma{=}1$", stat=stats(G, r["g1"], truth) + f"\\\\\\textit{{${k(r['g1'])}\\le1.5\\times{k(r['L1'])}$ communities: kept}}", arrow="normalised", kw=dict(frame=FR)),
        dict(lab=r["M5"], title="(d) after the merge", stat=stats(G, r["M5"], truth) + "\\\\\\textit{codelength $L$ falls: accepted}", arrow="merge"),
        dict(lab=mid, title=f"(e) Synwalk move of node {mv[0][1]}", stat=stats(G, mid, truth) + "\\\\\\textit{random walk stays inside}", arrow="move", kw=dict(rings=[mv[0][1]])),
        dict(lab=r["M6"], title=f"(f) Synwalk move of node {mv[1][1]}", stat=stats(G, r["M6"], truth) + "\\\\\\textit{$L_{\\mathrm{reg}}$ falls: moves kept}", arrow="move", kw=dict(rings=[mv[1][1]], frame=FR)),
    ]
    F = Fig(); snake(F, G, pos, col, items, stat_lines=2)
    NUMBERS["stage2"] = dict(k=[k(r[x]) for x in ("L1", "base", "g1", "M5", "M6")], nmi={x: round(NMI(G, r[x], truth), 3) for x in ("L1", "base", "g1", "M5", "M6")},
                             Q={x: round(Q(G, r[x]), 3) for x in ("L1", "base", "g1", "M5", "M6")},
                             L=[round(T.map_codelength(G, r["S2"]), 3), round(T.map_codelength(G, r["M5"]), 3)],
                             Lreg=[round(T.map_codelength(G, r["M5"], True), 3), round(T.map_codelength(G, r["M6"], True), 3)],
                             J=[round(T.synwalk_objective(G, r["M5"]), 4), round(T.synwalk_objective(G, r["M6"]), 4)],
                             moves=mv, mid_nmi=round(NMI(G, mid, truth), 3))
    cap = ("Community evolution (Stage~2) of SN-LPA on the example network of Fig.~\\ref{fig:stage1}: two evolution runs from the Stage~1 result, the choice between them by the granularity rule, the map-equation merge and the Synwalk boundary moves. The green frame marks the run that is kept and the final result.")
    return figure(F.tex(), cap, "fig:stage2")


# ================================================================== Stage 3 of SN-LPA (unchanged label filtering)
def fig_stage3():
    G, truth, r, pos, col = _example(); NI = r["NI"]
    det = stage3_detail(G, NI, r["M6"])
    keep2 = sorted(i for i, d in det.items() if len(d["keep"]) > 1)
    assert keep2 == sorted(v for v, s_ in r["ov"].items() if len(s_) > 1)
    notes = {i: "(" + ", ".join(str(c) for c in sorted(d["was"])) + ")" for i, d in det.items()}
    PH3 = 5.6; pos3 = ScaledLayout(pos, PWD - 0.5, PH - 0.5)
    F = Fig()
    row(F, G, pos3, col, ph=PH3, items=[
        dict(lab=r["M6"], title="(a) Stage 2 result"),
        dict(lab=r["M6"], title="(b) candidate labels of boundary nodes", arrow="WAS", kw=dict(notes=notes, rings=keep2)),
        dict(lab=r["M6"], title="(c) overlapping communities", arrow="$\\beta{=}0.8$", kw=dict(overlap=r["ov"])),
    ])
    NUMBERS["stage3"] = {str(i): dict(was={str(c): round(w, 3) for c, w in d["was"].items()}, keep=d["keep"]) for i, d in det.items()}
    cap = ("Community label filtering (Stage~3) of SN-LPA on the example network of Fig.~\\ref{fig:stage1}. "
           "In (b) each boundary node lists the communities among its neighbours (by core node); in (c) two-coloured nodes belong to two communities.")
    return figure(F.tex(), cap, "fig:stage3", pos="!b")   # bottom of the page that has Fig. 3 at the top


# ================================================================== motif similarity vs Jaccard
def fig_m1():
    spec = ((8, 4), 0.5, 0.04, 10)
    G, truth = planted(*spec); NI = T.node_importance(G); L1, _ = T.community_preselection(G, NI, 1.0)
    sj, lj = stage2_steps(G, NI, L1, "jaccard"); st, lt = stage2_steps(G, NI, L1, "tri_overlap")
    d = next((a, b) for a, b in zip(sj, st) if a["new"] != b["new"])
    i = d[0]["node"]; assert i == 6 and NMI(G, lt, truth) == 1.0
    # hand-placed so that node 6 sits in the middle and its four labelled edges have room
    fx = {6: (0, 0), 8: (-2.6, 0.0), 3: (3.0, 0.2), 4: (1.7, 2.0), 7: (1.7, -2.0), 2: (-1.2, 1.6), 1: (-2.6, 2.4), 5: (-0.6, 3.0),
          11: (-3.8, -0.8), 10: (-4.6, 0.2), 9: (-4.2, -2.0), 12: (-3.0, -2.5)}
    pos, _ = layout(G, truth, fixed=fx)
    col = colours(lt, truth, [L1, lj])
    el = {(i, j): f"{T.jaccard(G, i, j):.2f}\\,|\\,{T.tri_overlap(G, i, j):.2f}" for j in G.neighbors(i)}
    F = Fig()
    ws, wa = 4.45, FW - 2 * 4.45 - 2 * AG
    row(F, G, pos, col, ph=5.0, items=[
        dict(lab=lj, w=ws, title="(b) Stage 2 with Jaccard (TPOC-LPA)", stat=stats(G, lj, truth, q=False), kw=dict(rings=[i])),
        dict(lab=L1, w=wa, title="(a) Stage 1 result", stat=f"scores on the edges of node {i}: Jaccard\\,|\\,motif", arrow="Jaccard", arrow_dir="l", kw=dict(hl_edges=[(i, j) for j in G.neighbors(i)], rings=[i], elabels=el)),
        dict(lab=lt, w=ws, title="(c) Stage 2 with motif similarity", stat=stats(G, lt, truth, q=False), arrow="motif", kw=dict(rings=[i], frame=FR)),
    ])
    ca = lt[i]; cb = L1[i]
    sa = (d[0]["scores"][ca], d[1]["scores"][ca]); sb = (d[0]["scores"][cb], d[1]["scores"][cb])
    NUMBERS["m1"] = dict(spec=spec, node=i, scores=dict(jac=[round(sa[0], 3), round(sb[0], 3)], tri=[round(sa[1], 3), round(sb[1], 3)]),
                         nmi_jac=round(NMI(G, lj, truth), 3), nmi_tri=round(NMI(G, lt, truth), 3),
                         sims={j: (round(T.jaccard(G, i, j), 3), round(T.tri_overlap(G, i, j), 3)) for j in G.neighbors(i)})
    cap = "Jaccard versus motif similarity, both run from the Stage~1 partition (a) of a planted graph (nodes 1--8 and 9--12)."
    return figure(F.tex(), cap, "fig:m1")


# ================================================================== best-iterate guard
def fig_m2():
    spec = ((8, 4), 0.6, 0.12, 7)
    G, truth = planted(*spec); NI = T.node_importance(G); L1, _ = T.community_preselection(G, NI, 1.0)
    R = rounds(G, NI, L1, "jaccard"); assert len(R) == 3
    qs = [Q(G, x) for x in R]; best = max(range(len(R)), key=lambda t: qs[t]); assert best == 1
    g, _ = T.community_evolution(G, NI, dict(L1), guard=True); assert g == R[1]
    pos, _ = layout(G, truth, seed=2)
    col = colours(R[1], truth, R)
    F = Fig()
    row(F, G, pos, col, [dict(lab=lab, title=["(a) round 0 (Stage 1)", "(b) round 1: highest $Q$", "(c) round 2: TPOC-LPA output"][t],
                              stat=f"{k(lab)} comm., $Q{{=}}{qs[t]:.3f}$, NMI {NMI(G, lab, truth):.3f}", arrow=None if t == 0 else f"round {t}",
                              kw=dict(frame=FR) if t == best else {}) for t, lab in enumerate(R)])
    xb = PWD + AG + PWD / 2
    F.add(f"\\node[barrow, rotate=-90, minimum height=0.85cm] at ({xb:.3f},-1.0) {{}};")
    F.add(f"\\node[note, anchor=west, font=\\scriptsize\\itshape] at ({xb + 0.4:.3f},-1.0) {{highest $Q$: passed to Stage~3}};")
    E = [row_ for row_ in json.load(open(os.path.join(OUT, "email_rounds.json")))["jaccard"] if row_[0] <= 2]
    NUMBERS["m2"] = dict(spec=spec, Q=[round(q, 3) for q in qs], k=[k(x) for x in R], nmi=[round(NMI(G, x, truth), 3) for x in R], email=E)
    cap = "Best-iterate guard: Stage~2 rounds of TPOC-LPA on a planted graph (nodes 1--8 and 9--12). The guard passes the round with the highest $Q$ (green frame) to Stage~3."
    return figure(F.tex(), cap, "fig:m2", pos="!b")   # bottom: shares the page with the figure above


def fig_m2chart():
    """email-Eu (real data): communities, Q and NMI after each round of TPOC-LPA's Stage 2 (column figure)."""
    E = [row_ for row_ in json.load(open(os.path.join(OUT, "email_rounds.json")))["jaccard"] if row_[0] <= 2]
    pts_q = " ".join(f"({x[0]},{x[2]})" for x in E); pts_n = " ".join(f"({x[0]},{x[3]})" for x in E)
    lbl = "\n".join(f"\\node[font=\\scriptsize, anchor=south west, inner sep=1.5pt] at (axis cs:{x[0]},{max(x[2], x[3])}) {{{x[1]} comm.}};" for x in E)
    body = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[width=7.4cm, height=4.2cm, xmin=-0.2, xmax=2.4, ymin=0, ymax=0.7, xtick={{0,1,2}}, xticklabels={{0 (Stage 1),1,2}}, ytick={{0,0.2,0.4,0.6}},
  tick label style={{font=\\scriptsize}}, label style={{font=\\scriptsize}}, xlabel={{Stage-2 round}},
  axis line style={{black!50}}, tick style={{black!50}}, ymajorgrids, grid style={{black!10}},
  legend style={{font=\\scriptsize, draw=none, fill=none, at={{(0.98,0.95)}}, anchor=north east}}, legend cell align=left, clip=false]
\\addplot[pc0, line width=1pt, mark=*, mark size=1.6pt] coordinates {{{pts_n}}}; \\addlegendentry{{NMI}}
\\addplot[pc1, line width=1pt, mark=square*, mark size=1.5pt] coordinates {{{pts_q}}}; \\addlegendentry{{modularity $Q$}}
{lbl}
\\end{{axis}}
\\end{{tikzpicture}}"""
    return figure(body, "TPOC-LPA's Stage~2 on email-Eu (986 nodes): NMI and $Q$ after each round, with the number of communities.", "fig:m2chart", star=False, pos="!hb")


# ================================================================== size-normalised gain
def fig_m3():
    spec = ((8, 4), 0.8, 0.16, 7)
    G, truth = planted(*spec); NI = T.node_importance(G); L1, _ = T.community_preselection(G, NI, 1.0)
    assert NMI(G, L1, truth) == 1.0
    R0 = rounds(G, NI, L1, "jaccard"); assert len(R0) == 3 and k(R0[-1]) == 1
    s0, _ = stage2_steps(G, NI, L1, "jaccard", 0.0); s1, l1 = stage2_steps(G, NI, L1, "jaccard", 1.0)
    assert l1 == L1
    mv = [s_ for s_ in s0 if s_["new"] != s_["old"]]; first = mv[0]; i = first["node"]
    g1 = next(s_ for s_ in s1 if s_["node"] == i and s_["round"] == 1)
    big, small = first["new"], first["old"]
    later = sorted(s_["node"] for s_ in mv[1:])
    pos, _ = layout(G, truth, seed=4); pos = ScaledLayout(pos, PWD - 0.5, PH - 0.5)
    col = colours(L1, truth, R0)
    ph8 = 3.2; F = Fig(); pitch = ph8 + 1.6
    F.add(f"\\node[font=\\footnotesize\\bfseries, anchor=west] at (0,{pitch + ph8 + 0.8:.3f}) {{TPOC-LPA: plain sum ($\\gamma{{=}}0$)}};")
    row(F, G, pos, col, [
        dict(lab=R0[0], title="(a) Stage 1 (= ground truth)", stat=stats(G, R0[0], truth), kw=dict(rings=[i])),
        dict(lab=R0[1], title="(b) after round 1", stat=stats(G, R0[1], truth), arrow=f"node {i}", kw=dict(rings=[i])),
        dict(lab=R0[2], title="(c) after round 2", stat=stats(G, R0[2], truth), arrow=", ".join(map(str, later)), kw=dict(rings=later)),
    ], y=pitch, ph=ph8)
    F.add(f"\\node[font=\\footnotesize\\bfseries, anchor=west] at ({(FW - (2 * PWD + AG)) / 2:.3f},{ph8 + 0.8:.3f}) {{SN-LPA: size-normalised gain ($\\gamma{{=}}1$)}};")
    row(F, G, pos, col, [
        dict(lab=L1, title="(d) Stage 1 (= ground truth)", stat=stats(G, L1, truth), kw=dict(rings=[i])),
        dict(lab=l1, title="(e) after round 1: no node moves", stat=stats(G, l1, truth), arrow="round 1", kw=dict(rings=[i], frame=FR)),
    ], y=0, centre=True, ph=ph8)
    sc0 = first["scores"]
    NUMBERS["m3"] = dict(spec=spec, node=i, sum=[round(sc0[big], 3), round(sc0[small], 3)],
                         gain_big=[round(x, 3) for x in g1["scores"][big]], gain_small=[round(x, 3) for x in g1["scores"][small]],
                         deg_hub=G.degree(5), later=later, Q=[round(Q(G, x), 3) for x in R0])
    cap = "Plain-sum Stage~2 of TPOC-LPA (top) and the size-normalised gain of SN-LPA (bottom), from the same Stage~1 partition of a planted graph (nodes 1--8 and 9--12)."
    return figure(F.tex(), cap, "fig:m3")


# ================================================================== granularity rule
def fig_m4():
    K = nx.karate_club_graph(); kt = {v: K.nodes[v]["club"] for v in K}
    KG = nx.Graph(); KG.add_nodes_from(K.nodes()); KG.add_edges_from(K.edges())   # structure only (the code ignores weights)
    rk = pipeline(KG)
    assert not rk["refined"] and k(rk["g0"]) == 2 and k(rk["g1"]) == 4
    spec = ((6, 5, 5, 4), 0.5, 0.04, 14)
    G, truth = planted(*spec); r = pipeline(G); assert r["refined"] and NMI(G, r["g1"], truth) == 1.0
    kpos, _ = layout(KG, small=True); kpos = ScaledLayout(kpos, PWD - 0.44, PH - 0.44)
    kcol = colours(rk["g0"], kt, [rk["g1"], rk["L1"]])
    pos, _ = layout(G, truth, seed=5); pos = ScaledLayout(pos, PWD - 0.5, PH - 0.5)
    col = colours(r["g1"], truth, [r["g0"], r["L1"]])
    ph9 = 3.1; F = Fig(); pitch = ph9 + 1.2
    rk_ = k(rk["g1"]) / k(rk["g0"]); rr_ = k(r["g1"]) / k(r["g0"])
    row(F, KG, kpos, kcol, [
        dict(lab=rk["g0"], title="(b) Karate, $\\gamma{=}0$: kept", stat=f"{k(rk['g0'])} comm., NMI {NMI(KG, rk['g0'], kt):.3f}", kw=dict(numbers=False, small=True, frame=FR)),
        dict(lab=rk["L1"], title="(a) Karate, Stage 1 result", stat=f"{k(rk['L1'])} fragments", arrow="$\\gamma{=}0$", arrow_dir="l", kw=dict(numbers=False, small=True)),
        dict(lab=rk["g1"], title=f"(c) Karate, $\\gamma{{=}}1$: ${k(rk['g1'])}/{k(rk['g0'])}={rk_:.1f}>1.5$", stat=f"{k(rk['g1'])} comm., NMI {NMI(KG, rk['g1'], kt):.3f}", arrow="$\\gamma{=}1$", kw=dict(numbers=False, small=True)),
    ], y=pitch, ph=ph9)
    row(F, G, pos, col, [
        dict(lab=r["g0"], title="(e) planted graph, $\\gamma{=}0$", stat=f"{k(r['g0'])} comm., NMI {NMI(G, r['g0'], truth):.3f}"),
        dict(lab=r["L1"], title="(d) planted graph, Stage 1 result", stat=f"{k(r['L1'])} fragments", arrow="$\\gamma{=}0$", arrow_dir="l"),
        dict(lab=r["g1"], title=f"(f) $\\gamma{{=}}1$: ${k(r['g1'])}/{k(r['g0'])}={rr_:.2f}\\le1.5$, kept", stat=f"{k(r['g1'])} comm., NMI {NMI(G, r['g1'], truth):.3f}", arrow="$\\gamma{=}1$", kw=dict(frame=FR)),
    ], y=0, ph=ph9)
    NUMBERS["m4"] = dict(karate=dict(kL1=k(rk["L1"]), k0=k(rk["g0"]), k1=k(rk["g1"]), n0=round(NMI(KG, rk["g0"], kt), 4), n1=round(NMI(KG, rk["g1"], kt), 4)),
                         example=dict(spec=spec, kL1=k(r["L1"]), k0=k(r["g0"]), k1=k(r["g1"]), n0=round(NMI(G, r["g0"], truth), 3), n1=round(NMI(G, r["g1"], truth), 3)))
    cap = "Granularity rule on Karate (top) and on a planted graph with four communities (bottom): both Stage~2 runs start from the Stage~1 result; the green frame marks the run the rule keeps."
    return figure(F.tex(), cap, "fig:m4", pos="!b")   # bottom: shares the page with the figure above


# ================================================================== Synwalk boundary moves
def fig_m6():
    spec = ((8, 4), 0.6, 0.08, 14)
    G, truth = planted(*spec); r = pipeline(G)
    assert r["moves"] == 1 and NMI(G, r["M6"], truth) == 1.0
    i = next(v for v in G if r["M5"][v] != r["M6"][v])
    b = r["M6"][i]
    pos, _ = layout(G, truth, seed=6)
    col = colours(r["M6"], truth, [r["M5"], r["L1"]])
    J0, J1 = T.synwalk_objective(G, r["M5"]), T.synwalk_objective(G, r["M6"])
    R0, R1 = T.map_codelength(G, r["M5"], True), T.map_codelength(G, r["M6"], True)
    kin = sum(1 for j in G.neighbors(i) if r["M5"][j] == b)
    F = Fig()
    row(F, G, ScaledLayout(pos, PWD - 0.5, PH - 0.5), col, ph=3.6, items=[
        dict(lab=r["L1"], title="(a) Stage 1 result", stat=stats(G, r["L1"], truth)),
        dict(lab=r["M5"], title="(b) Stage 2, before the moves", stat=stats(G, r["M5"], truth), arrow="Stage 2", kw=dict(rings=[i])),
        dict(lab=r["M6"], title=f"(c) Synwalk move of node {i}", stat=stats(G, r["M6"], truth), arrow=f"move {i}", kw=dict(rings=[i], frame=FR)),
    ])
    NUMBERS["m6"] = dict(spec=spec, node=i, links_to_new=kin, deg=G.degree(i), J=[round(J0, 4), round(J1, 4)], Lreg=[round(R0, 3), round(R1, 3)],
                         Q=[round(Q(G, r["M5"]), 3), round(Q(G, r["M6"]), 3)], nmi_L1=round(NMI(G, r["L1"], truth), 3))
    cap = "A Synwalk boundary move on a planted graph (nodes 1--8 and 9--12)."
    return figure(F.tex(), cap, "fig:m6")   # bottom of the page, so text separates it from Fig. 9


if __name__ == "__main__":
    figs = dict(stage1=fig_stage1(), stage2=fig_stage2(), stage3=fig_stage3(), m1=fig_m1(), m2=fig_m2(), m3=fig_m3(), m4=fig_m4(), m6=fig_m6())
    for name, tex in figs.items():
        open(os.path.join(OUT, f"fig_{name}.tex"), "w").write(tex)
    open(os.path.join(OUT, "fig_preamble.tex"), "w").write(PREAMBLE + "\n")
    json.dump(NUMBERS, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=str)
    print("wrote", ", ".join(f"fig_{n}.tex" for n in figs))
