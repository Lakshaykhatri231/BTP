"""
Phase 3 ablation: which structural mechanism actually closes TPOC-LPA's gaps?

Usage:
    python3 phase3_ablation.py real        # 8 real ground-truth networks
    python3 phase3_ablation.py lfr         # LFR sweep, multi-seed, mean+/-std
    python3 phase3_ablation.py determinism # 15-run stability check
    python3 phase3_ablation.py large       # SNAP amazon/dblp/youtube (slow)

Protocol (from the approved plan):
  * one factor at a time, so each mechanism's contribution is attributable;
  * EVERY synthetic number is multi-seed with std -- a single-seed "+0.015"
    for damped aggregation evaporated across 3 seeds during planning, so
    single-seed results are not claims;
  * NMI and Q reported side by side: a config that raises Q while lowering
    NMI is a regression, not an improvement (the Karate split trap);
  * a regression gate rejects any config that degrades a benchmark which
    currently matches the paper.
"""
import statistics as st
import sys
import time
from collections import defaultdict

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community.quality import modularity

import datasets
from tpoc_lpa import tpoc_lpa

# (label, kwargs) -- "base" must stay first so deltas are measured against it
CONFIGS = [
    ("base (jaccard/sum)", dict(sim="jaccard", agg="sum")),
    ("tri_overlap/sum", dict(sim="tri_overlap", agg="sum")),
    ("guard", dict(sim="jaccard", agg="sum", guard=True)),
    ("tri_overlap+guard", dict(sim="tri_overlap", agg="sum", guard=True)),
]

# Benchmarks that currently match the paper -- the regression gate protects these.
PROTECTED = {"dolphin": ("nmi", 0.9053), "football": ("q", 0.6032)}


def evaluate(G, truth, **kw):
    nodes = list(G.nodes())
    label, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8, **kw)
    groups = defaultdict(set)
    for n in nodes:
        groups[label[n]].add(n)
    y_true = [truth[n] for n in nodes]
    y_pred = [label[n] for n in nodes]
    return {
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "q": modularity(G, list(groups.values())),
        "ncomm": len(groups),
        "ntrue": len(set(y_true)),
    }


def run_real(registry=None):
    registry = registry or datasets.REAL
    print(f"{'network':<11}{'n':>7}{'true':>6}  " +
          "".join(f"{lbl:>26}" for lbl, _ in CONFIGS))
    regressions = []
    for name, loader in registry.items():
        G, truth = loader()
        cells, base = [], None
        for lbl, kw in CONFIGS:
            t0 = time.time()
            r = evaluate(G, truth, **kw)
            if base is None:
                base = r
                cells.append(f"NMI {r['nmi']:.4f} Q {r['q']:.4f} c{r['ncomm']}")
            else:
                cells.append(f"NMI {r['nmi']:.4f}({r['nmi']-base['nmi']:+.3f}) c{r['ncomm']}")
                if name in PROTECTED:
                    metric, target = PROTECTED[name]
                    if r[metric] < target - 1e-4:
                        regressions.append(f"{name}/{lbl}: {metric} {r[metric]:.4f} < {target}")
                # the Karate trap: Q up but NMI down
                if r["q"] > base["q"] + 1e-6 and r["nmi"] < base["nmi"] - 1e-6:
                    regressions.append(
                        f"{name}/{lbl}: Q rose {base['q']:.4f}->{r['q']:.4f} "
                        f"but NMI FELL {base['nmi']:.4f}->{r['nmi']:.4f}")
        print(f"{name:<11}{G.number_of_nodes():>7}{base['ntrue']:>6}  " +
              "".join(f"{c:>26}" for c in cells))
    print()
    if regressions:
        print("REGRESSION GATE -- the following are rejected:")
        for r in regressions:
            print("   ", r)
    else:
        print("REGRESSION GATE: clean (no protected benchmark degraded, no Q-up/NMI-down).")


def run_lfr(sizes=(1000,), mus=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), seeds=(1, 2, 3, 4, 5)):
    from lfr_sweep import generate_lfr
    print(f"LFR: sizes={list(sizes)} mu={list(mus)} seeds={len(seeds)}  (mean +/- std over seeds)")
    for n in sizes:
        print(f"\n  n={n}")
        print(f"    {'mu':<6}" + "".join(f"{lbl:>22}" for lbl, _ in CONFIGS))
        for mu in mus:
            graphs = []
            for s in seeds:
                try:
                    graphs.append(generate_lfr(n=n, mu=mu, seed=s))
                except Exception as e:
                    print(f"      mu={mu} seed={s} generation failed: {type(e).__name__}")
            if not graphs:
                continue
            row = f"    {mu:<6}"
            for lbl, kw in CONFIGS:
                vals = [evaluate(G, t, **kw)["nmi"] for G, t in graphs]
                row += f"{st.mean(vals):>14.4f}+/-{st.pstdev(vals):.3f}"
            print(row)


def run_determinism(registry=None, runs=15):
    import random
    registry = registry or datasets.SMALL
    print(f"Determinism: {runs} runs each, deterministic ties (rng=None)")
    for name, loader in registry.items():
        G, truth = loader()
        for lbl, kw in CONFIGS[:2]:
            vals = [evaluate(G, truth, **kw)["nmi"] for _ in range(runs)]
            sd = st.pstdev(vals)
            print(f"  {name:<10} {lbl:<20} NMI {st.mean(vals):.4f} +/- {sd:.6f} "
                  f"{'OK' if sd == 0 else 'NONDETERMINISTIC'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "real"
    if mode == "real":
        run_real()
    elif mode == "lfr":
        run_lfr()
    elif mode == "determinism":
        run_determinism()
    elif mode == "large":
        run_real(datasets.LARGE)
    else:
        print(__doc__)
