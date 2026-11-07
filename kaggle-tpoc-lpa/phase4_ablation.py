"""
Phase 4: bounding TPOC-LPA's Stage 2 and choosing the regime without labels.

Usage:
    python3 phase4_ablation.py records dev        # compute + cache per-graph records (dev set)
    python3 phase4_ablation.py records heldout    # ... held-out set (fresh seeds, larger n)
    python3 phase4_ablation.py real               # per-network table, all regimes
    python3 phase4_ablation.py selectors          # W1: label-free regime selectors, LODO
    python3 phase4_ablation.py merge              # W2: gated merge pass, epsilon sweep
    python3 phase4_ablation.py kappa              # W4a: adaptive triangle saturation
    python3 phase4_ablation.py heldout            # apply dev-chosen rules to the held-out set

Design (from the approved plan):
  * every graph is run ONCE per regime and the record is cached on disk, so
    selectors / thresholds are post-hoc functions of the cache and can be
    swept, leave-one-dataset-out (LODO) cross-validated, and re-run instantly;
  * a selector may only use label-free quantities (modularity, stability under
    edge perturbation, whether the guard rejected Stage 2) -- never NMI;
  * thresholds are chosen by LODO on the DEV set and then frozen for the
    held-out set (fresh LFR seeds, n=5000/10000);
  * synthetic results are multi-seed with paired per-seed deltas.
"""
import json
import os
import pickle
import random
import statistics as st
import sys
import time
from collections import defaultdict
from itertools import combinations

import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import normalized_mutual_info_score as nmi_score

import datasets
from tpoc_lpa import tpoc_lpa

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(datasets.D, "phase4_cache")
os.makedirs(CACHE, exist_ok=True)

REGIMES = {
    "base":       dict(),                                                   # the paper
    "phase3":     dict(sim="tri_overlap", guard=True),                      # Phase 3 default
    "gamma1":     dict(sim="tri_overlap", gamma=1.0),                       # null-model (LPAm-like)
    "fallback":   dict(sim="tri_overlap", guard=True, fallback="collapse"), # S1
    "p3_merge":   dict(sim="tri_overlap", guard=True, merge=True, merge_min_gain=0.0),
    "jac_merge":  dict(sim="jaccard", guard=True, merge=True, merge_min_gain=0.0),
    "g1_merge":   dict(sim="tri_overlap", gamma=1.0, merge=True, merge_min_gain=0.0),
    "fb_merge":   dict(sim="tri_overlap", guard=True, fallback="collapse", merge=True, merge_min_gain=0.0),
    # post-hoc exploration (added after the unseen real graphs showed motif hurting on CS/Physics)
    "jac_guard":  dict(sim="jaccard", guard=True),
    "jac_gamma1": dict(sim="jaccard", gamma=1.0),
    "kappa_auto": dict(sim="tri_overlap", guard=True, kappa="auto"),
    "kappa2":     dict(sim="tri_overlap", guard=True, kappa=2.0),
}

# LODO groups: each real network alone, each LFR mu (all seeds) as one group
DEV_LFR = dict(n=1000, mus=(0.3, 0.5, 0.6, 0.7), seeds=(1, 2, 3, 4, 5))
HELD_LFR = [dict(n=1000, mus=(0.5, 0.6, 0.7), seeds=tuple(range(11, 21))),
            dict(n=5000, mus=(0.5, 0.6, 0.7), seeds=(1, 2, 3)),
            dict(n=10000, mus=(0.5, 0.6, 0.7), seeds=(1, 2, 3))]
S6_RHO, S6_DQ = 1.5, 0.10
RHOS = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0)
# Held-out graphs in the regime where gamma=1 is expected to FAIL (coarse ground truth:
# few, large communities -- the Karate/Polbooks pattern): never used to pick any threshold.
HELD_COARSE = [
    dict(tag="coarse1000", n=1000, mus=(0.2, 0.4, 0.6), seeds=(1, 2, 3, 4),
         kw=dict(min_community=100, max_community=250)),
    dict(tag="small150", n=150, mus=(0.2, 0.4), seeds=(1, 2, 3, 4, 5, 6),
         kw=dict(average_degree=8, max_degree=20, min_community=30, max_community=80)),
]
# amazon_photo was inspected while the selector was being designed, so it is DEV data (below);
# these are untouched:
HELD_REAL = ("amazon_computers", "coauthor_cs", "coauthor_physics")
FRESH_REAL = ("cora_ml", "pubmed", "dblp_authors")
DELTAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 9.0)   # 9.0 == never pick gamma1


# ---------------------------------------------------------------- graphs ----
def get_lfr(n, mu, seed, **kw):
    tag = "".join(f"_{k}{v}" for k, v in sorted(kw.items()))
    p = os.path.join(CACHE, f"lfr_n{n}_mu{mu}_s{seed}{tag}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    from lfr_sweep import generate_lfr
    G, t = generate_lfr(n=n, mu=mu, seed=seed, **kw)
    pickle.dump((G, t), open(p, "wb"))
    return G, t


def graph_specs(which):
    """[(gid, group, loader)] -- group is the LODO unit."""
    specs = []
    if which == "dev":
        for name, ld in datasets.REAL.items():
            specs.append((name, name, ld))
        specs.append(("amazon_photo", "amazon_photo", lambda: load_npz_graph("amazon_photo")))
        c = DEV_LFR
        for mu in c["mus"]:
            for s in c["seeds"]:
                specs.append((f"lfr{c['n']}_mu{mu}_s{s}", f"lfr{c['n']}_mu{mu}",
                              lambda n=c["n"], mu=mu, s=s: get_lfr(n, mu, s)))
    elif which == "fresh":
        # fetched AFTER the unseen-real-graph results exposed the guard/motif failure on Physics,
        # used only to validate, never to tune
        for name in FRESH_REAL:
            specs.append((name, name, lambda name=name: load_npz_graph(name)))
    else:
        for c in HELD_LFR:
            for mu in c["mus"]:
                for s in c["seeds"]:
                    specs.append((f"lfr{c['n']}_mu{mu}_s{s}", f"lfr{c['n']}_mu{mu}",
                                  lambda n=c["n"], mu=mu, s=s: get_lfr(n, mu, s)))
        for c in HELD_COARSE:
            for mu in c["mus"]:
                for s in c["seeds"]:
                    specs.append((f"{c['tag']}_mu{mu}_s{s}", f"{c['tag']}_mu{mu}",
                                  lambda n=c["n"], mu=mu, s=s, kw=c["kw"]: get_lfr(n, mu, s, **kw)))
        for name in HELD_REAL:
            specs.append((name, name, lambda name=name: load_npz_graph(name)))
    return specs


def load_npz_graph(name):
    """Shchur et al. GNN-benchmark graphs (class labels as ground truth), LCC-restricted."""
    import numpy as np
    from scipy.sparse import csr_matrix
    fn = {"pubmed": "pubmed", "cora_ml": "cora_ml", "dblp_authors": "dblp",
          "amazon_photo": "amazon_electronics_photo", "amazon_computers": "amazon_electronics_computers",
          "coauthor_cs": "ms_academic_cs",
          "coauthor_physics": "ms_academic_phy"}[name]
    z = np.load(os.path.join(datasets.D, "heldout_real", fn + ".npz"), allow_pickle=True)
    A = csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]), shape=tuple(z["adj_shape"]))
    G = nx.from_scipy_sparse_array(A)
    return datasets._lcc(G, {i: str(int(l)) for i, l in enumerate(z["labels"])})


# --------------------------------------------------------------- running ----
def _labels(G, lab):
    nodes = list(G.nodes())
    return nodes, [lab[n] for n in nodes]


def evaluate(G, truth, **kw):
    info = {}
    lab, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8, info=info, **kw)
    nodes, pred = _labels(G, lab)
    groups = defaultdict(set)
    for n in nodes:
        groups[lab[n]].add(n)
    return dict(nmi=nmi_score([truth[n] for n in nodes], pred),
                q=modularity(G, list(groups.values())), nc=len(groups),
                fell_back=bool(info.get("fell_back")), merge_gain=info.get("merge_gain", 0.0),
                merges=info.get("merges", 0))


def stability(G, kw, k=5, frac=0.10):
    """Mean pairwise NMI of k runs on graphs with a random 10% of edges removed.
    Label-free: a regime whose partition survives perturbation is 'stable'."""
    edges = list(G.edges())
    runs = []
    for r in range(k):
        rng = random.Random(1000 + r)
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(e for e in edges if rng.random() > frac)
        lab, _, _, _ = tpoc_lpa(H, **kw)
        runs.append([lab[n] for n in G.nodes()])
    return st.mean(nmi_score(a, b) for a, b in combinations(runs, 2))


def record_graph(gid, G, truth, regimes=None, with_stability=True):
    p = os.path.join(CACHE, f"rec_{gid}.json")
    rec = json.load(open(p)) if os.path.exists(p) else {}
    rec["n"] = G.number_of_nodes()
    rec["m"] = G.number_of_edges()
    dirty = False
    for name, kw in (regimes or REGIMES).items():
        if name not in rec:
            rec[name] = evaluate(G, truth, **kw)
            dirty = True
    if with_stability and "stab" not in rec and G.number_of_nodes() <= 3500:
        rec["stab"] = {r: stability(G, REGIMES[r]) for r in ("phase3", "gamma1")}
        dirty = True
    if dirty:
        json.dump(rec, open(p, "w"))
    return rec


def load_records(which):
    out = {}
    for gid, group, _ in graph_specs(which):
        p = os.path.join(CACHE, f"rec_{gid}.json")
        if os.path.exists(p):
            out[gid] = (group, json.load(open(p)))
    return out


def cmd_records(which, only=None):
    for gid, group, loader in graph_specs(which):
        if only == "lfr" and not gid.startswith("lfr"):
            continue
        if only == "coarse" and not (gid.startswith("coarse") or gid.startswith("small")):
            continue
        if only == "real" and gid not in HELD_REAL:
            continue
        if only and only.startswith("gid=") and gid != only[4:]:
            continue
        t0 = time.time()
        G, truth = loader()
        big = which in ("heldout", "fresh") and G.number_of_nodes() > 5000
        regs = {k: REGIMES[k] for k in ("base", "phase3", "gamma1", "fallback", "p3_merge", "g1_merge", "jac_guard", "jac_gamma1")} if big else None
        rec = record_graph(gid, G, truth, regimes=regs, with_stability=(which == "dev"))
        print(f"  {gid:<26} n={rec['n']:>6}  phase3 {rec['phase3']['nmi']:.4f}  "
              f"gamma1 {rec['gamma1']['nmi']:.4f}  fallback {rec['fallback']['nmi']:.4f}"
              f"{' [fell back]' if rec['fallback']['fell_back'] else ''}  ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------- selectors ----
def select(rec, kind, delta=None):
    """Return the regime name a label-free selector chooses for one graph.

    S5 (added AFTER inspecting the dev set -- validated only on held-out data):
    accept gamma=1 iff it does not fragment the Phase 3 partition, i.e.
    nc(gamma1) <= rho * nc(phase3); `delta` carries rho. Rationale: when the two
    regimes agree on granularity, gamma=1 only refines boundaries; when it
    changes the scale of the partition, modularity's preferred granularity has
    diverged from the published dynamics and we cannot tell which is right
    without labels, so the conservative published-style result is kept."""
    if kind == "S0":
        return "phase3"
    if kind == "S1":
        return "fallback"
    q0, q1 = rec["phase3"]["q"], rec["gamma1"]["q"]
    if kind == "S2":
        return "gamma1" if q1 - q0 > delta else "phase3"
    if kind == "S3":
        return "gamma1" if q1 > q0 + 1e-9 else "phase3"
    if kind == "S4":
        s = rec.get("stab")
        return "gamma1" if s and s["gamma1"] > s["phase3"] + 1e-9 else "phase3"
    if kind == "S5":
        return "gamma1" if rec["gamma1"]["nc"] <= delta * rec["phase3"]["nc"] else "phase3"
    if kind == "S5J":   # S5 with the paper's Jaccard similarity (post-hoc exploration)
        return "jac_gamma1" if rec["jac_gamma1"]["nc"] <= delta * rec["jac_guard"]["nc"] else "jac_guard"
    if kind == "S6":
        # S5 OR a large modularity gain. Declared (rho=1.5 from S5's LODO, dq=0.10 from S2's
        # whole-pool choice) BEFORE looking at the unseen real held-out graphs; the dq clause
        # rests on a single dev graph (Amazon Photo, dQ=+0.126), so it is a hypothesis under test.
        return ("gamma1" if rec["gamma1"]["nc"] <= S6_RHO * rec["phase3"]["nc"]
                or rec["gamma1"]["q"] - rec["phase3"]["q"] >= S6_DQ else "phase3")
    if kind == "ORACLE":  # uses labels -- an upper bound only, never a real selector
        return "gamma1" if rec["gamma1"]["nmi"] > rec["phase3"]["nmi"] else "phase3"
    raise ValueError(kind)


def group_means(records, kind, delta=None, regimes_override=None):
    g = defaultdict(list)
    for gid, (group, rec) in records.items():
        g[group].append(rec[select(rec, kind, delta)]["nmi"])
    return {k: st.mean(v) for k, v in g.items()}


def lodo_delta(records, held_group):
    """Pick S2's delta on every group EXCEPT `held_group` (most conservative on ties)."""
    best, best_score = None, -1
    for d in DELTAS:
        gm = group_means({k: v for k, v in records.items() if v[0] != held_group}, "S2", d)
        score = st.mean(gm.values())
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and d > best):
            best, best_score = d, score
    return best


def lodo_rho(records, held_group):
    """rho for S5 chosen on every group EXCEPT `held_group`; ties -> smallest rho."""
    best, best_score = None, -1
    for r in RHOS:
        gm = group_means({k: v for k, v in records.items() if v[0] != held_group}, "S5", r)
        score = st.mean(gm.values())
        if score > best_score + 1e-9:
            best, best_score = r, score
    return best


def cmd_selectors():
    recs = load_records("dev")
    groups = sorted({g for g, _ in recs.values()}, key=lambda g: (g.startswith("lfr"), g))
    cols = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "ORACLE"]
    table = {}
    for kind in cols:
        if kind == "S2":
            gm = {}
            for g in groups:
                d = lodo_delta(recs, g)
                gm[g] = group_means({k: v for k, v in recs.items() if v[0] == g}, "S2", d)[g]
            table[kind] = gm
        elif kind == "S5":
            gm = {}
            for g in groups:
                gm[g] = group_means({k: v for k, v in recs.items() if v[0] == g}, "S5", lodo_rho(recs, g))[g]
            table[kind] = gm
        else:
            table[kind] = group_means(recs, kind)
    print(f"{'group':<18}" + "".join(f"{c:>9}" for c in cols) + "   (NMI; S2 uses a LODO-chosen delta)")
    for g in groups:
        print(f"{g:<18}" + "".join(f"{table[c][g]:>9.4f}" for c in cols))
    print(f"{'MEAN over groups':<18}" + "".join(f"{st.mean(table[c].values()):>9.4f}" for c in cols))
    print("\nLODO-chosen S2 delta per held-out group:",
          {g: lodo_delta(recs, g) for g in groups})
    full = max(DELTAS, key=lambda d: (st.mean(group_means(recs, "S2", d).values()), d))
    print(f"S2 delta chosen on the WHOLE dev pool (frozen for held-out): {full}")
    print("LODO-chosen S5 rho per held-out group:", {g: lodo_rho(recs, g) for g in groups})
    rho_full = min(RHOS, key=lambda r: (-st.mean(group_means(recs, "S5", r).values()), r))
    print(f"S5 rho chosen on the WHOLE dev pool (frozen for held-out): {rho_full}")
    print("S5 mean NMI by rho:", {r: round(st.mean(group_means(recs, "S5", r).values()), 4) for r in RHOS})
    # regression gate against Phase 3 (S0): protected groups must not drop
    print("\nRegression gate vs S0 (Phase 3), protected: karate, dolphin, football, lfr1000_mu0.3")
    for kind in cols[1:7]:
        bad = [f"{g} {table['S0'][g]:.4f}->{table[kind][g]:.4f}" for g in
               ("karate", "dolphin", "football", "lfr1000_mu0.3")
               if table[kind][g] < table["S0"][g] - 1e-4]
        print(f"  {kind}: {'CLEAN' if not bad else 'REJECTED  ' + '; '.join(bad)}")
    # S4 diagnostics: what stability said
    print("\nStability (mean pairwise NMI under 10% edge removal), phase3 vs gamma1:")
    seen = set()
    for gid, (g, rec) in recs.items():
        if g in seen or "stab" not in rec or g.startswith("lfr"):
            continue
        seen.add(g)
        print(f"  {g:<10} phase3 {rec['stab']['phase3']:.3f}   gamma1 {rec['stab']['gamma1']:.3f}")


def cmd_real():
    recs = load_records("dev")
    cols = ["base", "phase3", "gamma1", "fallback", "p3_merge", "jac_merge", "kappa_auto", "kappa2"]
    print(f"{'network':<10}" + "".join(f"{c:>13}" for c in cols) + "   NMI (Q in second row)")
    for gid, (g, rec) in recs.items():
        if g.startswith("lfr"):
            continue
        print(f"{g:<10}" + "".join(f"{rec[c]['nmi']:>13.4f}" for c in cols))
        print(f"{'  Q':<10}" + "".join(f"{rec[c]['q']:>13.4f}" for c in cols))
    print()
    for mu_group in sorted({g for g, _ in recs.values() if g.startswith("lfr")}):
        rows = [r for gg, r in recs.values() if gg == mu_group]
        print(f"{mu_group:<18}" + "".join(
            f"{st.mean(r[c]['nmi'] for r in rows):>9.4f}±{st.pstdev(r[c]['nmi'] for r in rows):.3f}" for c in cols))


def cmd_merge():
    recs = load_records("dev")
    print("W2: merge pass. total dQ and NMI change if accepted, per graph (regime p3_merge = motif+guard+merge):")
    print(f"{'graph':<10}{'phase3':>9}{'merged':>9}{'dNMI':>8}{'total dQ':>10}{'merges':>8}")
    for gid, (g, rec) in recs.items():
        if g.startswith("lfr"):
            continue
        a, b = rec["phase3"], rec["p3_merge"]
        print(f"{g:<10}{a['nmi']:>9.4f}{b['nmi']:>9.4f}{b['nmi']-a['nmi']:>+8.3f}{b['merge_gain']:>10.4f}{b['merges']:>8}")
    print("\nepsilon sweep (mean NMI over the dev groups; merge accepted iff total dQ >= eps):")
    groups = sorted({g for g, _ in recs.values()}, key=lambda g: (g.startswith("lfr"), g))
    for eps in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 9.0):
        per = defaultdict(list)
        for gid, (g, rec) in recs.items():
            use = rec["p3_merge"] if rec["p3_merge"]["merge_gain"] >= eps else rec["phase3"]
            per[g].append(use["nmi"])
        gm = {g: st.mean(v) for g, v in per.items()}
        prot = ", ".join(f"{g} {gm[g]:.3f}" for g in ("football", "email-Eu", "cora", "citeseer", "karate"))
        print(f"  eps={eps:<6} mean {st.mean(gm.values()):.4f} | {prot}")


EPSILONS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 9.0)      # 9.0 == never merge


def final_choice(rec, eps):
    """S5 regime selection, then the gated merge pass on the SELECTED regime's partition."""
    reg = select(rec, "S5", S6_RHO)
    m = rec["g1_merge" if reg == "gamma1" else "p3_merge"]
    base = rec[reg]
    return (m if m["merge_gain"] >= eps and m["merges"] > 0 else base), reg, m


def cmd_final(which="dev", eps_frozen=None):
    recs = load_records(which)
    groups = sorted({g for g, _ in recs.values()}, key=lambda g: (g.startswith("lfr"), g))
    print(f"S5 selection + gated merge, {which} set: mean NMI per group by epsilon (merge accepted iff total dQ >= eps)")
    print(f"{'group':<18}{'S5 only':>9}" + "".join(f"{('eps=' + str(e)):>10}" for e in EPSILONS[:-1]))
    tab = {}
    for g in groups:
        rows = [r for gg, r in recs.values() if gg == g]
        s5 = st.mean(r[select(r, "S5", S6_RHO)]["nmi"] for r in rows)
        cells = [st.mean(final_choice(r, e)[0]["nmi"] for r in rows) for e in EPSILONS[:-1]]
        tab[g] = (s5, cells)
        print(f"{g:<18}{s5:>9.4f}" + "".join(f"{c:>10.4f}" for c in cells))
    print(f"{'MEAN':<18}{st.mean(v[0] for v in tab.values()):>9.4f}" +
          "".join(f"{st.mean(v[1][i] for v in tab.values()):>10.4f}" for i in range(len(EPSILONS) - 1)))


def cmd_kappa():
    recs = load_records("dev")
    print("W4a: triangle saturation kappa (motif+guard). NMI")
    cols = ["phase3", "kappa2", "kappa_auto", "base"]
    print(f"{'group':<18}" + "".join(f"{c:>12}" for c in cols))
    groups = sorted({g for g, _ in recs.values()}, key=lambda g: (g.startswith("lfr"), g))
    for g in groups:
        rows = [r for gg, r in recs.values() if gg == g]
        print(f"{g:<18}" + "".join(f"{st.mean(r[c]['nmi'] for r in rows):>12.4f}" for c in cols))


def cmd_heldout():
    recs = load_records("heldout")
    dev = load_records("dev")
    full = max(DELTAS, key=lambda d: (st.mean(group_means(dev, "S2", d).values()), d))
    print(f"held-out records: {len(recs)} graphs; S2 delta frozen from dev = {full}")
    groups = sorted({g for g, _ in recs.values()})
    rho_full = min(RHOS, key=lambda r: (-st.mean(group_means(dev, "S5", r).values()), r))
    print(f"S5 rho frozen from dev = {rho_full}")
    kinds = [("S0", None), ("S1", None), ("S2", full), ("S3", None), ("S5", rho_full), ("S6", None), ("ORACLE", None)]
    print(f"{'group':<18}{'seeds':>6}" + "".join(f"{k:>9}" for k, _ in kinds) + f"{'gamma1':>9}{'base':>9}")
    for g in groups:
        rows = [r for gg, r in recs.values() if gg == g]
        cells = [st.mean(r[select(r, k, d)]["nmi"] for r in rows) for k, d in kinds]
        print(f"{g:<18}{len(rows):>6}" + "".join(f"{c:>9.4f}" for c in cells) +
              f"{st.mean(r['gamma1']['nmi'] for r in rows):>9.4f}{st.mean(r['base']['nmi'] for r in rows):>9.4f}")
    print("\npaired per-seed wins vs S0 (Phase 3):")
    for k, d in kinds[1:]:
        for g in groups:
            rows = [r for gg, r in recs.values() if gg == g]
            diffs = [r[select(r, k, d)]["nmi"] - r["phase3"]["nmi"] for r in rows]
            w, l = sum(x > 1e-6 for x in diffs), sum(x < -1e-6 for x in diffs)
            print(f"  {k:<7}{g:<18} better {w:>2} / equal {len(diffs)-w-l:>2} / worse {l:>2}  mean {st.mean(diffs):+.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "records":
        cmd_records(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "final":
        cmd_final(sys.argv[2] if len(sys.argv) > 2 else "dev")
    elif cmd in ("real", "selectors", "merge", "kappa", "heldout"):
        globals()["cmd_" + cmd]()
    else:
        print(__doc__)
