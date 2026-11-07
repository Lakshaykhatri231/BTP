"""Phase 5: map-equation criteria for the merge pass (CHANGE 6) and the guard (CHANGE 7).

Reuses the Phase 4 graph sets, loaders and S5 selector (phase4_ablation). Records go to
data/phase5_cache/rec_<gid>.json so the Phase 4 cache stays untouched.

  python3 phase5_ablation.py records dev|heldout|fresh   # compute (slow, cached)
  python3 phase5_ablation.py report  dev|heldout|fresh   # tables, instant
"""
import json
import os
import statistics as st
import sys
import time

import datasets
import phase4_ablation as p4

CACHE = os.path.join(datasets.D, "phase5_cache")
os.makedirs(CACHE, exist_ok=True)

TRI = dict(sim="tri_overlap")
REGIMES = {
    "base":      dict(),
    "phase3":    dict(TRI, guard=True),
    "gamma1":    dict(TRI, gamma=1.0),
    "p3_qmerge": dict(TRI, guard=True, merge=True, merge_min_gain=0.0),     # CHANGE 5 (gate applied offline)
    "g1_qmerge": dict(TRI, gamma=1.0, merge=True, merge_min_gain=0.0),
    "p3_mapeq":  dict(TRI, guard=True, merge=True, merge_criterion="mapeq"),  # CHANGE 6
    "g1_mapeq":  dict(TRI, gamma=1.0, merge=True, merge_criterion="mapeq"),
    "p3_mguard": dict(TRI, guard=True, guard_criterion="mapeq"),             # CHANGE 7 (dev only: failed there)
    "p4_moves":  dict(TRI, guard=True, refine="granularity", moves=True),     # CHANGE 8
    "p4_mapeq_moves": dict(TRI, guard=True, refine="granularity", merge=True, merge_criterion="mapeq", moves=True),
    "p4_syn":    dict(TRI, guard=True, refine="granularity", moves="synwalk"),   # CHANGE 9
    "p4_mapeq_syn": dict(TRI, guard=True, refine="granularity", merge=True, merge_criterion="mapeq", moves="synwalk"),
}
# identical kwargs in Phase 4 (outputs hash-verified unchanged), so those cached results are reused
FROM_P4 = {"base": "base", "phase3": "phase3", "gamma1": "gamma1", "p3_qmerge": "p3_merge", "g1_qmerge": "g1_merge"}


def record(gid, G, truth, dev=True):
    p = os.path.join(CACHE, f"rec_{gid}.json")
    rec = json.load(open(p)) if os.path.exists(p) else {}
    dirty = False
    p4p = os.path.join(p4.CACHE, f"rec_{gid}.json")
    old = json.load(open(p4p)) if os.path.exists(p4p) else {}
    for name, kw in REGIMES.items():
        if name == "p3_mguard" and not dev:
            continue
        if name not in rec and FROM_P4.get(name) in old:
            rec[name] = old[FROM_P4[name]]
            dirty = True
        if name not in rec:
            rec[name] = p4.evaluate(G, truth, **kw)
            dirty = True
    if dirty:
        json.dump(rec, open(p, "w"))
    return rec


def load(which):
    out = {}
    for gid, group, _ in p4.graph_specs(which):
        p = os.path.join(CACHE, f"rec_{gid}.json")
        if os.path.exists(p):
            out[gid] = (group, json.load(open(p)))
    return out


def s5(rec, guard_reg="phase3"):
    return "gamma1" if rec["gamma1"]["nc"] <= p4.S6_RHO * rec[guard_reg]["nc"] else guard_reg


def finals(rec):
    """Final configurations, each = S5 selection then the merge variant on the selected regime."""
    reg = s5(rec)
    qm = rec["g1_qmerge" if reg == "gamma1" else "p3_qmerge"]
    me = rec["g1_mapeq" if reg == "gamma1" else "p3_mapeq"]
    out = {
        "paper": rec["base"],
        "phase4": rec[reg],
        "p4+merge.03": qm if qm["merge_gain"] >= 0.03 and qm["merges"] > 0 else rec[reg],
        "p4+mapeq": me,
        "p4+moves": rec["p4_moves"],
        "p4+mapeq+mv": rec["p4_mapeq_moves"],
        "p4+syn": rec["p4_syn"],
        "p4+mapeq+syn": rec["p4_mapeq_syn"],
    }
    if "p3_mguard" in rec:
        out["p4[mguard]"] = rec[s5(rec, "p3_mguard")]
    return out


def cmd_records(which):
    for gid, group, loader in p4.graph_specs(which):
        t0 = time.time()
        G, truth = loader()
        rec = record(gid, G, truth, dev=(which == "dev"))
        f = finals(rec)
        print(f"  {gid:<26} n={G.number_of_nodes():>6}  " +
              "  ".join(f"{k} {v['nmi']:.4f}" for k, v in f.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)


def cmd_report(which):
    recs = load(which)
    groups = sorted({g for g, _ in recs.values()}, key=lambda g: (g.startswith(("lfr", "coarse", "small")), g))
    cols = list(finals(next(iter(recs.values()))[1]).keys())
    print(f"{which}: mean NMI per group (nc = communities of the last column pair)")
    print(f"{'group':<20}" + "".join(f"{c:>13}" for c in cols) + f"{'nc p4/mapeq':>14}")
    per = {c: [] for c in cols}
    for g in groups:
        rows = [finals(r) for gg, r in recs.values() if gg == g]
        means = {c: st.mean(r[c]["nmi"] for r in rows) for c in cols}
        for c in cols:
            per[c].append(means[c])
        nc = f"{st.mean(r['phase4']['nc'] for r in rows):.0f}/{st.mean(r['p4+mapeq']['nc'] for r in rows):.0f}"
        print(f"{g:<20}" + "".join(f"{means[c]:>13.4f}" for c in cols) + f"{nc:>14}")
    print(f"{'MEAN':<20}" + "".join(f"{st.mean(per[c]):>13.4f}" for c in cols))
    for c in cols[2:]:
        d = [finals(r)[c]["nmi"] - finals(r)["phase4"]["nmi"] for _, r in recs.values()]
        print(f"  {c:<12} vs phase4 per graph: better {sum(x > 1e-4 for x in d)} / equal "
              f"{sum(abs(x) <= 1e-4 for x in d)} / worse {sum(x < -1e-4 for x in d)}")


if __name__ == "__main__":
    {"records": cmd_records, "report": cmd_report}[sys.argv[1]](sys.argv[2])
