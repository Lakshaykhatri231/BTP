import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovprec_snap import snap_label, CONF
from persona import persona_cover
from phase4_overlap import onmi, f1_avg, as_cover
from tpoc_lpa import community_label_filtering, node_importance, overlap_near_tie
for name in sys.argv[1:]:
    G, truth, L = snap_label(name, "best"); n = G.number_of_nodes(); NI = node_importance(G)
    t0 = time.time(); a, P = persona_cover(G, 3, **CONF["best"]); Lm = persona_cover.main
    W2 = community_label_filtering(G, NI, Lm, 0.8); T2 = overlap_near_tie(G, Lm, 0.5)
    rows = [("best WAS", community_label_filtering(G, NI, L, 0.8)), ("best near-tie", overlap_near_tie(G, L, 0.5)),
            ("persona only", a), ("persona+WAS", {v: a[v] | W2[v] for v in G}),
            ("persona+tie", {v: a[v] | T2[v] for v in G})]
    for rn, x in rows:
        cov = as_cover(x)
        print(f"{name} {rn:<14} ONMI_max {onmi(cov, truth, n):.4f} LFK {onmi(cov, truth, n, norm='lfk'):.4f} F1 {f1_avg(cov, truth):.4f} flag {sum(len(x[v])>1 for v in G)/n:.3f}", flush=True)
    print(f"  personas {P.number_of_nodes()} ({time.time()-t0:.0f}s)", flush=True)
