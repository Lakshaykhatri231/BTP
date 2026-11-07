"""persona_snap.py with progress: stage timings + a stack dump every 10 min showing where it is."""
import sys, os, time, faulthandler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
faulthandler.dump_traceback_later(600, repeat=True, file=sys.stdout)
from ovprec_snap import snap_label, CONF
import persona
from persona import persona_cover, persona_graph
from phase4_overlap import onmi, f1_avg, as_cover
from tpoc_lpa import community_label_filtering, node_importance, overlap_near_tie
import tpoc_lpa as T
for name in sys.argv[1:]:
    log(f"{name}: snap_label (loads graph + runs best pipeline on G) ...")
    G, truth, L = snap_label(name, "best"); n = G.number_of_nodes(); NI = node_importance(G)
    log(f"{name}: done, n={n} m={G.number_of_edges()}")
    t = time.time(); P, owner = persona_graph(G, 3)
    log(f"persona graph built: {P.number_of_nodes()} personas, {P.number_of_edges()} edges ({time.time()-t:.0f}s)")
    orig = persona.persona_graph; persona.persona_graph = lambda *a, **k: (P, owner)
    log("running tpoc_lpa best config on persona graph ...")
    a, P = persona_cover(G, 3, **CONF["best"]); Lm = persona_cover.main
    log("persona cover done")
    W2 = community_label_filtering(G, NI, Lm, 0.8); T2 = overlap_near_tie(G, Lm, 0.5)
    rows = [("best WAS", community_label_filtering(G, NI, L, 0.8)), ("best near-tie", overlap_near_tie(G, L, 0.5)),
            ("persona only", a), ("persona+WAS", {v: a[v] | W2[v] for v in G}),
            ("persona+tie", {v: a[v] | T2[v] for v in G})]
    for rn, x in rows:
        cov = as_cover(x)
        print(f"{name} {rn:<14} ONMI_max {onmi(cov, truth, n):.4f} LFK {onmi(cov, truth, n, norm='lfk'):.4f} F1 {f1_avg(cov, truth):.4f} flag {sum(len(x[v])>1 for v in G)/n:.3f}", flush=True)
