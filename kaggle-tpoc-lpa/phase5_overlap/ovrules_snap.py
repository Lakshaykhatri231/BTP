import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovprec_snap import snap_label
from ovrules import rules2 as rules
from phase4_overlap import onmi, f1_avg, as_cover
name, conf = sys.argv[1], sys.argv[2]
G, truth, L = snap_label(name, conf); n = G.number_of_nodes()
for rn, a in rules(G, L):
    cov = as_cover(a)
    print(f"{name} {conf} {rn:<9} ONMI_max {onmi(cov, truth, n):.4f} LFK {onmi(cov, truth, n, norm='lfk'):.4f} F1 {f1_avg(cov, truth):.4f} flag {sum(len(a[v])>1 for v in G)/n:.3f}", flush=True)
