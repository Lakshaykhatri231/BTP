"""Table 10/11 analogue: NMI and Q on the paper's six LFR settings N1-N6 (Table 5), seeds 1..S, calibrated realized mu.
usage: python3 run_lfr.py <N1..N6> [seeds]   -> results/lfr_<N>.json
Deviation: N1/N2 use cmin=15 (the paper's 10 is infeasible with networkx's generator, see lfr_sweep.py); tau1=3, tau2=1.5."""
import sys, json, os, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import *
from lfr_sweep import generate_lfr
NETS = {"N1": (1000, 50, 15, 50, 0.1), "N2": (1000, 50, 15, 50, 0.6), "N3": (5000, 50, 20, 100, 0.1),
        "N4": (5000, 50, 20, 100, 0.6), "N5": (10000, 100, 20, 200, 0.1), "N6": (10000, 100, 20, 200, 0.6)}
net = sys.argv[1]; S = int(sys.argv[2]) if len(sys.argv) > 2 else 5
n, kmax, cmin, cmax, mu = NETS[net]
res = {"net": net, "params": dict(n=n, kmax=kmax, cmin=cmin, cmax=cmax, mu=mu), "per_seed": {}}
for seed in range(1, S + 1):
    G, truth = generate_lfr(n=n, mu=mu, average_degree=20, max_degree=kmax, min_community=cmin, max_community=cmax, seed=seed)
    row = {}
    for alg, (fn, stochastic) in DISJOINT.items():
        reps = [0, 1, 2] if stochastic else [0]
        nm, qs = [], []
        for r in reps:
            L, err = run_timeout(fn, G, r, 3000)
            if L is None: row[alg] = {"error": err}; break
            nm.append(nmi_of(G, truth, L)); qs.append(modularity_of(G, L))
        else: row[alg] = {"NMI": st.mean(nm), "Q": st.mean(qs), "NMI_best": max(nm), "Q_best": qs[nm.index(max(nm))]}
    res["per_seed"][seed] = row
    print(net, seed, {a: round(v.get("NMI", -1), 4) for a, v in row.items()}, flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/lfr_{net}.json"), "w"), indent=1)
