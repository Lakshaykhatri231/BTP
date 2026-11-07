"""Table 8/9 analogue: NMI and Q of every runnable baseline and of our pipeline on one real graph.
usage: python3 run_real.py <graph>   -> results/real_<graph>.json
Graphs are made unweighted (networkx's Karate carries edge weights that igraph ignores and the paper does not use)."""
import sys, json, time, os, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import *
import networkx as nx
name = sys.argv[1]
import datasets
if name in ("amazon", "dblp", "youtube"):
    from load_snap import load_snap_network
    G0, truth, _ = load_snap_network(name); truth = {n: str(v) for n, v in truth.items()}
elif name == "lastfm":
    G0, truth = datasets.lastfm_asia()
else:
    G0, truth = datasets.ALL[name]()
G = nx.Graph(); G.add_nodes_from(G0.nodes()); G.add_edges_from(G0.edges())
big = G.number_of_nodes() > 5000
TO = 3000 if G.number_of_nodes() > 50000 else 1200
out = {"graph": name, "n": G.number_of_nodes(), "m": G.number_of_edges(), "classes": len(set(truth.values())), "algs": {}}
for alg, (fn, stochastic) in DISJOINT.items():
    seeds = list(range(3 if big else 10)) if stochastic else [0]
    nm, qs, cs, t0 = [], [], [], time.time()
    for s in seeds:
        L, err = run_timeout(fn, G, s, TO)
        if L is None: out["algs"][alg] = {"error": err, "sec": round(time.time() - t0)}; break
        nm.append(nmi_of(G, truth, L)); qs.append(modularity_of(G, L)); cs.append(len(set(L.values())))
    else:
        out["algs"][alg] = {"NMI": st.mean(nm), "NMI_sd": st.pstdev(nm), "NMI_best": max(nm), "Q_best": qs[nm.index(max(nm))], "Q": st.mean(qs), "comms": st.mean(cs), "runs": len(seeds), "sec": round(time.time() - t0)}
    print(name, alg, out["algs"][alg], flush=True)
    json.dump(out, open(f"{os.path.dirname(os.path.abspath(__file__))}/results/real_{name}.json", "w"), indent=1)
