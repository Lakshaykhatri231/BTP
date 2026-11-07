"""Latency on real graphs (idle machine, in-process, sequential). Fast algorithms: 3 repetitions (median); algorithms slower than 20 s: 1 run.
Includes the networkx->igraph conversion for igraph methods. usage: python3 run_latency_real.py -> results/latency_real.json"""
import sys, os, json, time, statistics as st
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from baselines import *
import networkx as nx, datasets
from load_snap import load_snap_network
GR = ["karate", "dolphin", "football", "polblogs", "email-Eu", "cora", "citeseer", "amazon", "youtube", "dblp"]
ALG = {"Fastgreedy": fastgreedy, "LPA": lpa, "Louvain": louvain, "Walktrap": walktrap, "Infomap*": infomap, "TPOC-LPA (repro)": tpoc_paper, "Ours": tpoc_best}
out = {}
for g in GR:
    if g in ("amazon", "youtube", "dblp"):
        G0, _, _ = load_snap_network(g)
    else:
        G0, _ = datasets.ALL[g]()
    G = nx.Graph(); G.add_nodes_from(G0.nodes()); G.add_edges_from(G0.edges())
    out[g] = {"n": G.number_of_nodes(), "m": G.number_of_edges(), "time": {}}
    for a, fn in ALG.items():
        ts = []
        t = time.perf_counter(); fn(G, 0); ts.append(time.perf_counter() - t)
        if ts[0] < 20:
            for r in (1, 2):
                t = time.perf_counter(); fn(G, r); ts.append(time.perf_counter() - t)
        out[g]["time"][a] = {"median": st.median(ts), "runs": len(ts)}
        print(f"{g:<9} {a:<18} {st.median(ts):9.3f}s ({len(ts)} run{'s' if len(ts)>1 else ''})", flush=True)
    json.dump(out, open(f"{H}/results/latency_real.json", "w"), indent=1)
print("done")
