"""Latency / scalability study, mirroring Sec. 5.3 of Kong et al. (2026): overlapping LFR with k=5, kmax=10, mu=0.1, cmin=10, cmax=20, om=2, on=10,
N = 400, 800, ..., 25600. Every algorithm runs in-process, sequentially, on the same graph; wall-clock time (perf_counter) INCLUDES the
networkx->igraph conversion for the igraph methods. 3 repetitions, median reported. Ours also returns its per-stage timing (info dict).
usage: python3 run_latency.py   -> results/latency_lfr.json"""
import sys, os, json, time, subprocess, tempfile, statistics as st
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from baselines import *
import networkx as nx
from tpoc_lpa import tpoc_lpa
BIN = os.path.join(os.path.dirname(H), "phase5_overlap", "lfrbench")
NS = [400, 800, 1600, 3200, 6400, 12800, 25600]
REPS = 3

def gen(n, seed=1):
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/seed.txt", "w").write(str(1000 * seed + n))
        subprocess.run([BIN, "-N", str(n), "-k", "5", "-maxk", "10", "-mut", "0.1", "-muw", "0.1", "-minc", "10", "-maxc", "20", "-on", "10", "-om", "2",
                        "-name", "net", "-cnl", "1", "-seed", "seed.txt"], cwd=d, check=True, capture_output=True)
        G = nx.Graph(); G.add_nodes_from(range(1, n + 1))
        for line in open(f"{d}/net.nse"):
            if line.startswith("#"): continue
            a, b = line.split()[:2]
            if a != b: G.add_edge(int(a), int(b))
    return G

def slpa_run(G, seed=0):
    mem = slpa_memory(G, 100, seed); return slpa_cover(G, mem, 0.05)

ALG = {"Fastgreedy": fastgreedy, "LPA": lpa, "Louvain": louvain, "Walktrap": walktrap, "Infomap*": infomap, "SLPA": slpa_run,
       "TPOC-LPA (repro)": tpoc_paper, "Ours": tpoc_best}
res = {"N": NS, "reps": REPS, "graph": "overlapping LFR k=5 kmax=10 mu=0.1 cmin=10 cmax=20 om=2 on=10", "edges": {}, "time": {a: {} for a in ALG}, "stages": {}}
for n in NS:
    G = gen(n); res["edges"][n] = G.number_of_edges()
    print(f"N={n} nodes={G.number_of_nodes()} edges={G.number_of_edges()}", flush=True)
    for a, fn in ALG.items():
        ts = []
        for r in range(REPS):
            t = time.perf_counter(); fn(G, r); ts.append(time.perf_counter() - t)
        res["time"][a][n] = {"median": st.median(ts), "min": min(ts), "max": max(ts)}
        print(f"   {a:<18} median {st.median(ts):8.3f}s  (min {min(ts):.3f}, max {max(ts):.3f})", flush=True)
    stg = []
    for r in range(REPS):
        info = {}; tpoc_lpa(G, info=info, **BEST); stg.append(info)
    keys = ["t_node_importance", "t_stage1", "t_stage2", "t_refine", "t_merge", "t_stage3", "t_total"]
    res["stages"][n] = {k: st.median(i[k] for i in stg) for k in keys}
    print("   stages(Ours):", {k: round(v, 3) for k, v in res["stages"][n].items()}, flush=True)
    json.dump(res, open(f"{H}/results/latency_lfr.json", "w"), indent=1)
print("done")
