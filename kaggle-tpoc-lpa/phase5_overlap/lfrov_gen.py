"""Generate the paper's Table 7 overlapping LFR networks (R1-R6, om 2..8) with the
Lancichinetti-Fortunato generator (t1=2, t2=1 defaults; unweighted use: weights ignored)."""
import os, subprocess, sys, pickle, tempfile
import networkx as nx
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfrbench")  # build: g++ -O3 -w -o lfrbench lfr_overlap_generator/benchm.cpp
OUT = "/Users/lakshaykhatri/Downloads/BTP/kaggle-tpoc-lpa/data/phase5_cache/lfrov"
R = {"R1": (1000, 10, 50, 0.1, 10, 50, 100), "R2": (1000, 10, 50, 0.3, 10, 50, 100),
     "R3": (5000, 10, 50, 0.1, 20, 100, 500), "R4": (5000, 10, 50, 0.3, 20, 100, 500),
     "R5": (10000, 20, 100, 0.1, 20, 100, 2000), "R6": (10000, 20, 100, 0.3, 20, 100, 2000)}

def gen(r, om, seed):
    path = f"{OUT}/{r}_om{om}_s{seed}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    n, k, kmax, mu, cmin, cmax, on = R[r]
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/seed.txt", "w").write(str(1000 * seed + om))
        subprocess.run([BIN, "-N", str(n), "-k", str(k), "-maxk", str(kmax), "-mut", str(mu), "-muw", str(mu),
                        "-minc", str(cmin), "-maxc", str(cmax), "-on", str(on), "-om", str(om),
                        "-name", "net", "-cnl", "1", "-seed", "seed.txt"], cwd=d, check=True, capture_output=True)
        G = nx.Graph()
        G.add_nodes_from(range(1, n + 1))
        for line in open(f"{d}/net.nse"):
            if line.startswith("#"): continue
            a, b = line.split()[:2]
            G.add_edge(int(a), int(b))
        truth = [set(map(int, l.split())) for l in open(f"{d}/net.cnl") if l.strip()]
    pickle.dump((G, truth), open(path, "wb"))
    return G, truth

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for r in R:
        for om in range(2, 9):
            G, t = gen(r, om, seed)
            from collections import Counter
            c = Counter(v for x in t for v in x)
            print(r, om, G.number_of_nodes(), G.number_of_edges(), len(t), sum(1 for v in c if c[v] > 1), flush=True)
