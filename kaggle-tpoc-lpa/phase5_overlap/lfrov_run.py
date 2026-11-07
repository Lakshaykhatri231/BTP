"""Paper Table 13/14 benchmark: overlapping LFR R1-R6, om 2..8. Per config: LFK NMI, NMI_max, F1, flagged rate.
Disjoint labels are cached so Stage 3 rules can be re-scored offline."""
import sys, time, os, pickle
from collections import Counter, defaultdict
sys.path.insert(0, "/Users/lakshaykhatri/Downloads/BTP/kaggle-tpoc-lpa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_overlap import onmi, f1_avg, as_cover
from tpoc_lpa import tpoc_lpa
from lfrov_gen import gen, R, OUT

P4 = dict(sim="tri_overlap", guard=True, refine="granularity")
CONF = {"paper": {}, "phase4": P4, "best": dict(P4, merge=True, merge_criterion="mapeq", moves="synwalk")}

def run(r, om, seed, conf):
    path = f"{OUT}/lab_{r}_om{om}_s{seed}_{conf}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    G, truth = gen(r, om, seed)
    t0 = time.time()
    L, ov, NI, _ = tpoc_lpa(G, beta=0.8, **CONF[conf])
    res = (L, {v: set(s) for v, s in ov.items()}, time.time() - t0)
    pickle.dump(res, open(path, "wb"))
    return res

def score(G, truth, assign):
    n = G.number_of_nodes(); cov = as_cover(assign)
    return (onmi(cov, truth, n, norm="lfk"), onmi(cov, truth, n), f1_avg(cov, truth),
            sum(len(assign[v]) > 1 for v in G) / n)

if __name__ == "__main__":
    seed = int(sys.argv[1]); confs = sys.argv[2].split(","); rs = sys.argv[3].split(",") if len(sys.argv) > 3 else list(R)
    for r in rs:
        for om in range(2, 9):
            G, truth = gen(r, om, seed)
            for c in confs:
                L, ov, t = run(r, om, seed, c)
                a, b, f, o = score(G, truth, ov)
                print(f"{r} om{om} {c:<7} NMI {a:.4f} NMImax {b:.4f} F1 {f:.4f} flag {o:.3f} comms {len(set(L.values()))} ({t:.0f}s)", flush=True)
