import sys, os, pickle, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/lakshaykhatri/Downloads/BTP/kaggle-tpoc-lpa")
from phase4_overlap import load_snap_overlap
from tpoc_lpa import tpoc_lpa
from ovprec import prec
from ovrules import rules, links
OUT = "/Users/lakshaykhatri/Downloads/BTP/kaggle-tpoc-lpa/data/phase5_cache/lfrov"
P4 = dict(sim="tri_overlap", guard=True, refine="granularity")
CONF = {"paper": {}, "phase4": P4, "best": dict(P4, merge=True, merge_criterion="mapeq", moves="synwalk"),
        "p4syn": dict(P4, moves="synwalk")}
def snap_label(name, conf):
    path = f"{OUT}/snap_{name}_{conf}.pkl"
    G, truth = load_snap_overlap(name)
    if not os.path.exists(path):
        L = tpoc_lpa(G, beta=0.8, **CONF[conf])[0]; pickle.dump(L, open(path, "wb"))
    return G, truth, pickle.load(open(path, "rb"))
if __name__ == "__main__":
    name, conf = sys.argv[1], sys.argv[2]
    G, truth, L = snap_label(name, conf)
    for rn, a in rules(G, L):
        if rn in ("WAS0.8", "tie0.5", "sig0.05"):
            p, t, byk = prec(G, L, truth, a)
            # precision split by node degree
            print(f"{name} {rn:<8} extra {t:6d} precision {p:.3f} by links {byk}", flush=True)
