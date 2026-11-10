"""Run TPOC-LPA (baseline) and/or SN-LPA (proposed) on a graph and print the results.

Examples
  python3 run.py --dataset karate                       # both algorithms on Karate
  python3 run.py --dataset football --algo sn-lpa
  python3 run.py --dataset lfr --n 1000 --mu 0.6 --seed 1
  python3 run.py --edges my_graph.txt --truth my_labels.txt --save out
  python3 run.py --list                                 # list the bundled datasets

Datasets are read from ./datasets (set TPOC_DATA to use another folder).
Graphs are treated as undirected and unweighted.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TPOC_DATA", os.path.join(HERE, "datasets"))
sys.path.insert(0, os.path.join(HERE, "kaggle-tpoc-lpa"))

import networkx as nx                                                   # noqa: E402
from networkx.algorithms.community.quality import modularity            # noqa: E402
from sklearn.metrics import normalized_mutual_info_score               # noqa: E402

import datasets as D                                                    # noqa: E402
from tpoc_lpa import tpoc_lpa                                           # noqa: E402

# SN-LPA = TPOC-LPA with these options switched on (see README, "Method")
SN_LPA = dict(sim="tri_overlap", guard=True, refine="granularity",
              merge=True, merge_criterion="mapeq", moves="synwalk")
ALGOS = {"tpoc-lpa": {}, "sn-lpa": SN_LPA}

REAL = ["karate", "dolphin", "football", "polbooks", "polblogs", "email-Eu", "cora", "citeseer"]
SNAP = ["amazon", "dblp", "youtube"]


def load(args):
    """Return (G, truth or None, name)."""
    if args.edges:
        G = nx.Graph()
        with open(args.edges) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2 and not line.startswith("#"):
                    G.add_edge(p[0], p[1])
        truth = None
        if args.truth:
            truth = {}
            with open(args.truth) as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 2 and not line.startswith("#"):
                        truth[p[0]] = p[1]
        return G, truth, os.path.basename(args.edges)
    name = args.dataset
    if name == "lfr":
        from lfr_sweep import generate_lfr
        G, truth = generate_lfr(args.n, args.mu, seed=args.seed)[:2]
        return G, truth, f"LFR n={args.n} mu={args.mu} seed={args.seed}"
    if name in SNAP:
        from load_snap import load_snap_network
        G, truth, _ = load_snap_network(name)
        return G, truth, f"SNAP-{name}"
    if name in D.ALL:
        G, truth = D.ALL[name]()
        return G, truth, name
    sys.exit(f"unknown dataset {name!r}; run with --list")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="karate", help="bundled dataset name, or 'lfr' (default: karate)")
    ap.add_argument("--algo", default="both", choices=["both", "tpoc-lpa", "sn-lpa"])
    ap.add_argument("--edges", help="your own graph: an edge list, one 'u v' pair per line")
    ap.add_argument("--truth", help="optional ground truth for --edges: one 'node label' pair per line")
    ap.add_argument("--n", type=int, default=1000, help="LFR: number of nodes")
    ap.add_argument("--mu", type=float, default=0.6, help="LFR: mixing parameter (realised)")
    ap.add_argument("--seed", type=int, default=1, help="LFR: random seed")
    ap.add_argument("--beta", type=float, default=0.8, help="Stage 3 overlap threshold (default 0.8)")
    ap.add_argument("--save", help="write <prefix>_<algo>.tsv with node, community, overlapping communities")
    ap.add_argument("--list", action="store_true", help="list the bundled datasets and exit")
    args = ap.parse_args()

    if args.list:
        print("real networks :", ", ".join(REAL))
        print("SNAP networks :", ", ".join(SNAP), "(slow: DBLP takes about a minute)")
        print("synthetic     : lfr  (use --n, --mu, --seed)")
        return

    G0, truth, name = load(args)
    G = nx.Graph()                                  # undirected, unweighted, no self-loops
    G.add_nodes_from(G0.nodes())
    G.add_edges_from((u, v) for u, v in G0.edges() if u != v)
    print(f"{name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
          + (f", {len(set(truth.values()))} ground-truth classes" if truth else ""))
    print(f"{'algorithm':<10} {'communities':>11} {'overlapping':>11} {'Q':>7} {'NMI':>7} {'time (s)':>9}")

    for algo in (["tpoc-lpa", "sn-lpa"] if args.algo == "both" else [args.algo]):
        t0 = time.perf_counter()
        labels, cover, _, _ = tpoc_lpa(G, beta=args.beta, **ALGOS[algo])
        sec = time.perf_counter() - t0
        groups = {}
        for v, c in labels.items():
            groups.setdefault(c, set()).add(v)
        q = modularity(G, list(groups.values()), weight=None)
        nodes = [v for v in G if truth and v in truth]
        nmi = normalized_mutual_info_score([str(truth[v]) for v in nodes], [str(labels[v]) for v in nodes]) if nodes else None
        n_ov = sum(1 for s in cover.values() if len(s) > 1)
        print(f"{algo:<10} {len(groups):>11} {n_ov:>11} {q:>7.4f} {(f'{nmi:.4f}' if nmi is not None else '-'):>7} {sec:>9.2f}")
        if args.save:
            path = f"{args.save}_{algo}.tsv"
            with open(path, "w") as f:
                f.write("node\tcommunity\tall_communities\n")
                for v in G:
                    f.write(f"{v}\t{labels[v]}\t{','.join(map(str, sorted(cover[v], key=str)))}\n")
            print(f"           saved {path}")


if __name__ == "__main__":
    main()
