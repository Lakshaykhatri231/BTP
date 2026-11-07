"""Ground-truth benchmark registry.

Every loader returns (G, truth) with G an undirected simple graph restricted to
its largest connected component (community detection on a disconnected graph
conflates "found a community" with "found a component), and truth mapping every
node of G to a string class label.
"""
import gzip
import json
import os
import tarfile

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
# TPOC_DATA lets the same loaders run on Kaggle or on the teacher-ready bundle (BTP/datasets)
D = os.environ.get("TPOC_DATA", os.path.join(HERE, "data"))


def _lcc(G, truth):
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        return G, {}
    keep = max(nx.connected_components(G), key=len)
    G = G.subgraph(keep).copy()
    return G, {n: truth[n] for n in G.nodes()}


def karate():
    G = nx.karate_club_graph()
    return _lcc(G, {n: str(G.nodes[n]["club"]) for n in G.nodes()})


def dolphin():
    from dolphin_ground_truth import dolphin_truth_by_label
    G = nx.read_gml(f"{D}/dolphins/dolphins.gml")
    t = dolphin_truth_by_label()
    return _lcc(G, {n: str(t[n]) for n in G.nodes()})


def football():
    G = nx.read_gml(f"{D}/football/football.gml", label="id")
    return _lcc(G, {n: str(G.nodes[n]["value"]) for n in G.nodes()})


def polbooks():
    G = nx.read_gml(f"{D}/polbooks/polbooks.gml")
    return _lcc(G, {n: str(G.nodes[n]["value"]) for n in G.nodes()})


def polblogs():
    p = json.load(open(f"{D}/polblogs/polblogs_lcc.json"))
    G = nx.Graph()
    G.add_nodes_from(p["labels"])
    G.add_edges_from((p["labels"][a], p["labels"][b]) for a, b in p["edges"])
    return _lcc(G, {l: str(v) for l, v in zip(p["labels"], p["value"])})


def email_eu():
    """SNAP email-Eu-core: 1005 researchers, 42 departments as ground truth."""
    G = nx.Graph()
    with gzip.open(f"{D}/email/edges.txt.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            a, b = line.split()
            G.add_edge(int(a), int(b))
    truth = {}
    with gzip.open(f"{D}/email/labels.txt.gz", "rt") as f:
        for line in f:
            n, c = line.split()
            truth[int(n)] = str(c)
    G.add_nodes_from(truth)
    return _lcc(G, truth)


def _linqs(name):
    """Cora / Citeseer from LINQS: <id> <features...> <class> plus a cites file."""
    base = f"{D}/{name}/{name}"
    truth = {}
    with open(f"{base}/{name}.content") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            truth[parts[0]] = parts[-1]
    G = nx.Graph()
    G.add_nodes_from(truth)
    with open(f"{base}/{name}.cites") as f:
        for line in f:
            a, b = line.split()
            # citeseer's cites file references a few ids absent from .content
            if a in truth and b in truth:
                G.add_edge(a, b)
    return _lcc(G, truth)


def cora():
    return _linqs("cora")


def citeseer():
    return _linqs("citeseer")


def snap(name):
    from load_snap import load_snap_network
    G, truth, _ = load_snap_network(name)
    return _lcc(G, {n: str(v) for n, v in truth.items()})


SMALL = {          # the Phase 1 core, kept for regression comparability
    "karate": karate, "dolphin": dolphin, "football": football,
    "polbooks": polbooks,
}
MEDIUM = {         # added in Phase 3 for statistical power
    "polblogs": polblogs, "email-Eu": email_eu, "cora": cora, "citeseer": citeseer,
}
LARGE = {          # SNAP ground-truth communities
    "amazon": lambda: snap("amazon"),
    "dblp": lambda: snap("dblp"),
    "youtube": lambda: snap("youtube"),
}
REAL = {**SMALL, **MEDIUM}
ALL = {**REAL, **LARGE}


if __name__ == "__main__":
    for name, fn in ALL.items():
        try:
            G, t = fn()
            print(f"  {name:<10} {G.number_of_nodes():>7} nodes {G.number_of_edges():>8} edges "
                  f"{len(set(t.values())):>5} classes")
        except Exception as e:
            print(f"  {name:<10} FAILED: {type(e).__name__}: {str(e)[:80]}")


def lfr_npz(name):
    """Load an exported LFR graph: <D>/lfr/<name>.npz holds `edges` (M x 2) and `truth`
    (community id of node i). Node ids are 0..n-1, exactly as generated."""
    import numpy as np
    z = np.load(os.path.join(D, "lfr", name + ".npz"))
    G = nx.Graph()
    G.add_nodes_from(range(len(z["truth"])))
    G.add_edges_from(map(tuple, z["edges"].tolist()))
    return G, {i: int(c) for i, c in enumerate(z["truth"])}


# ---- Phase 5 fresh validation set #2 (fetched 2026-10-24, used ONCE for validation, never for design) ----
def lastfm_asia():
    """SNAP LastFM-Asia social network (Rozemberczki & Sarkar 2020); truth = user country (18 classes)."""
    import csv
    base = os.path.join(D, "fresh2", "lastfm", "lasftm_asia")
    G = nx.Graph()
    with open(os.path.join(base, "lastfm_asia_edges.csv")) as f:
        for r in csv.DictReader(f):
            G.add_edge(int(r["node_1"]), int(r["node_2"]))
    with open(os.path.join(base, "lastfm_asia_target.csv")) as f:
        truth = {int(r["id"]): str(r["target"]) for r in csv.DictReader(f)}
    return _lcc(G, truth)


def wikics():
    """Wiki-CS hyperlink graph (Mernyei & Cangea 2020), made undirected; truth = CS branch (10 classes)."""
    import json
    d = json.load(open(os.path.join(D, "fresh2", "wikics.json")))
    G = nx.Graph()
    for i, nbrs in enumerate(d["links"]):
        G.add_node(i)
        for j in nbrs:
            G.add_edge(i, j)
    return _lcc(G, {i: str(l) for i, l in enumerate(d["labels"])})


FRESH2 = {"lastfm_asia": lastfm_asia, "wikics": wikics}
