"""Table 12 analogue: EQ (Eq. 21) of SLPA (own impl., r in 0.05..0.5, best EQ per graph -- a label-free choice) vs the paper algorithm
(our reproduction) and our best pipeline, both with WAS beta=0.8, on the paper's real networks. GREESE/MDPA/WLC/UMSTMO have no public
code: the paper's own numbers are quoted in the table. usage: python3 run_eq.py <graph> -> results/eq_<graph>.json"""
import sys, json, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import *
import networkx as nx
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def eq_fast(G, cover):
    """EQ = 1/2m [ sum_edges shared*2/(O_u O_v) - sum_c S_c^2/2m ],  S_c = sum_{u in c} k_u/O_u  (identical to eq_metric.eq_score)."""
    m = G.number_of_edges(); k = dict(G.degree()); O = {v: len(cover[v]) for v in G}
    S = defaultdict(float); a = 0.0
    for v, cs in cover.items():
        for c in cs: S[c] += k[v] / O[v]
    for u, v in G.edges():
        sh = len(cover[u] & cover[v])
        if sh: a += 2.0 * sh / (O[u] * O[v])
    return (a - sum(s * s for s in S.values()) / (2 * m)) / (2 * m)

def konect(d):
    G = nx.Graph()
    for l in open(f"{D}/eqnets/{d}/out.{d}"):
        if l[0] == "%" or not l.strip(): continue
        a, b = l.split()[:2]
        if a != b: G.add_edge(a, b)
    return G

def snapgz(p):
    import gzip; G = nx.Graph()
    for l in gzip.open(f"{D}/eqnets/{p}", "rt"):
        if l[0] == "#" or not l.strip(): continue
        a, b = l.split()[:2]
        if a != b: G.add_edge(a, b)
    return G

def load(name):
    import datasets
    if name == "karate": G = datasets.karate()[0]
    elif name == "dolphin": G = datasets.dolphin()[0]
    elif name == "lesmis": G = nx.Graph(nx.read_gml(f"{D}/lesmis/lesmis.gml"))
    elif name == "lastfm": G = datasets.lastfm_asia()[0]
    elif name == "jazz": G = konect("arenas-jazz")
    elif name == "power": G = konect("opsahl-powergrid")
    elif name == "pgp": G = konect("arenas-pgp")
    elif name == "facebook": G = snapgz("facebook_combined.txt.gz")
    elif name == "ca-grqc": G = snapgz("ca-GrQc.txt.gz")
    elif name == "ca-hepth": G = snapgz("ca-HepTh.txt.gz")
    elif name == "deezer":
        G = nx.read_edgelist(f"{D}/eqnets/deezer_europe/deezer_europe_edges.csv", delimiter=",", comments="n")
    elif name == "cora":   # full 2708-node Planetoid-size graph (the paper's), not the LCC
        G = nx.Graph()
        for l in open(f"{D}/cora/cora/cora.cites"):
            a, b = l.split()[:2]
            if a != b: G.add_edge(a, b)
    H = nx.Graph(); H.add_nodes_from(G.nodes()); H.add_edges_from(G.edges()); return H

if __name__ == "__main__":
    name = sys.argv[1]; G = load(name); res = {"graph": name, "n": G.number_of_nodes(), "m": G.number_of_edges()}
    from tpoc_lpa import tpoc_lpa
    for alg, conf in (("TPOC-LPA (repro)", {}), ("Ours", BEST)):
        for beta in ((0.8, 1.0) if name in ("karate", "dolphin") else (0.8,)):
            t = time.time(); ov = tpoc_lpa(G, beta=beta, **conf)[1]
            res[f"{alg} b{beta}"] = {"EQ": eq_fast(G, ov), "flag": sum(len(x) > 1 for x in ov.values()) / G.number_of_nodes(), "sec": round(time.time() - t)}
    mem = slpa_memory(G, 100, 0); best = None; t = time.time()
    for r in [round(0.05 * i, 2) for i in range(1, 11)]:
        c = slpa_cover(G, mem, r); e = eq_fast(G, c)
        if best is None or e > best["EQ"]: best = {"EQ": e, "r": r, "flag": sum(len(x) > 1 for x in c.values()) / G.number_of_nodes()}
    res["SLPA"] = best
    print(name, {k: (round(v["EQ"], 4) if isinstance(v, dict) else v) for k, v in res.items() if isinstance(v, dict)}, flush=True)
    json.dump(res, open(f"{os.path.dirname(os.path.abspath(__file__))}/results/eq_{name}.json", "w"), indent=1)
