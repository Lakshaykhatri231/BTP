"""
Load SNAP com-* ground-truth community networks (Yang & Leskovec 2012),
the exact citation the TPOC-LPA paper uses for its Amazon/DBLP/Youtube rows
in Table 4.

Paper's reported sizes (Table 4): Amazon 6428n/16223e/893 comms,
DBLP 23654n/75618e/2974 comms, Youtube 10364n/18717e/2798 comms.

The full SNAP graphs are much bigger (hundreds of thousands of nodes).
Standard practice (matching these exact node counts) is: take the "top
5000 quality communities" ground-truth file, build the induced subgraph
on the union of nodes appearing in those communities, and assign each
node to the community it appears in (first one, if a node appears in
more than one, to get a single-label ground truth for non-overlapping
NMI/Q comparison).
"""
import gzip
import os
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(os.environ.get("TPOC_DATA", os.path.join(HERE, "data")), "snap")


def load_snap_network(name):
    cmty_path = os.path.join(SNAP_DIR, f"com-{name}.top5000.cmty.txt.gz")
    graph_path = os.path.join(SNAP_DIR, f"com-{name}.ungraph.txt.gz")

    communities = []
    with gzip.open(cmty_path, 'rt') as f:
        for line in f:
            nodes = [int(x) for x in line.split()]
            if nodes:
                communities.append(nodes)

    truth = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            if n not in truth:  # first community wins -> single-label ground truth
                truth[n] = cid

    keep_nodes = set(truth.keys())

    G = nx.Graph()
    G.add_nodes_from(keep_nodes)
    with gzip.open(graph_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            a, b = line.split()
            a, b = int(a), int(b)
            if a in keep_nodes and b in keep_nodes:
                G.add_edge(a, b)

    # drop isolated nodes (no edges within the induced subgraph) to match
    # how these "top-5000 induced subgraph" benchmarks are usually built
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    for n in isolates:
        truth.pop(n, None)

    return G, truth, len(communities)


if __name__ == '__main__':
    targets = {
        'amazon': (6428, 16223, 893),
        'dblp': (23654, 75618, 2974),
        'youtube': (10364, 18717, 2798),
    }
    for name, (n_t, e_t, c_t) in targets.items():
        G, truth, n_comms_raw = load_snap_network(name)
        n_comms_used = len(set(truth.values()))
        print(f"{name}: nodes={G.number_of_nodes()} (paper {n_t}), "
              f"edges={G.number_of_edges()} (paper {e_t}), "
              f"ground-truth comms used={n_comms_used} (paper {c_t}, "
              f"raw file had {n_comms_raw})")
