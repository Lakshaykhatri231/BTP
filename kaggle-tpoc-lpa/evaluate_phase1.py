"""
Phase 1 gate: reproduce TPOC-LPA's Table 8/9 (non-overlapping NMI/Q) on
Karate and Football, and compare to the paper's published numbers.

Published (Table 8, NMI) / (Table 9, Q):
  Karate:   NMI 1.0000  Q 0.3715
  Football: NMI 0.9146  Q 0.6032
"""
import os
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community.quality import modularity
from tpoc_lpa import tpoc_lpa

HERE = os.path.dirname(os.path.abspath(__file__))


def load_karate():
    G = nx.karate_club_graph()
    truth = {n: (0 if G.nodes[n]['club'] == 'Mr. Hi' else 1) for n in G.nodes()}
    return G, truth


def load_football():
    G = nx.read_gml(os.path.join(HERE, 'data', 'football', 'football.gml'), label='id')
    truth = {n: G.nodes[n]['value'] for n in G.nodes()}
    return G, truth


def to_partition(label_dict, nodes):
    """dict node->label  ->  list of sets, for networkx modularity()."""
    groups = {}
    for n in nodes:
        groups.setdefault(label_dict[n], set()).add(n)
    return list(groups.values())


def evaluate(name, G, truth, alpha=1.0, beta=0.8):
    nodes = list(G.nodes())
    disjoint_label, overlap_comm, NI, iters = tpoc_lpa(G, alpha=alpha, beta=beta)

    y_true = [truth[n] for n in nodes]
    y_pred = [disjoint_label[n] for n in nodes]
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    q = modularity(G, to_partition(disjoint_label, nodes))
    n_comm_pred = len(set(y_pred))
    n_comm_true = len(set(y_true))
    overlap_count = sum(1 for n in nodes if len(overlap_comm[n]) > 1)

    print(f"--- {name} ---")
    print(f"  nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
          f"iters={iters} alpha={alpha} beta={beta}")
    print(f"  predicted communities: {n_comm_pred}  (ground truth: {n_comm_true})")
    print(f"  NMI = {nmi:.4f}   ARI = {ari:.4f}   Q = {q:.4f}")
    print(f"  overlapping nodes flagged by Stage 3: {overlap_count}/{len(nodes)}")
    return dict(name=name, nmi=nmi, ari=ari, q=q, n_comm_pred=n_comm_pred,
                n_comm_true=n_comm_true, overlap_count=overlap_count)


if __name__ == '__main__':
    G_k, t_k = load_karate()
    G_f, t_f = load_football()

    print("=" * 60)
    print("TPOC-LPA Phase 1 reproduction")
    print("Published targets -> Karate: NMI 1.0000, Q 0.3715 (Table 8/9)")
    print("                     Football: NMI 0.9146, Q 0.6032 (Table 8/9)")
    print("=" * 60)
    r_k = evaluate("Karate (34n/78e, 2 truth comms)", G_k, t_k)
    r_f = evaluate("Football (115n/613e, 12 truth comms)", G_f, t_f)
