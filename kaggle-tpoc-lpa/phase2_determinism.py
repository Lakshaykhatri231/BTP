"""
Phase 2, contribution 1: determinism.

TPOC-LPA as published breaks ties at random (the paper states this explicitly,
e.g. in its Fig. 4 walkthrough: "As their improved node importance are equal,
a random selection is made to designate one as a community"). That makes it
unstable across runs in exactly the way classic LPA is -- the defect MFLP and
NINP-LPA were specifically designed to remove.

Our variant fixes every tie with a deterministic rule. This script measures
the difference: N repeated runs of each variant on each graph, reporting
mean +/- std of NMI/Q and the number of DISTINCT partitions produced.

Protocol matches MFLP/NINP-LPA's own stability protocol (repeated runs,
report std; std = 0 <=> fully deterministic).
"""
import random
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity
from tpoc_lpa import tpoc_lpa
from dolphin_ground_truth import dolphin_truth_by_label

N_RUNS = 15


def to_partition(label_dict, nodes):
    groups = {}
    for n in nodes:
        groups.setdefault(label_dict[n], set()).add(n)
    return list(groups.values())


def partition_signature(label_dict, nodes):
    """Canonical form of a partition, so two runs that produce the same
    grouping (up to label renaming) compare equal."""
    groups = {}
    for n in nodes:
        groups.setdefault(label_dict[n], set()).add(n)
    return frozenset(frozenset(sorted(map(str, g))) for g in groups.values())


def stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var ** 0.5


def run_variant(G, truth, random_ties, n_runs=N_RUNS, alpha=1.0, beta=0.8):
    nodes = list(G.nodes())
    nmis, qs = [], []
    sigs = set()
    for r in range(n_runs):
        rng = random.Random(1000 + r) if random_ties else None
        dl, oc, NI, iters = tpoc_lpa(G, alpha=alpha, beta=beta, rng=rng)
        if truth is not None:
            nmis.append(normalized_mutual_info_score(
                [truth[n] for n in nodes], [dl[n] for n in nodes]))
        qs.append(modularity(G, to_partition(dl, nodes)))
        sigs.add(partition_signature(dl, nodes))
    return nmis, qs, sigs


def report(name, G, truth):
    nodes = list(G.nodes())
    print(f"--- {name} ({G.number_of_nodes()}n/{G.number_of_edges()}e), "
          f"{N_RUNS} runs each ---")
    for random_ties, tag in [(True, "paper (random tie-breaks)"),
                              (False, "ours (deterministic)")]:
        nmis, qs, sigs = run_variant(G, truth, random_ties)
        q_mean, q_std = stats(qs)
        line = f"  {tag:28s} Q = {q_mean:.4f} +/- {q_std:.4f}"
        if nmis:
            n_mean, n_std = stats(nmis)
            line += f"   NMI = {n_mean:.4f} +/- {n_std:.4f}"
        line += f"   distinct partitions: {len(sigs)}/{N_RUNS}"
        print(line)
    print()


if __name__ == '__main__':
    print("=" * 72)
    print("Determinism: published TPOC-LPA (random ties) vs our deterministic variant")
    print("=" * 72)

    G = nx.karate_club_graph()
    truth = {n: (0 if G.nodes[n]['club'] == 'Mr. Hi' else 1) for n in G.nodes()}
    report("Karate", G, truth)

    G = nx.read_gml('data/dolphins/dolphins.gml')
    tbl = dolphin_truth_by_label()
    report("Dolphin", G, {n: tbl[n] for n in G.nodes()})

    G = nx.read_gml('data/football/football.gml', label='id')
    report("Football", G, {n: G.nodes[n]['value'] for n in G.nodes()})

    G = nx.Graph(nx.read_gml('data/lesmis/lesmis.gml'))
    report("Lesmis", G, None)
