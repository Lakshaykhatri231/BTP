"""
Phase 2, step C: validate the Stage 4 policy with MOCK responders, and
measure how much headroom a semantic corrector actually has -- before
spending any GPU on a real LLM.

Responders:
  * oracle(p=1.0)   -- always returns the ground-truth community. This is the
                       UPPER BOUND on what any perfect semantic signal could
                       achieve. If the oracle barely helps, a real LLM cannot.
  * noisy oracle(p) -- ground truth with probability p, else a wrong candidate.
                       Sweeping p answers "how reliable must the LLM be for
                       this to pay off?", which is exactly the question behind
                       Alper's hardcoded delta_LLM = 0.95.
  * random          -- picks a candidate uniformly (floor / sanity check).
"""
import random
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community.quality import modularity

from tpoc_lpa import tpoc_lpa, node_importance
from stage4 import stage4_semantic_correction


def to_partition(label_dict, nodes):
    groups = {}
    for n in nodes:
        groups.setdefault(label_dict[n], set()).add(n)
    return list(groups.values())


def make_noisy_oracle(truth, label_of_truth_community, p, seed=0):
    """Returns ground truth with prob p, else a random wrong candidate.
    `label_of_truth_community` maps a ground-truth class -> the predicted
    community id that currently best represents it."""
    rng = random.Random(seed)

    def responder(node, candidates, context):
        gt_comm = label_of_truth_community.get(truth[node])
        if rng.random() < p and gt_comm is not None:
            return {gt_comm}
        wrong = [c for c in candidates if c != gt_comm]
        return {rng.choice(wrong)} if wrong else set()

    return responder


def make_random_responder(seed=0):
    rng = random.Random(seed)

    def responder(node, candidates, context):
        return {rng.choice(candidates)} if candidates else set()

    return responder


def majority_map(truth, label, nodes):
    """For each ground-truth class, the predicted community that holds most of
    its nodes -- so the oracle can speak in the algorithm's own label space."""
    from collections import Counter, defaultdict
    counts = defaultdict(Counter)
    for n in nodes:
        counts[truth[n]][label[n]] += 1
    return {gt: c.most_common(1)[0][0] for gt, c in counts.items()}


def run(name, G, truth, beta=0.8, budgets=(10, 25, 50, 1000)):
    nodes = list(G.nodes())
    y_true = [truth[n] for n in nodes]
    NI = node_importance(G)

    base_label, base_overlap, _, _ = tpoc_lpa(G, alpha=1.0, beta=beta)
    base_nmi = normalized_mutual_info_score(y_true, [base_label[n] for n in nodes])
    base_q = modularity(G, to_partition(base_label, nodes))
    print(f"=== {name} ({G.number_of_nodes()}n/{G.number_of_edges()}e) ===")
    print(f"  structural baseline (no Stage 4): NMI={base_nmi:.4f}  Q={base_q:.4f}  "
          f"comms={len(set(base_label.values()))}  truth_comms={len(set(y_true))}")

    gt_to_comm = majority_map(truth, base_label, nodes)

    print(f"  -- headroom sweep: perfect oracle, varying budget --")
    for B in budgets:
        resp = make_noisy_oracle(truth, gt_to_comm, p=1.0, seed=1)
        lab, ov, st = stage4_semantic_correction(
            G, base_label, beta=beta, budget=B, responder=resp, NI=NI)
        nmi = normalized_mutual_info_score(y_true, [lab[n] for n in nodes])
        q = modularity(G, to_partition(lab, nodes))
        print(f"    budget={B:<5} queried={st['queried']:<4} changes={st['label_changes']:<4} "
              f"NMI={nmi:.4f} ({nmi-base_nmi:+.4f})  Q={q:.4f}  overlap={st['overlap_nodes']}")

    print(f"  -- reliability sweep: how good must the LLM be? (budget=50) --")
    for p in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3):
        nmis = []
        for seed in range(5):
            resp = make_noisy_oracle(truth, gt_to_comm, p=p, seed=seed)
            lab, ov, st = stage4_semantic_correction(
                G, base_label, beta=beta, budget=50, responder=resp, NI=NI)
            nmis.append(normalized_mutual_info_score(y_true, [lab[n] for n in nodes]))
        mean = sum(nmis) / len(nmis)
        verdict = "helps" if mean > base_nmi + 1e-6 else ("neutral" if abs(mean-base_nmi) <= 1e-6 else "HURTS")
        print(f"    reliability p={p:<4} NMI={mean:.4f} ({mean-base_nmi:+.4f})  -> {verdict}")

    resp = make_random_responder(seed=3)
    lab, ov, st = stage4_semantic_correction(
        G, base_label, beta=beta, budget=50, responder=resp, NI=NI)
    nmi = normalized_mutual_info_score(y_true, [lab[n] for n in nodes])
    print(f"  -- random responder floor (budget=50): NMI={nmi:.4f} ({nmi-base_nmi:+.4f})")
    print()


if __name__ == '__main__':
    G = nx.read_gml('data/football/football.gml', label='id')
    truth = {n: G.nodes[n]['value'] for n in G.nodes()}
    run("Football", G, truth)

    G = nx.karate_club_graph()
    truth = {n: (0 if G.nodes[n]['club'] == 'Mr. Hi' else 1) for n in G.nodes()}
    run("Karate", G, truth, budgets=(5, 10, 34))
