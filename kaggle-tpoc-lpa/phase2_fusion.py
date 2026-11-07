"""
Phase 2, step C2: does soft fusion survive the reliability cliff?

The headroom experiment showed hard override (Alper's behaviour, designed for
a reliable frontier LLM) turns HARMFUL once the responder drops below ~90%
accuracy -- which is the regime a 4-bit quantized local 7B model plausibly
lives in. Soft fusion blends the LLM vote against the structural prior with
weight w, so the result should fall back towards the pure-structural baseline
instead of being dragged below it.

Also tests the calibration question directly (Alper hardcodes delta_LLM=0.95):
what happens when the trust weight w is MISCALIBRATED relative to the model's
true reliability p?
"""
import random
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score

from tpoc_lpa import tpoc_lpa, node_importance
from stage4 import stage4_semantic_correction
from phase2_headroom import make_noisy_oracle, majority_map

N_SEEDS = 5
BUDGET = 50


def sweep(name, G, truth):
    nodes = list(G.nodes())
    y_true = [truth[n] for n in nodes]
    NI = node_importance(G)
    base_label, _, _, _ = tpoc_lpa(G, alpha=1.0, beta=0.8)
    base = normalized_mutual_info_score(y_true, [base_label[n] for n in nodes])
    gt_to_comm = majority_map(truth, base_label, nodes)

    print(f"=== {name} === structural baseline NMI = {base:.4f}")
    print(f"{'reliability p':>14} | {'hard override':>14} | {'soft (w=p, calibrated)':>24} | "
          f"{'soft (w=0.95, Alper)':>22}")
    print("-" * 82)

    for p in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3):
        row = {}
        for mode, kwargs in [("hard", dict(fusion="hard")),
                              ("soft_cal", dict(fusion="soft", trust=p)),
                              ("soft_fixed", dict(fusion="soft", trust=0.95))]:
            vals = []
            for seed in range(N_SEEDS):
                resp = make_noisy_oracle(truth, gt_to_comm, p=p, seed=seed)
                lab, ov, st = stage4_semantic_correction(
                    G, base_label, beta=0.8, budget=BUDGET, responder=resp,
                    NI=NI, **kwargs)
                vals.append(normalized_mutual_info_score(
                    y_true, [lab[n] for n in nodes]))
            row[mode] = sum(vals) / len(vals)

        def fmt(v):
            return f"{v:.4f} ({v-base:+.4f})"
        print(f"{p:>14} | {fmt(row['hard']):>14} | {fmt(row['soft_cal']):>24} | "
              f"{fmt(row['soft_fixed']):>22}")
    print()


if __name__ == '__main__':
    G = nx.read_gml('data/football/football.gml', label='id')
    sweep("Football", G, {n: G.nodes[n]['value'] for n in G.nodes()})

    G = nx.karate_club_graph()
    sweep("Karate", G, {n: (0 if G.nodes[n]['club'] == 'Mr. Hi' else 1) for n in G.nodes()})
