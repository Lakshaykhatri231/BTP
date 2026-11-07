"""
Direct validation of Stage 3 (WAS-based overlap filtering).

Test 1: the paper's own worked example (Table 2 + Fig. 6 text, Section 3.3).
Test 2: planted-overlap synthetic graph -- two cliques sharing known nodes;
        Stage 3 should flag the shared nodes as overlapping and nothing else.
"""
import networkx as nx
from tpoc_lpa import tpoc_lpa, community_label_filtering, node_importance


def test_paper_worked_example():
    """Paper Section 3.3 / Table 2: given these WAS values and beta=0.8,
    the paper states node 1 -> {C1}, node 2 -> {C1,C3}, node 5 -> {C3},
    node 6 -> {C1}, with a filtering threshold of 0.4."""
    was_table = {
        1: {"C1": 0.7404, "C3": 0.2596},
        2: {"C1": 0.4880, "C3": 0.5120},
        5: {"C1": 0.3111, "C3": 0.6889},
        6: {"C1": 0.7126, "C3": 0.2874},
    }
    stage2_label = {1: "C1", 2: "C3", 5: "C3", 6: "C1"}
    expected = {1: {"C1"}, 2: {"C1", "C3"}, 5: {"C3"}, 6: {"C1"}}
    beta = 0.8

    print("Test 1: paper's worked example (Table 2, beta=0.8)")
    all_ok = True
    for node, was in was_table.items():
        avg_was = sum(was.values()) / len(was)
        threshold = beta * avg_was
        valid = {c for c, v in was.items() if v >= threshold}
        result = {stage2_label[node]} | valid  # Algorithm 1 lines 50-52 (union)
        ok = result == expected[node]
        all_ok &= ok
        print(f"  node {node}: threshold={threshold:.4f} (paper says 0.4) "
              f"-> {sorted(result)}  expected {sorted(expected[node])}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  => {'ALL PASS' if all_ok else 'FAILURES PRESENT'}\n")
    return all_ok


def _two_groups_with_bridge(size=12, n_bridge=1, inter_edges=1, seed=0):
    """Two dense groups plus `n_bridge` node(s) wired into BOTH groups.
    The bridge node(s) are the planted overlapping nodes."""
    import random
    rng = random.Random(seed)
    A = list(range(size))
    B = list(range(size, 2 * size))
    G = nx.Graph()
    for grp in (A, B):
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                G.add_edge(grp[i], grp[j])
    bridges = []
    for b in range(n_bridge):
        bnode = 2 * size + b
        bridges.append(bnode)
        # wire the bridge node into roughly half of each group, symmetrically
        for target in rng.sample(A, size // 2) + rng.sample(B, size // 2):
            G.add_edge(bnode, target)
    # a couple of direct inter-group edges so the two groups aren't ONLY
    # reachable via the bridge (otherwise the bridge is a cut vertex)
    for k in range(inter_edges):
        G.add_edge(A[k], B[k])
    return G, set(bridges)


def test_planted_overlap():
    """A correct Stage 3 should flag the planted bridge node(s) -- which
    genuinely belong to both groups -- as overlapping, and leave the
    deep-interior nodes of each group alone."""
    print("Test 2: planted overlap (two dense groups + bridge node(s) in both)")
    for n_bridge in (1, 2):
        G, planted = _two_groups_with_bridge(size=12, n_bridge=n_bridge)
        print(f"  --- {n_bridge} bridge node(s): planted overlap = {sorted(planted)}, "
              f"n={G.number_of_nodes()} m={G.number_of_edges()} ---")
        for beta in [0.8, 1.0, 1.2]:
            disjoint_label, overlap_comm, NI, iters = tpoc_lpa(G, alpha=1.0, beta=beta)
            flagged = {n for n in G.nodes() if len(overlap_comm[n]) > 1}
            n_comms = len(set(disjoint_label.values()))
            hit = planted & flagged
            false_pos = flagged - planted
            print(f"    beta={beta}: disjoint_comms={n_comms}  flagged={sorted(flagged)}"
                  f"  -> caught {sorted(hit)}/{sorted(planted)}, false_pos={sorted(false_pos)}")
    print()


def test_beta_constraint():
    """Why beta governs whether overlap is detectable at all.

    WAS(i,c) over the k communities node i borders sums to 1 (Eq. 14's
    denominator is the NI-sum over ALL of i's neighbours). So avg_WAS = 1/k
    and the filter threshold is beta/k. Node i is overlapping iff >=2
    communities clear that threshold; if two values are both >= beta/k their
    sum is >= 2*beta/k, and the total is 1, so overlap requires beta <= k/2.

    For k=2 -- a node on the boundary between exactly two communities, by far
    the most common case -- this means beta <= 1.0, and beta = 1.0 requires an
    exact 50/50 affiliation tie (measure zero with real-valued NI weights).
    So beta MUST be < 1.0 for ordinary boundary nodes to ever be flagged.
    """
    print("Test 3: beta constraint -- overlap possible iff beta <= k/2")
    for k in (2, 3, 4):
        allowed = [b for b in (0.8, 0.9, 1.0, 1.1, 1.2) if b <= k / 2 + 1e-12]
        print(f"  k={k} communities: requires beta <= {k/2:.1f}  "
              f"-> usable beta from paper's range: {allowed}")
    print("  => beta < 1.0 is required in practice to detect overlap at "
          "ordinary (k=2) boundaries.\n")


if __name__ == '__main__':
    test_paper_worked_example()
    test_planted_overlap()
    test_beta_constraint()
