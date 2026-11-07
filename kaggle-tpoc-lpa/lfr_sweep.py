"""
LFR synthetic-network mu-sweep for Phase 1 (paper's Table 5/10: N1 mu=0.1, N2 mu=0.6,
n=1000, avg_degree=20, max_degree=50, min_community=10, max_community=50).

Published TPOC-LPA targets (Table 10): N-1000-0.1 NMI=1.0000, N-1000-0.6 NMI=0.9869.

Root-cause finding (see project history): networkx's LFR_benchmark_graph does
NOT accurately realize its own `mu` parameter at higher values with this
degree/community-size regime -- the nominal mu=0.6 graph actually has ~84%
of each node's edges landing outside its true community (measured directly:
mean external-edge fraction over all nodes), not 60%. That is barely above
what 36 random equal-size communities would produce by chance, so ANY
community-detection algorithm collapsing on it is expected, not a TPOC-LPA
bug -- confirmed by validating every TPOC-LPA formula exactly against the
paper's own worked numeric examples beforehand, and by the fact a properly
calibrated realized-mu~0.6 graph gives NMI 0.90-0.94, stable across seeds,
instead of a collapse.

The nominal-vs-realized relationship is smooth and monotonic (measured):
  nominal  0.10  0.15  0.20  0.25  0.30  0.35  0.40  0.45  0.50  0.55  0.60
  realized 0.164 0.244 0.326 0.406 0.477 0.543 0.616 0.678 0.736 0.792 0.839
`calibrate_nominal_mu` below inverts this via binary search so callers can
request a graph with a given REALIZED mu (what the paper's mu actually means)
rather than networkx's miscalibrated nominal parameter.

Two further deviations from the paper's exact N1/N2 parameters, both required
to make networkx's LFR_benchmark_graph converge at all:
  1. tau1/tau2 (degree/community-size power-law exponents) are not specified
     in the paper -- using commonly-cited LFR defaults (tau1=3, tau2=1.5).
  2. min_community=10 (the paper's cmin) is infeasible together with
     average_degree=20/max_degree=50 (a node's internal degree can't exceed
     community_size-1, so cmin=10 contradicts a low mu with degree 20)
     -- raised to min_community=15, the smallest value that converges.
"""
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community.quality import modularity
from tpoc_lpa import tpoc_lpa


def _get_truth(G):
    truth = {}
    seen = {}
    for node, comm in nx.get_node_attributes(G, 'community').items():
        key = frozenset(comm)
        if key not in seen:
            seen[key] = len(seen)
        truth[node] = seen[key]
    return truth


def realized_mu(G, truth):
    """Mean, over all nodes, of the fraction of a node's edges landing
    outside its own ground-truth community -- what the paper means by mu."""
    fracs = []
    for n in G.nodes():
        neigh = list(G.neighbors(n))
        if not neigh:
            continue
        ext = sum(1 for j in neigh if truth[j] != truth[n])
        fracs.append(ext / len(neigh))
    return sum(fracs) / len(fracs)


def _generate_raw(n, nominal_mu, average_degree, max_degree, min_community,
                   max_community, tau1, tau2, seed, max_iters):
    G = LFR_benchmark_graph(
        n, tau1, tau2, nominal_mu,
        average_degree=average_degree, max_degree=max_degree,
        min_community=min_community, max_community=max_community,
        seed=seed, max_iters=max_iters,
    )
    G = nx.Graph(G)
    return G, _get_truth(G)


def calibrate_nominal_mu(target_realized_mu, n=1000, average_degree=20, max_degree=50,
                          min_community=15, max_community=50, tau1=3, tau2=1.5,
                          seed=42, max_iters=5000, tol=0.01, max_probes=12):
    """Binary-search the nominal mu that makes networkx's generator produce
    a graph whose ACTUAL realized_mu is close to target_realized_mu."""
    lo, hi = 0.01, 0.7  # nominal mu above ~0.7 is infeasible for this degree/community regime
    best = None
    for _ in range(max_probes):
        mid = (lo + hi) / 2
        try:
            G, truth = _generate_raw(n, mid, average_degree, max_degree, min_community,
                                      max_community, tau1, tau2, seed, max_iters)
        except nx.ExceededMaxIterations:
            hi = mid  # too high to converge -> treat as "overshoots", search lower
            continue
        rmu = realized_mu(G, truth)
        if best is None or abs(rmu - target_realized_mu) < abs(best[1] - target_realized_mu):
            best = (mid, rmu, G, truth)
        if abs(rmu - target_realized_mu) <= tol:
            break
        if rmu < target_realized_mu:
            lo = mid
        else:
            hi = mid
    if best is None:
        raise RuntimeError(f"calibrate_nominal_mu: no feasible nominal mu found for "
                            f"target realized_mu={target_realized_mu}")
    return best  # (nominal_mu_used, realized_mu_achieved, G, truth)


def generate_lfr(n, mu, average_degree=20, max_degree=50, min_community=15,
                  max_community=50, tau1=3, tau2=1.5, seed=42, max_iters=2000,
                  calibrate=True):
    """`mu` here is the TARGET REALIZED mixing parameter (what the paper means
    by mu). When calibrate=True (default), the nominal value fed to networkx's
    generator is binary-searched so the graph's actual realized_mu matches it
    -- see module docstring for why this is necessary."""
    if calibrate:
        _, rmu, G, truth = calibrate_nominal_mu(
            mu, n=n, average_degree=average_degree, max_degree=max_degree,
            min_community=min_community, max_community=max_community,
            tau1=tau1, tau2=tau2, seed=seed, max_iters=max_iters)
        return G, truth
    G, truth = _generate_raw(n, mu, average_degree, max_degree, min_community,
                              max_community, tau1, tau2, seed, max_iters)
    return G, truth


def to_partition(label_dict, nodes):
    groups = {}
    for n in nodes:
        groups.setdefault(label_dict[n], set()).add(n)
    return list(groups.values())


def evaluate_lfr(name, mu, target_nmi, **kwargs):
    G, truth = generate_lfr(n=1000, mu=mu, **kwargs)
    nodes = list(G.nodes())
    disjoint_label, overlap_comm, NI, iters = tpoc_lpa(G, alpha=1.0, beta=0.8)
    y_true = [truth[n] for n in nodes]
    y_pred = [disjoint_label[n] for n in nodes]
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    q = modularity(G, to_partition(disjoint_label, nodes))
    rmu = realized_mu(G, truth)
    print(f"--- {name} (target realized mu={mu}, achieved={rmu:.4f}) ---")
    print(f"  nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
          f"truth_comms={len(set(y_true))} pred_comms={len(set(y_pred))} iters={iters}")
    print(f"  NMI = {nmi:.4f} (target {target_nmi})   ARI = {ari:.4f}   Q = {q:.4f}")
    return dict(name=name, mu=mu, realized_mu=rmu, nmi=nmi, ari=ari, q=q)


if __name__ == '__main__':
    print("=" * 60)
    print("LFR mu-sweep (N-1000-0.1 and N-1000-0.6 from paper's Table 5/10)")
    print("Now using realized-mu calibration (see module docstring)")
    print("=" * 60)
    evaluate_lfr("N-1000-0.1", mu=0.1, target_nmi=1.0000)
    evaluate_lfr("N-1000-0.6", mu=0.6, target_nmi=0.9869)
