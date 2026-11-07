"""
EQ (extended modularity for overlapping communities), Eq. 21 of the paper:
  EQ = (1/2m) * sum_c sum_{u,w in c} (1/(O_u*O_w)) * (A_uw - d_u*d_w/2m)
where O_u = number of communities node u belongs to, sum over (u,w) pairs
(including u=w) within each community c, analogous to how standard
modularity Q sums over all (i,j) pairs with delta(c_i,c_j)=1.

Sanity check: reduces exactly to standard Q when no node is in >1 community.
"""
import networkx as nx


def eq_score(G: nx.Graph, communities: dict) -> float:
    """communities: dict node -> set of community ids (Stage 3 output)."""
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    degree = dict(G.degree())
    O = {n: len(communities[n]) for n in G.nodes()}

    # community -> members
    comm_members = {}
    for n, cs in communities.items():
        for c in cs:
            comm_members.setdefault(c, set()).add(n)

    adj = {n: set(G.neighbors(n)) for n in G.nodes()}

    total = 0.0
    for c, members in comm_members.items():
        members = list(members)
        for u in members:
            for w in members:
                a_uw = 1.0 if w in adj[u] else 0.0
                total += (a_uw - (degree[u] * degree[w]) / (2 * m)) / (O[u] * O[w])
    return total / (2 * m)
