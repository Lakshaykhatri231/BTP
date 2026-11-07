"""Candidate Stage 3 rules, scored offline from cached disjoint labels."""
import math, sys
from collections import defaultdict
sys.path.insert(0, "/Users/lakshaykhatri/Downloads/BTP/kaggle-tpoc-lpa")
from tpoc_lpa import overlap_near_tie, community_label_filtering, node_importance
from scipy.stats import binom

def links(G, label, i):
    k = defaultdict(int)
    for j in G.neighbors(i): k[label[j]] += 1
    return k

def signif(G, label, alpha=0.01, minlinks=2):
    """OSLOM-style: i also joins neighbour community c if its links into c are improbable
    under the configuration null (Binomial(k_i, vol_c/2m) upper tail < alpha / #candidate comms)."""
    vol = defaultdict(int)
    for v in G: vol[label[v]] += G.degree(v)
    m2 = 2 * G.number_of_edges()
    out = {i: {label[i]} for i in G}
    for i in G:
        k = links(G, label, i); ki = G.degree(i)
        cand = [c for c in k if c != label[i] and k[c] >= minlinks]
        for c in cand:
            p = (vol[c]) / m2
            if binom.sf(k[c] - 1, ki, p) < alpha / max(1, len(k) - 1):
                out[i].add(c)
    return out

def near_tie_ni(G, label, NI, tau=0.5):
    out = {i: {label[i]} for i in G}
    for i in G:
        w = defaultdict(float); k = defaultdict(int)
        for j in G.neighbors(i): w[label[j]] += NI[j]; k[label[j]] += 1
        own = w.get(label[i], 0.0)
        for c in w:
            if c != label[i] and k[c] >= 2 and w[c] >= tau * own: out[i].add(c)
    return out

def rules(G, label, NI=None):
    NI = NI or node_importance(G)
    yield "WAS0.8", community_label_filtering(G, NI, label, 0.8)
    yield "disjoint", {v: {label[v]} for v in G}
    for t in (0.5, 0.75): yield f"tie{t}", overlap_near_tie(G, label, t)
    yield "tieNI0.5", near_tie_ni(G, label, NI, 0.5)
    for a in (0.05, 0.01, 0.001): yield f"sig{a}", signif(G, label, a)

def was_recount(G, NI, label, cover, beta=0.8, own_keep=True):
    """One more WAS round where each neighbour votes with every label in its current cover."""
    out = {}
    for i in G:
        w = defaultdict(float); denom = 0.0
        for j in G.neighbors(i):
            for c in cover[j]: w[c] += NI[j]
            denom += NI[j]
        out[i] = {label[i]}
        if len(w) > 1:
            avg = sum(w.values()) / len(w)
            out[i] |= {c for c in w if w[c] >= beta * avg}
    return out

def rules2(G, label, NI=None):
    NI = NI or node_importance(G)
    was = community_label_filtering(G, NI, label, 0.8)
    yield "WAS0.8", was
    yield "WASrc", was_recount(G, NI, label, was)
    tie = overlap_near_tie(G, label, 0.5)
    yield "tie0.5", tie
    yield "tie-rc", was_recount(G, NI, label, tie)
    yield "sig0.05", signif(G, label, 0.05)
