"""
Offline evaluation of Stage 4 policies on CACHED LLM evidence.

Input : collect_<model>.json files produced by the GPU evidence-collection
        notebook (per node: the LLM's soft preference `ell` over the node's
        candidate communities, per prompt variant, plus per-permutation raw
        probabilities).
Output: for every (model, dataset, prompt variant):
  1. accuracy of the LLM vs. the structural label, and the agreement matrix
     (both right / structure-only / LLM-only / both wrong) -- the number that
     decides whether overriding structure can EVER help;
  2. position bias (raw prob on letter A relative to uniform) and how well the
     model follows the answer format;
  3. policy sweeps (hard override, product-of-experts fusion) with NMI and
     per-node helped / hurt counts;
  4. CROSS-DATASET transfer: fit the fusion weight on one graph, test on the
     other (fitting and testing on one 100-node graph would prove nothing);
  5. label-free reliability signals: do agreement-with-structure on confident
     nodes, or the model's own confidence, predict where it is right?

"Target" community of a node = the predicted community holding most of its
ground-truth class (majority mapping). LLM/structure are "right" if they pick it.
"""
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

import networkx as nx
from sklearn.metrics import normalized_mutual_info_score

from tpoc_lpa import tpoc_lpa, node_importance
from stage4 import was_distribution, structural_confidence

EPS = 1e-6


# --------------------------------------------------------------------- graphs
def load_graph(name):
    if name == "football":
        G = nx.read_gml("data/football/football.gml", label="id")
        G = nx.relabel_nodes(G, {n: G.nodes[n]["label"] for n in G.nodes()})
        return G, {n: str(G.nodes[n]["value"]) for n in G.nodes()}
    if name == "polbooks":
        G = nx.read_gml("data/polbooks/polbooks.gml")
        return G, {n: str(G.nodes[n]["value"]) for n in G.nodes()}
    if name == "polblogs":
        p = json.load(open("data/polblogs/polblogs_lcc.json"))
        G = nx.Graph()
        G.add_nodes_from(p["labels"])
        G.add_edges_from((p["labels"][a], p["labels"][b]) for a, b in p["edges"])
        return G, {l: str(v) for l, v in zip(p["labels"], p["value"])}
    raise ValueError(name)


class DS:
    """A dataset with its structural baseline and everything derived from it."""

    def __init__(self, name):
        self.name = name
        self.G, self.truth = load_graph(name)
        self.NI = node_importance(self.G)
        self.base, _, _, _ = tpoc_lpa(self.G, alpha=1.0, beta=0.8)
        self.nodes = list(self.G.nodes())
        cnt = defaultdict(Counter)
        for n in self.nodes:
            cnt[self.truth[n]][self.base[n]] += 1
        self.target_comm = {gt: c.most_common(1)[0][0] for gt, c in cnt.items()}
        self.y_true = [self.truth[n] for n in self.nodes]
        self.base_nmi = nmi(self.y_true, [self.base[n] for n in self.nodes])
        # ambiguity ranking exactly as Stage 4 does it
        self.gain = {}
        for n in self.nodes:
            dist = was_distribution(self.G, self.NI, self.base, n)
            if len(dist) >= 2:
                self.gain[n] = max(0.0, 0.95 - structural_confidence(dist))

    def target(self, node):
        return self.target_comm[self.truth[node]]


def nmi(a, b):
    return normalized_mutual_info_score(a, b)


# --------------------------------------------------------------- record views
def view(ds, rec, variant):
    node = rec["node"]
    cands = rec["cands"]
    was_c = [max(rec["was"].get(c, 0.0), 0.0) for c in cands]
    z = sum(was_c) or 1.0
    s = {c: w / z for c, w in zip(cands, was_c)}
    ell = rec["variants"][variant]["ell"]
    return node, cands, s, ell


def ranked_nodes(ds, recs):
    by = {r["node"]: r for r in recs}
    order = sorted((n for n in ds.gain if str(n) in by), key=lambda n: (-ds.gain[n], str(n)))
    return [by[str(n)] for n in order]


def argmax(d):
    return max(d, key=lambda c: (d[c], c))


# ------------------------------------------------------------------- analysis
def agreement(ds, recs, variant):
    """Return counts of (both, struct_only, llm_only, neither) over nodes whose
    target community is among the candidates."""
    both = so = lo = ne = 0
    for r in recs:
        node, cands, s, ell = view(ds, r, variant)
        n = _node(ds, node)
        t = ds.target(n)
        if t not in cands:
            continue
        sr = ds.base[n] == t
        lr = argmax(ell) == t
        both += sr and lr
        so += sr and not lr
        lo += (not sr) and lr
        ne += (not sr) and (not lr)
    return both, so, lo, ne


def _node(ds, name):
    m = getattr(ds, "_by_name", None)
    if m is None:
        m = ds._by_name = {str(n): n for n in ds.nodes}
    return m[name]


def position_bias(recs, variant):
    ratios, mass = [], []
    for r in recs:
        for p in r["variants"][variant]["perms"]:
            k = len(p["probs"])
            ratios.append(p["probs"][0] * k)
            mass.append(p["letter_mass"])
    return sum(ratios) / len(ratios), sum(mass) / len(mass)


def apply_policy(ds, recs, variant, b, B):
    """Relabel the top-B most structurally ambiguous boundary nodes by
    argmax_c [log s(c) + b * log ell(c)] over their candidates. b = large ~ hard override."""
    new = dict(ds.base)
    helped = hurt = changed = 0
    for r in ranked_nodes(ds, recs)[:B]:
        node, cands, s, ell = view(ds, r, variant)
        n = _node(ds, node)
        score = {c: math.log(s[c] + EPS) + b * math.log(ell[c] + EPS) for c in cands}
        choice = argmax(score)
        if choice != new[n]:
            t = ds.target(n)
            before, after = new[n] == t, choice == t
            helped += (not before) and after
            hurt += before and (not after)
            changed += 1
            new[n] = choice
    n_ = nmi(ds.y_true, [new[n] for n in ds.nodes])
    return n_, changed, helped, hurt


def nll(ds, recs, variant, b):
    tot, cnt = 0.0, 0
    for r in recs:
        node, cands, s, ell = view(ds, r, variant)
        t = ds.target(_node(ds, node))
        if t not in cands:
            continue
        sc = {c: math.log(s[c] + EPS) + b * math.log(ell[c] + EPS) for c in cands}
        m = max(sc.values())
        z = sum(math.exp(v - m) for v in sc.values())
        tot += -(sc[t] - m - math.log(z))
        cnt += 1
    return tot / max(cnt, 1)


B_FRACS = (0.10, 0.25, 0.50, 1.00)   # share of boundary nodes queried, most ambiguous first
W_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for p in pos:
        for q in neg:
            tot += 1.0 if p > q else 0.5 if p == q else 0.0
    return tot / (len(pos) * len(neg))


def report_model(model, data, dss):
    print("=" * 96)
    print(f"MODEL: {model}")
    print("=" * 96)
    for dname, blob in data.items():
        ds = dss[dname]
        recs = blob["nodes"]
        mism = sum(1 for n in ds.nodes if blob["base_label"][str(n)] != str(ds.base[n]))
        print(f"\n--- {dname}: {len(recs)} boundary nodes | structural NMI {ds.base_nmi:.4f} "
              f"| base-label mismatches vs Kaggle run: {mism}")
        for variant in sorted(recs[0]["variants"], key=int):
            bias, mass = position_bias(recs, variant)
            both, so, lo, ne = agreement(ds, recs, variant)
            tot = both + so + lo + ne
            print(f"  prompt shows {variant:>2} members/candidate | position-A bias x{bias:.2f} "
                  f"(1.0=none) | letter mass {mass:.3f}")
            print(f"    LLM acc {(both+lo)/tot:.3f} vs structure acc {(both+so)/tot:.3f} (n={tot}) | "
                  f"both right {both}, STRUCT-only {so}, LLM-only {lo}, both wrong {ne} "
                  f"=> net if LLM overrides on disagreements: {lo-so:+d}")
        # restrict the rest to the richer prompt
        variant = sorted(recs[0]["variants"], key=int)[-1]
        print(f"  [prompt variant {variant}] policy sweep (NMI; helped/hurt = nodes moved wrong->right / right->wrong)")
        print(f"    {'policy':<26}" + "".join(f"top {int(f*100):>3}%          " for f in B_FRACS))
        for label, b in [("structure argmax (b=0)", 0.0), ("fusion b=0.5", 0.5), ("fusion b=1", 1.0),
                         ("fusion b=2", 2.0), ("hard override (b=1e3)", 1e3)]:
            row = f"    {label:<26}"
            for f in B_FRACS:
                B = max(1, round(f * len(recs)))
                n_, ch, h, hu = apply_policy(ds, recs, variant, b, B)
                row += f"{n_:.3f} +{h}/-{hu}     "
            print(row)
        # label-free reliability signals
        order = ranked_nodes(ds, recs)
        half = len(order) // 2
        conf_half, amb_half = order[half:], order[:half]   # least ambiguous / most ambiguous
        def acc(rs, ref):
            ok = tot = 0
            for r in rs:
                node, cands, s, ell = view(ds, r, variant)
                n = _node(ds, node)
                t = ds.target(n)
                if ref == "truth" and t not in cands:
                    continue
                target = t if ref == "truth" else ds.base[n]
                ok += argmax(ell) == target
                tot += 1
            return ok / max(tot, 1)
        pseudo = acc(conf_half, "base")      # deployable: agree with structure on CONFIDENT nodes
        true_conf = acc(conf_half, "truth")
        true_amb = acc(amb_half, "truth")    # what actually matters: accuracy on the nodes we'd query
        print(f"  label-free reliability estimate (LLM agrees w/ structure on the confident half): {pseudo:.3f}")
        print(f"    true LLM accuracy: confident half {true_conf:.3f} | AMBIGUOUS half (the nodes queried) {true_amb:.3f}"
              f"  => estimate over-states reliability by {pseudo-true_amb:+.3f}")
        pos, neg = [], []
        for r in recs:
            node, cands, s, ell = view(ds, r, variant)
            t = ds.target(_node(ds, node))
            if t not in cands:
                continue
            (pos if argmax(ell) == t else neg).append(max(ell.values()))
        print(f"  AUROC of the model's own confidence (max prob) as a per-node correctness signal: "
              f"{auroc(pos, neg):.3f}  (0.5 = useless)")


def structural_uncertainty(ds, recs, variant):
    """Does structural ambiguity point at the nodes where structure is WRONG?
    That is the whole premise of spending a query budget on the most ambiguous
    nodes. Reports AUROC and the share of all structural errors captured by the
    top-q% most ambiguous nodes (random selection would capture q%)."""
    order = ranked_nodes(ds, recs)
    err, ok = [], []
    for rank, r in enumerate(order):
        node, cands, s_, ell = view(ds, r, variant)
        n = _node(ds, node)
        t = ds.target(n)
        if t not in cands:
            continue
        (ok if ds.base[n] == t else err).append(len(order) - rank)   # higher = more ambiguous
    tot_err = len(err)
    cap = {}
    for q in (0.10, 0.25, 0.50):
        k = round(q * len(order))
        cutoff = len(order) - k
        cap[q] = sum(1 for v in err if v > cutoff) / max(tot_err, 1)
    return auroc(err, ok), cap, tot_err, len(err) + len(ok)


def fit_b_pooled(train, data, dss, variants):
    """Weight b minimising the AVERAGE (per-dataset, so a big graph cannot
    dominate) NLL of the true community over the training datasets."""
    fits = {}
    for b in W_GRID:
        fits[b] = sum(nll(dss[d], data[d]["nodes"], variants[d], b) for d in train) / len(train)
    return min(fits, key=fits.get), fits


def cross_dataset(model, data, dss):
    names = sorted(data)
    if len(names) < 2:
        return
    variants = {d: sorted(data[d]["nodes"][0]["variants"], key=int)[-1] for d in names}
    print("\n" + "-" * 96)
    print(f"LEAVE-ONE-DATASET-OUT ({model}): fit fusion weight b on the OTHER graphs, apply to the held-out one")
    print("-" * 96)
    print(f"  {'held-out':<10}{'b_hat':>6}  " + "".join(f"{'top '+str(int(f*100))+'%':<24}" for f in (0.25, 1.0))
          + "baseline NMI")
    for te in names:
        tr = [d for d in names if d != te]
        b_hat, _ = fit_b_pooled(tr, data, dss, variants)
        row = f"  {te:<10}{b_hat:>6}  "
        for f in (0.25, 1.0):
            B = max(1, round(f * len(data[te]["nodes"])))
            n_, ch, h, hu = apply_policy(dss[te], data[te]["nodes"], variants[te], b_hat, B)
            row += f"{n_:.4f} ({n_-dss[te].base_nmi:+.4f}) +{h}/-{hu}   "
        print(row + f"{dss[te].base_nmi:.4f}")


def structural_report(data, dss):
    print("\n" + "=" * 96)
    print("PREMISE CHECK: does structural ambiguity point at structural errors? (model-independent)")
    print("=" * 96)
    for d in sorted(data):
        recs = data[d]["nodes"]
        variant = sorted(recs[0]["variants"], key=int)[-1]
        au, cap, n_err, n_tot = structural_uncertainty(dss[d], recs, variant)
        print(f"  {d:<9} structural errors {n_err}/{n_tot} boundary nodes | AUROC(ambiguity -> error) {au:.3f} | "
              f"errors captured by top 10%/25%/50% ambiguous: {cap[0.10]:.0%} / {cap[0.25]:.0%} / {cap[0.50]:.0%} "
              f"(random = 10%/25%/50%)")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "kernel_collect/output"
    files = sorted(glob.glob(os.path.join(folder, "collect_*.json")))
    if not files:
        print("no collect_*.json in", folder)
        return
    dss = {n: DS(n) for n in ("football", "polbooks", "polblogs")}
    first = True
    for f in files:
        model = os.path.basename(f)[len("collect_"):-len(".json")]
        data = json.load(open(f))
        data = {d: v for d, v in data.items() if d in dss}
        if first:
            structural_report(data, dss)
            first = False
        report_model(model, data, dss)
        cross_dataset(model, data, dss)


if __name__ == "__main__":
    main()
