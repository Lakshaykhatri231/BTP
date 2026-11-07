"""Precision of extra memberships: (i, c) correct iff i belongs to the true community best matching c."""
import sys, os
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovrules import links

def match(label, truth):
    mem = defaultdict(list)
    for t, T in enumerate(truth):
        for v in T: mem[v].append(t)
    comm = defaultdict(set)
    for v, c in label.items(): comm[c].add(v)
    best = {}
    for c, S in comm.items():
        cnt = Counter(t for v in S for t in mem.get(v, ()))
        best[c] = max(cnt, key=lambda t: cnt[t] / len(S | truth[t])) if cnt else None
    return best, mem

def prec(G, label, truth, assign):
    best, mem = match(label, truth)
    ok = tot = 0; byk = defaultdict(lambda: [0, 0])
    for i, cs in assign.items():
        k = links(G, label, i)
        for c in cs:
            if c == label[i]: continue
            good = best[c] is not None and best[c] in mem.get(i, ())
            ok += good; tot += 1
            b = byk[min(k[c], 4)]; b[0] += good; b[1] += 1
    return ok / max(tot, 1), tot, {kk: f"{v[0]}/{v[1]}" for kk, v in sorted(byk.items())}
