"""Readable node positions for small graph drawings.

Starts from several Kamada-Kawai / spring layouts and improves each by simulated annealing on a
cost that penalises (i) nodes closer than `dmin`, (ii) a node lying on or near an edge it is not
part of, (iii) edge crossings, (iv) very small angles between edges at a node, (v) uneven edge
lengths and (vi) other nodes covering an edge label. Positions are in cm inside a W x H box."""
import math, random
import networkx as nx


def _seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _cross(a, b, c, d):
    def o(p, q, r): return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (o(a, b, c) * o(a, b, d) < 0) and (o(c, d, a) * o(c, d, b) < 0)


class Opt:
    def __init__(self, G, W, H, dmin, clear, label_edges=()):
        self.G, self.W, self.H, self.dmin, self.clear = G, W, H, dmin, clear
        self.V = list(G.nodes()); self.E = [tuple(e) for e in G.edges()]
        self.inc = {v: [e for e in self.E if v in e] for v in self.V}
        self.nb = {v: list(G.neighbors(v)) for v in self.V}
        self.lab = [tuple(e) for e in label_edges]
        n = len(self.V)
        self.L0 = 0.95 * math.sqrt(W * H / n) if n else 1.0

    def angle_cost(self, P, u):
        nb = self.nb[u]
        if len(nb) < 2: return 0.0
        angs = sorted(math.atan2(P[x][1] - P[u][1], P[x][0] - P[u][0]) for x in nb)
        c = 0.0
        for i in range(len(angs)):
            d = (angs[(i + 1) % len(angs)] - angs[i]) % (2 * math.pi)
            lim = math.radians(22)
            if d < lim: c += 6.0 * ((lim - d) / lim) ** 2
        return c

    def local(self, P, v):
        c = 0.0; pv = P[v]; dmin, clear = self.dmin, self.clear
        for u in self.V:
            if u is v or u == v: continue
            d = math.hypot(pv[0] - P[u][0], pv[1] - P[u][1])
            if d < dmin: c += 60.0 * ((dmin - d) / dmin) ** 2
        for a, b in self.E:
            if v == a or v == b: continue
            d = _seg_dist(pv, P[a], P[b])
            if d < clear: c += 40.0 * ((clear - d) / clear) ** 2
        for a, b in self.inc[v]:
            for w in self.V:
                if w == a or w == b: continue
                d = _seg_dist(P[w], P[a], P[b])
                if d < clear: c += 40.0 * ((clear - d) / clear) ** 2
            for e, f in self.E:
                if e in (a, b) or f in (a, b): continue
                if _cross(P[a], P[b], P[e], P[f]): c += 5.0
            L = math.hypot(P[a][0] - P[b][0], P[a][1] - P[b][1])
            c += 0.6 * ((L - self.L0) / self.L0) ** 2
        c += self.angle_cost(P, v)
        for u in self.nb[v]: c += self.angle_cost(P, u)
        for a, b in self.lab:   # keep edge labels free
            mx, my = (P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2
            if v in (a, b):
                for w in self.V:
                    if w in (a, b): continue
                    d = math.hypot(P[w][0] - mx, P[w][1] - my)
                    if d < 0.95: c += 40.0 * ((0.95 - d) / 0.95) ** 2
            else:
                d = math.hypot(pv[0] - mx, pv[1] - my)
                if d < 0.95: c += 40.0 * ((0.95 - d) / 0.95) ** 2
        for a, b in self.lab:   # labelled edges need room for their label
            if v in (a, b):
                L = math.hypot(P[a][0] - P[b][0], P[a][1] - P[b][1])
                if L < 1.9: c += 30.0 * ((1.9 - L) / 1.9) ** 2
                mx, my = (P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2
                for e, f in self.E:   # label must not sit on another edge
                    if (e, f) in ((a, b), (b, a)): continue
                    if _seg_dist((mx, my), P[e], P[f]) < 0.3: c += 15.0
        lv = [(a, b) for a, b in self.lab if v in (a, b)]
        for a, b in lv:   # edge labels must not overlap each other
            mx, my = (P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2
            for e, f in self.lab:
                if (e, f) == (a, b): continue
                nx_, ny_ = (P[e][0] + P[f][0]) / 2, (P[e][1] + P[f][1]) / 2
                if abs(mx - nx_) < 1.35 and abs(my - ny_) < 0.5: c += 40.0
        return c

    def total(self, P):
        # every pair term counted twice by summing local(); good enough for comparing starts
        return sum(self.local(P, v) for v in self.V)

    def anneal(self, P, iters, seed):
        rnd = random.Random(seed)
        P = {v: list(p) for v, p in P.items()}
        T0, T1 = 2.0, 0.005
        s0, s1 = 0.35 * max(self.W, self.H), 0.02 * max(self.W, self.H)
        for it in range(iters):
            f = it / iters
            T = T0 * (T1 / T0) ** f; s = s0 * (s1 / s0) ** f
            v = self.V[rnd.randrange(len(self.V))]
            old = P[v][:]; c0 = self.local(P, v)
            P[v][0] = min(max(old[0] + rnd.gauss(0, s), 0.0), self.W)
            P[v][1] = min(max(old[1] + rnd.gauss(0, s), 0.0), self.H)
            d = self.local(P, v) - c0
            if d > 0 and rnd.random() >= math.exp(-d / T):
                P[v] = old
        return {v: tuple(p) for v, p in P.items()}


def optimise(G, W, H, dmin=0.85, clear=0.4, label_edges=(), starts=6, iters=None, seed=0, init=None):
    O = Opt(G, W, H, dmin, clear, label_edges)
    iters = iters or 2500 * len(O.V)
    best, bc = None, float("inf")
    inits = []
    if init is not None: inits.append(init)
    for s in range(starts):
        r = random.Random(seed * 100 + s)
        p0 = {v: (r.random(), r.random()) for v in G}
        inits.append(nx.kamada_kawai_layout(G, pos=p0) if s % 2 == 0 else nx.spring_layout(G, seed=seed * 100 + s, iterations=200))
    for k, p in enumerate(inits):
        xs = [q[0] for q in p.values()]; ys = [q[1] for q in p.values()]
        sx, sy = (max(xs) - min(xs)) or 1, (max(ys) - min(ys)) or 1
        P = {v: ((x - min(xs)) / sx * W, (y - min(ys)) / sy * H) for v, (x, y) in p.items()}
        P = O.anneal(P, iters, seed * 1000 + k)
        c = O.total(P)
        if c < bc: best, bc = P, c
    return best, bc
