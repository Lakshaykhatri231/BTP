import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/lakshaykhatri/Downloads/BTP/kaggle-tpoc-lpa')
from lfrov_gen import gen
from lfrov_run import score, CONF, run
from persona import persona_cover
from tpoc_lpa import community_label_filtering, node_importance
seed = int(sys.argv[1]); rs = sys.argv[2].split(",")
for r in rs:
    for om in range(2, 9):
        G, truth = gen(r, om, seed); out = []
        for c in ("paper", "best"):
            L, ov, _ = run(r, om, seed, c); out.append((c, score(G, truth, ov)))
        a, P = persona_cover(G, 3, **CONF["best"])
        w = community_label_filtering(G, node_importance(G), persona_cover.main, 0.8)
        out.append(("persona", score(G, truth, {v: a[v] | w[v] for v in G})))
        print(r, f"om{om}", " ".join(f"{c} {x[0]:.4f} {x[1]:.4f} {x[3]:.3f}" for c, x in out), flush=True)
