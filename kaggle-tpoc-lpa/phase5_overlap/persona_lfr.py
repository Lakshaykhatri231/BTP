import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from lfrov_gen import gen
from lfrov_run import score, CONF, run
from persona import persona_cover
sys.path.insert(0, '/Users/lakshaykhatri/Downloads/BTP/kaggle-tpoc-lpa')
from tpoc_lpa import community_label_filtering, node_importance
seed = int(sys.argv[1]); rs = sys.argv[2].split(","); conf = sys.argv[3]; s = int(sys.argv[4])
for r in rs:
    acc = [0] * 4
    for om in range(2, 9):
        G, truth = gen(r, om, seed); t0 = time.time()
        L0 = run(r, om, seed, conf)[0] if 'lab' in sys.argv[5:] else None
        a, P = persona_cover(G, s, L0=L0, **CONF[conf])
        if len(sys.argv) > 5:
            Lm = persona_cover.main; w = community_label_filtering(G, node_importance(G), Lm, 0.8)
            a = {v: a[v] | w[v] for v in G}
        x = score(G, truth, a)
        acc = [u + v / 7 for u, v in zip(acc, x)]
        print(f"{r} om{om} persona s={s} {conf} NMI {x[0]:.4f} NMImax {x[1]:.4f} F1 {x[2]:.4f} flag {x[3]:.3f} personas {P.number_of_nodes()} ({time.time()-t0:.0f}s)", flush=True)
    print(f"{r} AVG persona s={s} {conf} NMI {acc[0]:.4f} NMImax {acc[1]:.4f} F1 {acc[2]:.4f} flag {acc[3]:.3f}", flush=True)
