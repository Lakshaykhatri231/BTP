import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from lfrov_gen import gen
from lfrov_run import run, score
from ovrules import rules2 as rules
seed = int(sys.argv[1]); rs = sys.argv[2].split(","); conf = sys.argv[3]
acc = defaultdict(lambda: [0, 0, 0, 0, 0])
for r in rs:
    for om in range(2, 9):
        G, truth = gen(r, om, seed); L = run(r, om, seed, conf)[0]
        for name, a in rules(G, L):
            s = score(G, truth, a); x = acc[(r, name)]
            for q in range(4): x[q] += s[q] / 7
for (r, name), x in acc.items():
    print(f"{r} {conf:<7} {name:<9} NMI {x[0]:.4f} NMImax {x[1]:.4f} F1 {x[2]:.4f} flag {x[3]:.3f}")
