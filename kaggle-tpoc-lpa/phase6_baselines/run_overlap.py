"""Table 13-15 analogue: overlapping LFR R1-R6 (paper Table 7), om = 2..8. SLPA (own implementation, r scanned over the paper's
synthetic grid 0.01..0.10 and the best r per graph kept by LFK-NMI -- the paper also tunes r, which favours the baseline) vs the
paper algorithm (our reproduction) and our best pipeline, both with the paper's WAS Stage 3 (labels cached by phase5_overlap/lfrov_run.py).
usage: python3 run_overlap.py <R1..R6> <seed>  -> results/overlap_<R>_s<seed>.json"""
import sys, json, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import *
from lfrov_gen import gen
from lfrov_run import run
R, seed = sys.argv[1], int(sys.argv[2])
RS = [round(0.01 * i, 2) for i in range(1, 11)]
out = {}
for om in range(2, 9):
    G, truth = gen(R, om, seed); t0 = time.time()
    row = {}
    for alg, conf in (("TPOC-LPA (repro)", "paper"), ("Ours", "best")):
        _, ov, _ = run(R, om, seed, conf); row[alg] = overlap_scores(G, truth, ov)
    mem = slpa_memory(G, 100, seed)
    best = None
    for r in RS:
        sc = overlap_scores(G, truth, slpa_cover(G, mem, r)); sc["r"] = r
        if best is None or sc["LFK"] > best["LFK"]: best = sc
    row["SLPA"] = best; out[om] = row
    print(R, f"s{seed} om{om}", {a: (round(v["LFK"], 3), round(v["NMImax"], 3), round(v["Omega"], 3)) for a, v in row.items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/overlap_{R}_s{seed}.json"), "w"), indent=1)
