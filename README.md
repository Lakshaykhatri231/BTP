# Improving TPOC-LPA: label-propagation community detection

B.Tech project (NSUT). It reproduces and analyses the base paper:

> Kong, Huo, Wang, Guo (2026). *Three-stage overlapping community detection algorithm based on label propagation* (TPOC-LPA).
> Expert Systems With Applications 317, 131873.

**Status in one line:** the paper is reproduced (most published numbers matched). I found a real defect in its Stage 2 and added
**opt-in** fixes. They help a lot on some graph types and hurt a little on others. **No change helps on every graph type**, so
the defaults stay exactly the paper's algorithm.

## The algorithm and the defect

TPOC-LPA has three stages:
1. **Pre-selection.** Seed communities around the nodes with the highest importance (NI), grown with a tightness-sparsity gate (CTS).
2. **Evolution.** Each node takes the community with the highest attractiveness until the labels settle.
3. **Label filtering.** A weighted affiliation score (WAS, β = 0.8) lets a node keep several labels (overlap).

**Defect:** Stage 2 (Eq. 12) adds up neighbour attractiveness with no size correction, so big communities keep getting bigger.
On dense graphs everything collapses into one community (email-Eu: NMI 0.54 → 0.00; LFR μ=0.7: 0.05). On sparse graphs it
leaves too many communities (Cora: 214 communities vs 7 classes). Eq. 12 is the γ = 0 end of the null-model family
`Σ A(i,j) − γ·s_i·S_c/W` (γ = 1 is LPAm, Barber & Clark 2009).

## Changes (all off by default)

| # | Option | What it does | Source |
|---|---|---|---|
| 1 | `sim="tri_overlap"` | Triangle-based edge similarity `c/(1+c)` replaces Jaccard | triangle-weighted LP literature |
| 2 | `guard=True` | Return the Stage-2 round with the best modularity, not the last round | own |
| 3 | `gamma=1.0` | Size-corrected Stage 2 (subtracts the chance term) | Barber & Clark 2009 |
| 4 | `refine="granularity"` | Run γ=0 and γ=1. Keep γ=1 only if it gives ≤ 1.5× as many communities | own |
| 5 | `merge=True, merge_min_gain=0.03` | Greedy ΔQ merge pass, accepted only if total ΔQ ≥ 0.03 | Liu & Murata 2010 |
| 6 | `merge_criterion="mapeq"` | Stop merging at the minimum map-equation codelength | Kawamoto & Rosvall 2015 |
| 8 | `moves=True` | Map-equation boundary moves, kept only if the regularized codelength also falls | Infomap; Smiljanić 2020 |
| 9 | `moves="synwalk"` | The same boundary moves, scored with the Synwalk objective | Toth, Helic & Geiger 2022 |
| 11 | `overlap="near_tie"` | Alternative Stage 3 (relative COPRA-style belonging) | Gregory 2010 |

Recommended setting: `sim="tri_overlap", guard=True, refine="granularity", merge=True, merge_criterion="mapeq", moves="synwalk"`.

## Headline results (NMI)

| Benchmark | Paper | Our reproduction | Best of our changes |
|---|---|---|---|
| Karate | 1.0000 | 0.8372 | 0.8372 (one node; the paper's own plain-LPA baseline also gets 0.8372) |
| Dolphin | 0.9053 | 0.9053 | 0.9495 |
| Football | 0.9146 | 0.9095 | 0.9269 |
| LFR n=1000 μ=0.6 (15 seeds) | 0.9869 | 0.9311 | 0.9830 |
| LFR n=10000 μ=0.6 (3 seeds) | 0.9935 | 0.9879 | 0.9967 |
| email-Eu (not in the paper) | — | 0.0000 | 0.67 |
| LFR n=1000 μ=0.7 (not in the paper) | — | 0.0513 | 0.89 |

Held-out set (72 synthetic graphs never used for design): mean NMI 0.7814 (Phase 4) → 0.8069 (best pipeline).
Results got worse on Coauthor-Physics, SNAP-DBLP (ΔQ merge) and a few sparse real graphs (≤ 0.008 with boundary moves).
Every regression is listed in `kaggle-tpoc-lpa/phase6_baselines/RESULTS.md`.

## Repository layout

```
kaggle-tpoc-lpa/          all code (run commands from here)
  tpoc_lpa.py             core algorithm; edits marked ">>> CHANGE k", paper code marked "# PAPER"
  datasets.py, load_snap.py, lfr_sweep.py, eq_metric.py, dolphin_ground_truth.py
  evaluate_phase1.py, test_stage3.py         Phase 1: reproduction gate
  stage4.py, phase2_*.py, kernel*/           Phase 2: LLM Stage 4 (failed, closed) + Kaggle notebooks/logs
  phase3_*.py, phase4_*.py, phase5_*.py      ablations and paper comparisons
  phase5_overlap/         overlapping-LFR benchmark, persona Stage 3 (in progress)
  phase6_baselines/       paper's baselines vs our pipeline (RESULTS.md, results/*.json)
  kernel_final/           teacher-facing Kaggle notebook (Phase 1 vs final)
paper/                    IEEE paper: edit template.tex, then `python3 build_paper.py`
btp_update_deck/          supervisor progress deck (HTML slides) + mid-sem report
datasets/MANIFEST.json    list of every graph used (the data itself is not in git)
```

## Setup

Python 3.12. Packages:

```
pip install networkx==3.7 numpy scipy scikit-learn          # core
pip install python-igraph infomap                           # phase6_baselines only
pip install torch transformers bitsandbytes                 # Phase 2 LLM notebooks only (Kaggle GPU)
```

The overlapping-LFR generator (Phase 5) is C++:
`cd kaggle-tpoc-lpa/phase5_overlap && g++ -O3 -w -o lfrbench lfr_overlap_generator/benchm.cpp`.
The paper compiles with `tectonic paper/main.tex`.

### Data

The graphs (~850 MB) are **not in the repository**. `datasets/MANIFEST.json` lists every file. Rebuild them like this:
- Karate, Dolphin, Football, Lesmis, Polbooks, Polblogs, email-Eu, Cora, Citeseer, the SNAP graphs and the npz graphs (Amazon/Coauthor, Cora-ML, Pubmed, DBLP)
  are public. Download them into `kaggle-tpoc-lpa/data/` (sources: Newman's network data, SNAP, LINQS, Shchur et al. 2018, Bojchevski & Günnemann 2018).
- LFR graphs: generated by `lfr_sweep.generate_lfr(n, mu, seed)` (calibrated to the realized μ).
- `python3 build_dataset_bundle.py` builds the `datasets/` bundle; `python3 verify_bundle.py` checks it.
- To use the bundle: `export TPOC_DATA=/path/to/BTP/datasets`.

## Quick start

```
cd kaggle-tpoc-lpa
python3 evaluate_phase1.py      # must print Karate 0.8372 / Q 0.4036, Football 0.9095 / Q 0.6032
python3 test_stage3.py          # Stage 3 vs the paper's Table 2 example: ALL PASS
python3 phase4_vs_paper.py real football
python3 phase4_vs_paper.py lfr 1000 0.6 1
```

```python
import networkx as nx
from tpoc_lpa import tpoc_lpa

G = nx.karate_club_graph()
labels, overlap, NI, iters = tpoc_lpa(G)                       # the paper's algorithm
labels, overlap, NI, iters = tpoc_lpa(G, sim="tri_overlap", guard=True, refine="granularity",
                                      merge=True, merge_criterion="mapeq", moves="synwalk")
```

`rng=None` breaks ties deterministically. To get the paper's random tie-breaking, pass `rng=random.Random(seed)`.

Note: `nx.karate_club_graph()` has edge weights, so the Karate Q of 0.4036 is weighted modularity. Unweighted, it is 0.3715, the paper's value.

## Not included / known gaps

- Karate NMI 1.0 is not reachable (one node; 300 random-tie runs all give 0.8372).
- The paper's exact Amazon/DBLP/Youtube subsamples could not be reconstructed, so our SNAP numbers use larger graphs.
- Baselines with no public code (NSA, SimCMR, WLC, UMSTMO, GREESE, MDPA): only the paper's published numbers are quoted.
- Reference PDFs are not committed (copyright). Citations are in `paper/main.tex`.
