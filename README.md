# SN-LPA: Size Normalized Label Propagation Algorithm

This repository contains the implementation of **SN-LPA**, a label-propagation algorithm for disjoint and overlapping community
detection. It builds on the three-stage TPOC-LPA framework:

> Kong, Huo, Wang, Guo. *Three-stage overlapping community detection algorithm based on label propagation.*
> Expert Systems With Applications 317, 131873, 2026.

TPOC-LPA scores candidate communities in Stage 2 by summing neighbour attractiveness, with no correction for community size.
As a result, large communities attract more nodes simply because they are large. On dense graphs Stage 2 then collapses into a single
community, and on sparse graphs it leaves many fragments. SN-LPA replaces this with a size-normalised gain and adds resolution-aware
refinement steps: a granularity rule, a map-equation-stopped merge pass, and Synwalk boundary moves.

## Method

SN-LPA keeps the three-stage structure of TPOC-LPA:

1. **Pre-selection.** Seed communities around the nodes with the highest importance (NI), grown with a tightness-sparsity gate (CTS).
2. **Evolution.** Each node joins the community with the highest gain until labels settle.
3. **Label filtering.** A weighted affiliation score (WAS, β = 0.8) lets a node keep several labels, which produces overlap.

Stage 2 uses the null-model gain

```
gain(c, i) = Σ_{j ∈ N(i) ∩ c} A(i, j)  −  γ · s_i · S_c / W
```

Here A(i,j) is the NI-weighted attractiveness, s_i is the strength of node i, S_c is the total strength of community c, and W is
the total edge weight. With γ = 0 the gain is the original TPOC-LPA rule. γ = 1 subtracts the attraction expected by chance
(as in LPAm, Barber & Clark 2009).

The full pipeline consists of six modules (M1–M6):

| Module | Option | Description | Based on |
|---|---|---|---|
| M1 | `sim="tri_overlap"` | Triangle-based edge similarity `c/(1+c)` instead of Jaccard | triangle-weighted label propagation |
| M2 | `guard=True` | Return the Stage 2 iteration with the highest modularity | — |
| M3 | `gamma=1.0` | Size-normalised Stage 2 gain | Barber & Clark 2009 |
| M4 | `refine="granularity"` | Run γ = 0 and γ = 1, keep γ = 1 unless it gives more than 1.5× as many communities | — |
| M5 | `merge=True, merge_criterion="mapeq"` | Greedy community merging, stopped at the minimum map-equation codelength | Kawamoto & Rosvall 2015 |
| M6 | `moves="synwalk"` | Boundary-node moves scored by the Synwalk objective, accepted under a regularised map-equation gate | Toth et al. 2022; Smiljanić et al. 2021 |

All options default to off, so `tpoc_lpa(G)` runs the original TPOC-LPA. Additional variants are also available:
- `moves=True`: map-equation boundary moves.
- `merge_min_gain=0.03`: ΔQ-gated merging.
- `overlap="near_tie"`: an alternative Stage 3.

## Results

NMI, with all methods run on the same graphs using the same metric code (see `kaggle-tpoc-lpa/phase6_baselines/RESULTS.md`).

**Real networks (NMI)**

| Network | Fastgreedy | LPA | Louvain | Walktrap | Infomap | TPOC-LPA | **SN-LPA** |
|---|---|---|---|---|---|---|---|
| Karate | 0.5646 | 0.5254 | 0.5244 | 0.4899 | 0.5805 | **0.8372** | **0.8372** |
| Dolphin | 0.7026 | 0.8139 | 0.8297 | 0.6324 | 0.8993 | 0.9053 | **0.9495** |
| Football | 0.6977 | 0.8666 | 0.8820 | 0.8874 | 0.8683 | 0.9095 | **0.9242** |
| Polbooks | 0.5308 | 0.5495 | 0.5499 | 0.5427 | 0.5437 | 0.5156 | **0.5672** |
| Polblogs | 0.6541 | 0.7094 | 0.6379 | 0.6441 | 0.5270 | 0.7094 | **0.7107** |
| email-Eu-core | 0.4417 | 0.0122 | 0.5806 | 0.5685 | 0.5873 | 0.0000 | **0.6677** |
| Cora | **0.4647** | 0.4328 | 0.4631 | 0.4427 | 0.4166 | 0.4192 | 0.4192 |
| Citeseer | 0.3421 | 0.3405 | 0.3300 | **0.3541** | 0.3386 | 0.3465 | 0.3438 |
| SNAP-Amazon | 0.9820 | 0.9610 | **0.9825** | 0.9774 | 0.9710 | 0.9586 | 0.9731 |
| SNAP-Youtube | 0.5212 | 0.5040 | 0.5410 | 0.7222 | **0.7486** | 0.5195 | 0.5745 |
| SNAP-DBLP | 0.6040 | 0.7657 | 0.6068 | 0.7497 | 0.7631 | 0.7626 | **0.7676** |
| **Average** | 0.5914 | 0.5892 | 0.6298 | 0.6374 | 0.6585 | 0.6258 | **0.7031** |
| **Mean rank** | 4.64 | 4.73 | 4.09 | 4.09 | 4.18 | 4.18 | **2.00** |

**LFR benchmarks (NMI, mean over 5 seeds)**

| Setting | Fastgreedy | LPA | Louvain | Walktrap | Infomap | TPOC-LPA | **SN-LPA** |
|---|---|---|---|---|---|---|---|
| N1 (n=1000, μ=0.1) | 0.8098 | 0.9880 | 0.9872 | 0.9969 | 0.9951 | 0.9942 | **0.9995** |
| N2 (n=1000, μ=0.6) | 0.4749 | 0.4245 | 0.9184 | 0.9560 | 0.9637 | 0.9317 | **0.9815** |
| N3 (n=5000, μ=0.1) | 0.9013 | **1.0000** | 0.9956 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| N4 (n=5000, μ=0.6) | 0.3879 | 0.9828 | 0.9110 | 0.9598 | **0.9945** | 0.9812 | 0.9940 |
| N5 (n=10000, μ=0.1) | 0.8921 | **1.0000** | 0.9864 | **1.0000** | 0.9999 | **1.0000** | **1.0000** |
| N6 (n=10000, μ=0.6) | 0.3825 | 0.9773 | 0.9101 | 0.9178 | **0.9900** | 0.9758 | 0.9895 |
| **Average** | 0.6414 | 0.8954 | 0.9515 | 0.9718 | 0.9905 | 0.9805 | **0.9941** |

TPOC-LPA columns are our re-implementation; the original paper reports Karate 1.0000, Dolphin 0.9053, Football 0.9146 and N2 0.9869
on its own graphs. Stochastic baselines are averaged over 10 runs (3 on SNAP and LFR).

On overlapping LFR graphs SN-LPA matches or beats SLPA on families R1–R4.

**Limitations.**
- Infomap is clearly better on SNAP-Youtube (0.7486 vs 0.5745).
- SLPA is better on overlap families R5–R6.
- M2 and M4 cost NMI on Coauthor-Physics.
- The boundary moves lose up to 0.008 NMI on some sparse citation and co-purchase graphs.

## Repository structure

```
kaggle-tpoc-lpa/
  tpoc_lpa.py               core algorithm (TPOC-LPA + all SN-LPA modules)
  datasets.py, load_snap.py dataset loaders
  lfr_sweep.py              LFR generator with realized-μ calibration
  eq_metric.py              extended modularity (EQ)
  evaluate_phase1.py        reproduction check on Karate / Football
  test_stage3.py            Stage 3 unit test against the worked example of the original paper
  phase3_*.py … phase5_*.py ablation studies
  phase5_overlap/           overlapping LFR benchmark (C++ generator included)
  phase6_baselines/         comparison with Fastgreedy, LPA, Louvain, Walktrap, Infomap, SLPA
  kernel_final/             Kaggle notebook with the comparison tables
paper/                      LaTeX source of the paper (template.tex → build_paper.py → main.tex)
datasets/MANIFEST.json      list of all graphs used
```

## Installation

Requires Python 3.12.

```
pip install networkx numpy scipy scikit-learn
pip install python-igraph infomap        # only for the baseline comparison
```

To build the overlapping LFR generator:

```
cd kaggle-tpoc-lpa/phase5_overlap
g++ -O3 -w -o lfrbench lfr_overlap_generator/benchm.cpp
```

## Usage

```python
import networkx as nx
from tpoc_lpa import tpoc_lpa

G = nx.karate_club_graph()

# original TPOC-LPA
labels, overlap, NI, iters = tpoc_lpa(G)

# SN-LPA
labels, overlap, NI, iters = tpoc_lpa(
    G, sim="tri_overlap", guard=True, refine="granularity",
    merge=True, merge_criterion="mapeq", moves="synwalk")
```

`tpoc_lpa` returns the disjoint labels, the overlapping communities, the node importance scores and the number of iterations.
Ties are broken deterministically. Pass `rng=random.Random(seed)` for random tie-breaking.

Reproducing the experiments (run from `kaggle-tpoc-lpa/`):

```
python3 evaluate_phase1.py                    # TPOC-LPA on Karate and Football
python3 test_stage3.py                        # Stage 3 test
python3 phase4_vs_paper.py real football      # one real network
python3 phase4_vs_paper.py lfr 1000 0.6 1     # one LFR graph (n, mu, seed)
python3 phase6_baselines/run_real.py          # baseline comparison
```

## Data

The datasets are not included because of their size. `datasets/MANIFEST.json` lists every file.
- **Public networks:** Karate, Dolphin, Football, Les Misérables, Polbooks, Polblogs, email-Eu-core, Cora, Citeseer, the SNAP Amazon/DBLP/Youtube
  communities, and Amazon/Coauthor/Cora-ML/Pubmed (Shchur et al. 2018; Bojchevski & Günnemann 2018). Place them in `kaggle-tpoc-lpa/data/`,
  or set `TPOC_DATA` to another directory.
- **LFR graphs:** generated with `lfr_sweep.generate_lfr(n, mu, seed)`.

## Notes

- Karate has edge weights in networkx, so the modularity printed by `evaluate_phase1.py` (0.4036) is weighted. The unweighted value is 0.3715.
- The SNAP subsets used in the original paper are not published, so SNAP results here use the full ground-truth community subgraphs.
- The baselines NSA, SimCMR, WLC, UMSTMO, GREESE and MDPA have no public implementation and are not re-run.
