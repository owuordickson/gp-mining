# TGRAANK causal-discovery benchmark results

Refs #19. Published from the saved output of [benchmarking.ipynb](./benchmarking.ipynb) on 13 October 2026. The [machine-readable export](./benchmarking-results.json) records the notebook SHA-256, every saved result, the referenced scoring paths, and the current artifact values. This publication exports existing results; it does not represent a new benchmark campaign.

## Research focus and method families

The benchmark asks how TGRAANK, an asymmetric approach to causal discovery, performs across river-network graph structures alongside other causal-discovery families. The main comparison is TGRAANK, PCMCI, PC, VARLiNGAM and DYNOTEARS.

| Method | Family used in this comparison | Basis |
|---|---|---|
| TGRAANK / T-GRAANK | Asymmetric causal discovery, the author's proposed framing | Directional, temporally lagged gradual dependencies with ordered cause/effect roles. |
| PCMCI | Constraint-based | Conditional-independence tests adapted to time series. |
| PC | Constraint-based | Conditional-independence constraints and graph orientation rules. |
| VARLiNGAM | Asymmetry-based | Linear non-Gaussian structural models distinguish causal directions. |
| DYNOTEARS | Score-based | Continuous optimization of a graph objective with structural constraints. |

[Runge et al., *Causal inference for time series* (2023)](https://doi.org/10.1038/s43017-023-00431-y) supplies this taxonomy for the established methods; the supplied manuscript discusses these families on pages 14–15. TGRAANK's classification is the author's framing of the proposed approach, rather than an attribution of TGRAANK to that review. Directional gradual dependencies require empirical validation against causal ground truth; this report establishes no causal-identifiability theorem.

Names are normalized for publication: the notebook's `PCMI` means **PCMCI**, `Varlingam` means **VARLiNGAM**, and `Dynoteurs` means **DYNOTEARS**. PC denotes the constraint-based PC algorithm. The paper's **CP** denotes Causal Pretraining, a different method.

## CausalRivers: complete saved notebook comparison

The metric is **mean per graph AUROC**, named `Individual AUROC` by the harness, with autoregressive diagonal entries excluded. It differs from pooled `AUROC`. TGRAANK values below reproduce the saved notebook display at six decimal places; comparator values reproduce its three-decimal transcription.

| Graph set | TGRAANK | PCMCI | PC | VARLiNGAM | DYNOTEARS |
|---|---:|---:|---:|---:|---:|
| Close 3 | 0.677891 | 0.639 | 0.501 | 0.786 | 0.501 |
| Close 5 | 0.654067 | 0.617 | 0.514 | 0.773 | 0.499 |
| Root cause 3 | 0.700800 | 0.696 | 0.501 | 0.772 | 0.503 |
| Root cause 5 | 0.704923 | 0.737 | 0.517 | 0.769 | 0.564 |
| Random+1 3 | 0.828725 | 0.831 | 0.498 | 0.840 | 0.519 |
| Random+1 5 | 0.734016* | 0.735 | 0.513 | 0.788 | 0.608 |
| Confounder 3 | 0.651331 | 0.663 | 0.490 | 0.675 | 0.525 |
| Confounder 5 | 0.586420 | 0.640 | 0.514 | 0.698 | 0.532 |
| Random 3 | 0.669843 | 0.655 | 0.501 | 0.787 | 0.503 |
| Random 5 | 0.640697 | 0.653 | 0.506 | 0.755 | 0.608 |
| Disjoint 10 | Unavailable | 0.802 | Unavailable | 0.827 | 0.606 |

**Random+1-5 artifact identity.** The saved `0.734016` corresponds to the [preserved replacement's `scoring.csv`](https://github.com/owuordickson/causalrivers/blob/2e970afabd6952e04c3de093ede4df562c974e96/results/tgraank_1_random_5/recovered-replacement-3025328c/scoring.csv): its full value is `0.7340156985733362`. The path named in the notebook now contains the [restored original](https://github.com/owuordickson/causalrivers/blob/2e970afabd6952e04c3de093ede4df562c974e96/results/tgraank_1_random_5/20260921025013/scoring.csv), whose mean per graph AUROC is **0.7342750171305081**, or **0.734275** at the displayed precision. The original has independently reproducible rescoring against reconstructed ordered labels. The saved notebook is retained as an output snapshot; rerunning its scoring-table cell against the current gitlink changes this one displayed value.

**Missing results.** The [TGRAANK Disjoint-10 scoring artifact](https://github.com/owuordickson/causalrivers/blob/2e970afabd6952e04c3de093ede4df562c974e96/results/tgraank_disjoint_10/20260929125251/scoring.csv) contains NaN metrics. This is an unavailable result, not zero or a measured AUROC. PC also has no reported Disjoint-10 value. Ten graph sets have a finite TGRAANK result; the panel retains the eleventh missing entry.

### Comparator provenance and comparison limits

The [CausalRivers website](https://causalrivers.github.io/) links its official [leaderboard CSV at revision `22ffa6bd`](https://github.com/CausalRivers/CausalRivers.github.io/blob/22ffa6bdc7cb3ac6b1947c743677c35a473aa459/scoring.csv). A [local snapshot](./causalrivers-leaderboard.csv) preserves those full-precision values. All 43 available comparator cells in the notebook match the selected leaderboard rows rounded to three decimals.

PCMCI, VARLiNGAM and DYNOTEARS are marked `From publication` in that CSV and appear in Table 3 of [Stein et al., *CausalRivers* (ICLR 2025)](https://openreview.net/forum?id=wmV4cIbgl6), also available as an [author-hosted arXiv PDF](https://arxiv.org/pdf/2503.17452). PC comes from the separate `Vanilla PC method-ElliSchool` community submission dated 4 September 2025, rather than that paper's Table 3. Its root-cause values occupy the CSV's additional `Root Cause 3` and `Root Cause 5` columns.

The paper reports mean sample AUROC for each method's best hyperparameter combination, across multiple resolutions and preprocessing settings. This report combines those published references with archived local TGRAANK scores. The comparator methods were not rerun here, and identical graph order, preprocessing, time windows, search budgets and sample exclusions across this full panel have not been established. These are descriptive comparisons across method families, not a newly controlled ranking, superiority claim or speedup measurement. Original producer source/environment/command metadata for the historical TGRAANK campaign is incomplete.

### TGRAANK scoring paths recorded by the notebook

Paths below are relative to the pinned [CausalRivers child `2e970af`](https://github.com/owuordickson/causalrivers/tree/2e970afabd6952e04c3de093ede4df562c974e96).

| Graph set | Notebook scoring path |
|---|---|
| Close 3 | `results/tgraank_close_3/20260821124601/scoring.csv` |
| Close 5 | `results/tgraank_close_5/20260830185811/scoring.csv` |
| Root cause 3 | `results/tgraank_root_cause_3/20260825173222/scoring.csv` |
| Root cause 5 | `results/tgraank_root_cause_5/20260902024058/scoring.csv` |
| Random+1 3 | `results/tgraank_1_random_3/20260819161921/scoring.csv` |
| Random+1 5 | `results/tgraank_1_random_5/20260921025013/scoring.csv` (restored original; saved output used the replacement) |
| Confounder 3 | `results/tgraank_confounder_3/20260926075557/scoring.csv` |
| Confounder 5 | `results/tgraank_confounder_5/20260827034153/scoring.csv` |
| Random 3 | `results/tgraank_random_3/20260825054633/scoring.csv` |
| Random 5 | `results/tgraank_random_5/20260926033539/scoring.csv` |
| Disjoint 10 | `results/tgraank_disjoint_10/20260929125251/scoring.csv` (NaN) |

## Complete saved synthetic mock results

The notebook calls `generate_timegraph_mock()` and `generate_causalrivers_mock()` and evaluates 2×2 binary adjacency matrices. Both datasets in this section are **synthetic mocks**. The notebook's `CausalRivers (Real-World)` label here does not mean actual river observations were loaded. These precision/recall/F1 values are separate from the real river-network AUROC table above.

| Algorithm | CausalRivers mock precision | Recall | F1 | TimeGraph mock precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Classical Statistics | 0.50 | 1.00 | 0.67 | 0.00 | 0.00 | 0.00 |
| Granger Causality | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PC | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| TGRAANK | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Tigramite / PCMCI | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

These are the saved outputs, not a reproduced mock run. The imported `Synthetic.baseline_methods` and `Synthetic.benchmark` modules are absent from this checkout, and the notebook's original producer environment is not captured. No GRAANK, VARLiNGAM or DYNOTEARS mock result is present in the saved notebook.

## Supplementary reproducibility evidence

The audited 24-graph Confounder-3 pair checks the harness and scoring on identical frozen inputs. It supplements the broader family comparison. Its mean per graph AUROC is `0.6513310185185185` for TGRAANK and `0.7089120370370371` for VAR; pooled AUROC is `0.6093717603151566` and `0.6471076093717603`, respectively. Those pooled values must not replace the mean per graph values in the main table.

The [delivered evidence and complete commands](https://github.com/owuordickson/causalrivers/blob/2e970afabd6952e04c3de093ede4df562c974e96/evidence/review-232/README.md) identify the CPU profile (Windows x64, CPython 3.14.0, so4gp 1.0.8, Torch 2.13.0), source commits, frozen labels, configurations and predictions. The parent CI initializes the child and runs 19 hermetic regressions with minimal dependencies. Historical original producer source and scorer versions are explicitly unknown; later rescoring source is separately verified. This publication provides partial O1 river-network graph-recovery evidence. Maintainer synchronization with main, code-licence/integration decisions and contractual acceptance remain pending.

Notebook SHA-256: `7ae4c498f75605c0e5427684d6f2733e89bcb3cb0ce3aecd18a6725732ca7410`.
