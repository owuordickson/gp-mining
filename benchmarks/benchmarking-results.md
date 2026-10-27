# TGRAANK causal-discovery benchmark results

Refs #19. Published from the saved output of [benchmarking.ipynb](./benchmarking.ipynb) on 13 October 2026; updated on 26 October 2026 with the completed Disjoint-10 run. The current notebook comparison displays, [machine-readable export](./benchmarking-results.json) and table below include the completed result. The [earlier notebook snapshot](https://github.com/theseedship/deposium_geoai/blob/896075293f0be530f63df1e249d81ea282e09a92/benchmarks/benchmarking.ipynb) remains in Git history. Previous results are retained; no mining, rescoring or full notebook execution was performed for this publication update.

## Research focus and method families

The benchmark asks how TGRAANK, an asymmetric approach to causal discovery, performs across river-network graph structures alongside other causal-discovery families. The main comparison is TGRAANK, PCMCI, PC, VARLiNGAM and DYNOTEARS.

**TGRAANK outperforms PCMCI on the completed Disjoint-10 setting:** mean per graph AUROC is **0.815098** versus PCMCI's published **0.801787**, an absolute gain of **0.013312** (1.3312 percentage points). TGRAANK also scores higher on Close 3, Close 5, Root cause 3 and Random 3. It leads on five of the eleven reported settings; PCMCI leads on the other six. These setting-specific advantages compare archived TGRAANK results with the pinned published references, subject to the comparison limits below.

| Method | Family used in this comparison | Basis |
|---|---|---|
| TGRAANK / T-GRAANK | Asymmetric causal discovery, the author's proposed framing | Directional, temporally lagged gradual dependencies with ordered cause/effect roles. |
| PCMCI | Constraint-based | Conditional-independence tests adapted to time series. |
| PC | Constraint-based | Conditional-independence constraints and graph orientation rules. |
| VARLiNGAM | Asymmetry-based | Linear non-Gaussian structural models distinguish causal directions. |
| DYNOTEARS | Score-based | Continuous optimization of a graph objective with structural constraints. |

[Runge et al., *Causal inference for time series* (2023)](https://doi.org/10.1038/s43017-023-00431-y) supplies this taxonomy for the established methods; the supplied manuscript discusses these families on pages 14–15. TGRAANK's classification is the author's framing of the proposed approach, rather than an attribution of TGRAANK to that review. Directional gradual dependencies require empirical validation against causal ground truth; this report establishes no causal-identifiability theorem.

Names are normalized for publication: the notebook's `PCMI` means **PCMCI**, `Varlingam` means **VARLiNGAM**, and `Dynoteurs` means **DYNOTEARS**. PC denotes the constraint-based PC algorithm. The paper's **CP** denotes Causal Pretraining, a different method.

## CausalRivers: saved comparison with completed Disjoint-10

The metric is **mean per graph AUROC**, named `Individual AUROC` by the harness, with autoregressive diagonal entries excluded. It differs from pooled `AUROC`. The notebook's two saved comparison displays and the JSON table now show the completed Disjoint-10 result at six decimal places. The other ten TGRAANK values and the comparator values retain their previous display precision.

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
| Disjoint 10 | **0.815098** | 0.802 | Unavailable | 0.827 | 0.606 |

**Random+1-5 artifact identity.** The saved `0.734016` corresponds to the [preserved replacement's `scoring.csv`](https://github.com/owuordickson/causalrivers/blob/2e970afabd6952e04c3de093ede4df562c974e96/results/tgraank_1_random_5/recovered-replacement-3025328c/scoring.csv): its full value is `0.7340156985733362`. The path named in the notebook now contains the [restored original](https://github.com/owuordickson/causalrivers/blob/2e970afabd6952e04c3de093ede4df562c974e96/results/tgraank_1_random_5/20260921025013/scoring.csv), whose mean per graph AUROC is **0.7342750171305081**, or **0.734275** at the displayed precision. The original has independently reproducible rescoring against reconstructed ordered labels. This earlier saved value is retained in the notebook; reevaluating its scoring-table cell against the current gitlink would change this one displayed value.

**Completed Disjoint-10.** The [new scoring artifact](https://github.com/owuordickson/causalrivers/blob/318e88ff4b5fc9b146999ac1baf480715f155961/results/tgraank_disjoint_10/20261019151559/scoring.csv) reports mean per graph AUROC `0.8150984685418272` and pooled AUROC `0.7904192479040301`. The pinned PCMCI mean per graph reference is `0.8017869429388739`; the corresponding TGRAANK gain is `0.013311525602953345`. The [recorded configuration](https://github.com/owuordickson/causalrivers/blob/318e88ff4b5fc9b146999ac1baf480715f155961/results/tgraank_disjoint_10/20261019151559/config.yaml) selects East Germany Disjoint-10 graphs, lag 3, 6-hour resolution, interpolation, normalization and diagonal exclusion. The evaluated graph count and producer source/environment are not captured by this run's artifacts. All eleven settings now have finite TGRAANK scores. The current notebook displays, JSON table and Markdown table agree on Disjoint-10; the previous NaN is retained only in the linked historical snapshot. PC still has no reported Disjoint-10 value.

### Comparator provenance and comparison limits

The [CausalRivers website](https://causalrivers.github.io/) links its official [leaderboard CSV at revision `22ffa6bd`](https://github.com/CausalRivers/CausalRivers.github.io/blob/22ffa6bdc7cb3ac6b1947c743677c35a473aa459/scoring.csv). A [local snapshot](./causalrivers-leaderboard.csv) preserves those full-precision values. All 43 available comparator cells in the notebook match the selected leaderboard rows rounded to three decimals.

PCMCI, VARLiNGAM and DYNOTEARS are marked `From publication` in that CSV and appear in Table 3 of [Stein et al., *CausalRivers* (ICLR 2025)](https://openreview.net/forum?id=wmV4cIbgl6), also available as an [author-hosted arXiv PDF](https://arxiv.org/pdf/2503.17452). PC comes from the separate `Vanilla PC method-ElliSchool` community submission dated 4 September 2025, rather than that paper's Table 3. Its root-cause values occupy the CSV's additional `Root Cause 3` and `Root Cause 5` columns.

The paper reports mean sample AUROC for each method's best hyperparameter combination, across multiple resolutions and preprocessing settings. This report combines those published references with archived local TGRAANK scores. The comparator methods were not rerun here, and identical graph order, preprocessing, time windows, search budgets and sample exclusions across this full panel have not been established. These are descriptive comparisons across method families, not a newly controlled ranking, superiority claim or speedup measurement. Original producer source/environment/command metadata for the historical TGRAANK campaign is incomplete.

### TGRAANK scoring paths

The first ten paths retain the notebook's references at [CausalRivers child `2e970af`](https://github.com/owuordickson/causalrivers/tree/2e970afabd6952e04c3de093ede4df562c974e96). The completed Disjoint-10 path is at [child `318e88f`](https://github.com/owuordickson/causalrivers/tree/318e88ff4b5fc9b146999ac1baf480715f155961).

| Graph set | Scoring path |
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
| Disjoint 10 | `results/tgraank_disjoint_10/20261019151559/scoring.csv` (completed run) |

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

The [delivered evidence and complete commands](https://github.com/owuordickson/causalrivers/blob/2e970afabd6952e04c3de093ede4df562c974e96/evidence/review-232/README.md) identify the CPU profile (Windows x64, CPython 3.14.0, so4gp 1.0.8, Torch 2.13.0), source commits, frozen labels, configurations and predictions. The parent CI initializes the child and runs 19 hermetic regressions with minimal dependencies. Historical original producer source and scorer versions are explicitly unknown; later rescoring source is separately verified. This publication provides partial O1 river-network graph-recovery evidence. At the time of the review checkpoint, integration was pending; the maintainer subsequently accepted the integration decision and [PR #232](https://github.com/theseedship/deposium_geoai/pull/232) merged at [`229feb293eec74507a3bca45f34728522858a90f`](https://github.com/theseedship/deposium_geoai/commit/229feb293eec74507a3bca45f34728522858a90f). The merge does not establish an independent upstream source grant or contractual O1 acceptance; the applicable scientific mapping/receipt remains open.

Current notebook SHA-256 (LF-normalized file bytes): `9b5d0d814d3774cd4805fce4ab302fd387f3cb375018f5d2b8ec8fdd5afd6d4a`. The pre-formatting notebook SHA-256 was `3d2b7e67d1f5047e169b46af3bd923140f1e9212a72c415619536e848d0416c1`; the Ruff cleanup changed code layout/comments only and retained every saved cell output. No benchmark execution or rescoring was performed. The earlier snapshot SHA-256 is retained in the JSON export with its Git revision.
