# GRAANK and TGRAANK optimization benchmark

For each dataset and seed, the selected searches use the same rows, features,
target, and support threshold.

| Method | Search types | GRAANK source | Bitmap representation |
| --- | --- | --- | --- |
| `dense` | `apriori`, `pso`, `ga`, `hc`, `random` | Git revision `affc7860397002fba26af96dcb2f0461f2cdda9d` | Dense `n × n` bitmaps |
| `packed` | `apriori`, `pso`, `ga`, `hc`, `random` | `from so4gp.algorithms import GRAANK` in the selected Python interpreter | Packed bits |

Each trial runs in its own subprocess. The dense worker imports from a verified
archive of the pinned revision in local Git history. The packed worker uses the
interpreter's normal import path. The dense-to-packed comparisons also
reflect other differences between the two source revisions.

## Data

The GRAANK experiment reads six CSVs in [`data/`](data/). Every trial records
the source and prepared-data hashes, selected columns, and retained rows.

| Dataset | File | Default target |
| --- | --- | --- |
| BCR | `breast_cancer.csv` | `Glucose` |
| C2K | `c2k.csv` | `o_dlv_e` |
| AQY | `air_quality.csv` | `CO(GT)` |
| DIR | `directio15k.csv` | `Hm0` |
| PWC | `power_consumption50k.csv` | `Global_active_power` |
| APS | `aps_16k.csv` | `aa_000` |

The loader selects numeric attributes, excludes labels and IDs, and handles
dataset-specific missing values using complete cases. It prepares the selected
data in memory and sends it directly to each worker. Use `--site` if the
Directio file contains multiple sites. The datasets originate from the
[source data directory](https://github.com/owuordickson/meso-hpc-lr/tree/master/hpc%40lr/data).

## Run

Use an interpreter with a packed-bit SO4GP install. To use this checkout's
package, run `python -m pip install -e .` from the repository root. The dense
revision must be present in local Git history when running GRAANK.

In PyCharm, run `benchmarks/optimization/benchmark.py` as a script with that
interpreter or use `benchmarks.optimization.benchmark` as the module name from
the repository root.

```powershell
python benchmarks/optimization/benchmark.py --datasets BCR --warmups 0
python -m benchmarks.optimization.benchmark --datasets BCR AQY --profile study --search-types apriori
python benchmarks/optimization/benchmark.py --datasets BCR --search-types pso ga hc random --iterations 20
```

For GRAANK, the default `full` profile uses all six datasets at their full
prepared sizes and runs all five search types in both layouts. The defaults
are support 0.2, one measured trial per search and bitmap method with seed 0,
one discarded warm-up per trial, a 300-second worker timeout, and a best-effort
memory limit of 6,144 MiB per worker.
The `study` profile uses up to 1,000 rows, 12 features, and three seeds. The
`quick` profile uses up to 64 rows, six features, and one seed.

Use `--search-types` to select searches. `--iterations` sets the heuristic
iteration budget and defaults to 10. `--apriori-level` sets an optional depth
limit for both APRIORI methods. Use `--rows`, `--features`, `--support`, `--target`,
and `--seeds` to define a workload.
`--memory-mib`, `--timeout`, and `--warmups` control the trial limits and warm-ups.

## GRAANK Results

Each run creates a folder under [`results/`](results/) with `manifest.json`,
`trials.jsonl`, `summary.csv`, and `paired_summary.csv`. The manifest records
the run settings and algorithm sources. The runner saves each trial as
it finishes and refreshes both CSVs at the end of each dataset.

The GRAANK summary lists each `search_type` and bitmap `method` separately.
The paired summary compares packed against dense for the same search.
When APRIORI is included, it also compares each heuristic with APRIORI
in the same layout.

| Metric | Definition |
| --- | --- |
| Elapsed time | Miner construction and the selected search |
| Peak resident memory | Sampled worker process-tree RSS in bytes and MiB, including imports and result collection |
| Valid patterns | Patterns returned by GRAANK |
| Invalid patterns | Candidates rejected according to GRAANK's `Invalid Count` |
| Time and memory ratios | Baseline divided by comparison, using completed trials with matching inputs and seeds |
| Itemset overlap | Jaccard index of returned patterns, treating a pattern and its global sign reversal as equivalent |
| APRIORI output recall | Fraction of baseline APRIORI itemsets recovered by the comparison |

The CSVs report medians from completed trials. Timeouts and memory limits appear
as trial statuses. When GRAANK has an empty search space, the trial completes
with zero patterns and records `empty_search_space: true`.

Resume a run with `--resume --output <same-run-folder>` and the original
options, source, and data. The runner rebuilds the CSVs from `trials.jsonl`,
skips saved trials, and retries any interrupted trial without a record.

## TGRAANK comparison on rainfall and EVI

The temporal experiment uses the notebook's rainfall and Enhanced Vegetation
Index (EVI) CSVs. Each file has 23 annual observations from 2000 through 2022,
with NRB, KAK, MAK, and WAJ as the four targets.

```powershell
python benchmarks/optimization/benchmark.py --experiment temporal --warmups 0
python benchmarks/optimization/tools/analyze_tgraank.py
```

Both commands use [`results/tgraank`](results/tgraank/) by default. To
repeat the experiment in a separate folder, set `--output` on the benchmark
and pass that directory to the analysis helper.

All three methods mine fuzzy temporal gradual patterns with minimum support
0.01 and initial minimum representativity 0.75. The worker calls SO4GP's
`TGrad` and `TGradAMI` engines with descriptors enabled. TGRAANK tests shifts
of one through five observations on each dataset (`all` in the benchmark).
TGRAANK-AMI estimates shifts on rainfall with an MI error margin of 0.00001
and reuses those shifts for the corresponding EVI target (`ami`).

TGRAANK-AMI (Random) draws one integer k-step per feature from 1 through 6
for each target (`random`). The runner uses
`42 * target_column_index + trial_seed` as the random seed for those draws
and applies them to both rainfall and EVI. With the default trial seed of 0,
the draws use the notebook's target-specific seeds.

The analysis uses rainfall as the prediction and EVI as the reference proxy.
It matches patterns once using SO4GP's temporal similarity rule, which compares
gradual items and signed, formatted time delays. After matching, support of
at least 0.5 counts as positive, and an unmatched pattern gets zero support on
the other side. Accuracy is the fraction of matched or unmatched entries
that are true positives or true negatives. The summary averages the per-site
scores and reports their sample standard deviations.

- [Per-site confusion counts and scores](results/tgraank/confusion_matrix.csv)
- [Mean and standard deviation by algorithm](results/tgraank/metrics_summary.csv)
- [Mean confusion matrices](results/tgraank/confusion_matrix.png)
- [Runtime and memory by target](results/tgraank/summary.csv)

### Replication results

The saved run completed 16 trials with SO4GP 1.1.2 and seed 0. Scores are
percentages shown as mean ± sample standard deviation across the four sites.
[Run manifest](results/tgraank/manifest.json)

| Method | Accuracy | Precision | Recall | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TGRAANK | 81.23 ± 9.26 | 56.46 ± 19.72 | 56.67 ± 20.73 | 87.65 ± 6.46 | 56.53 ± 20.17 |
| TGRAANK-AMI | 84.18 ± 12.94 | 66.67 ± 27.22 | 66.67 ± 27.22 | 89.51 ± 8.62 | 66.67 ± 27.22 |

AMI improved mean accuracy by **2.95 percentage points** and reduced false
positives and false negatives from 6.5 to 1.0 each per site on average.
Accuracy rose at KAK, NRB, and WAJ. At MAK it fell from 76.92% to 69.23%.
[Per-site scores](results/tgraank/confusion_matrix.csv)

## Full BCR run at support 0.2

This run used all 116 rows and nine features, with Glucose as the target and
seed 0. It measured each search once without a warm-up and used 10 iterations
for each heuristic.

| Search | Bitmap | Time (s) | Peak RSS (MiB) | Valid patterns | Invalid patterns |
| --- | --- | ---: | ---: | ---: | ---: |
| APRIORI | Dense | 20.740 | 374.4 | 55 | 921 |
| APRIORI | Packed | 19.978 | 372.3 | 55 | 921 |
| GA | Packed | 0.058 | 372.0 | 5 | 11 |
| HC | Packed | 0.051 | 371.7 | 1 | 0 |
| PSO | Packed | 0.130 | 438.5 | 1 | 0 |
| Random | Packed | 0.057 | 371.4 | 4 | 1 |

The APRIORI itemsets agreed after sign normalization. GA and HC each recovered
one of the 53 distinct packed APRIORI itemsets. PSO and random had zero overlap
with that set.
