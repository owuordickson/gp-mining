# GRAANK optimization benchmark

For each dataset and seed, the selected searches use the same rows, features,
target, and support threshold.

| Method | Search types | GRAANK source | Bitmap representation |
| --- | --- | --- | --- |
| `dense` | `apriori` | Git revision `affc7860397002fba26af96dcb2f0461f2cdda9d` | Dense `n × n` bitmaps |
| `packed` | `apriori`, `pso`, `ga`, `hc`, `random` | `from so4gp.algorithms import GRAANK` in the selected Python interpreter | Packed bits |

Each trial runs in its own subprocess. The dense worker imports from a verified
archive of the pinned revision in local Git history. The packed worker uses the
interpreter's normal import path. The dense-to-packed APRIORI comparison
includes any other behavior that differs between the two sources.

## Data

The benchmark reads the six CSVs in [`data/`](data/). Every trial records the
source and prepared-data hashes, selected columns, and retained rows.

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
revision must be present in local Git history when running APRIORI.

In PyCharm, run `benchmarks/optimization/benchmark.py` as a script with that
interpreter or use `benchmarks.optimization.benchmark` as the module name from
the repository root.

```powershell
python benchmarks/optimization/benchmark.py --datasets BCR --warmups 0
python -m benchmarks.optimization.benchmark --datasets BCR AQY --profile study --search-types apriori
python benchmarks/optimization/benchmark.py --datasets BCR --search-types pso ga hc random --iterations 20
```

The default `full` profile uses all six datasets at their full prepared sizes
and runs all five search types. The defaults are support 0.2, one measured trial
per search and bitmap method with seed 0, one discarded warm-up per trial, a
300-second worker timeout, and a best-effort memory limit of 6,144 MiB per worker.
The `study` profile uses up to 1,000 rows, 12 features, and three seeds. The
`quick` profile uses up to 64 rows, six features, and one seed.

Use `--search-types` to select searches. `--iterations` sets the heuristic
iteration budget and defaults to 10. `--apriori-level` sets an optional depth
limit for both APRIORI methods. Use `--rows`, `--features`, `--support`, `--target`,
and `--seeds` to define a workload.
`--memory-mib`, `--timeout`, and `--warmups` control the trial limits and warm-ups.

## Results and resuming

Each run creates a folder under [`results/`](results/) with `manifest.json`,
`trials.jsonl`, `summary.csv`, and `paired_summary.csv`. The manifest records
the dense revision and the packed import path. The runner saves each trial as
it finishes and refreshes both CSVs at the end of each dataset.

The summary lists each `search_type` and bitmap `method` separately. Packed
APRIORI is paired with dense APRIORI. Each heuristic is paired with packed
APRIORI, so include `apriori` in the run to get those comparisons.

| Metric | Definition |
| --- | --- |
| Elapsed time | Miner construction and the selected search |
| Peak resident memory | Sampled worker process-tree RSS in bytes and MiB, including imports and result collection |
| Valid patterns | Patterns returned by GRAANK |
| Invalid patterns | Candidates rejected according to GRAANK's `Invalid Count` |
| Time and memory ratios | Baseline divided by comparison, using completed trials with matching inputs and seeds |
| Itemset overlap | Jaccard index of returned patterns, treating a pattern and its global sign reversal as equivalent |
| APRIORI output recall | Fraction of baseline itemsets recovered by the comparison |

The CSVs report medians from completed trials. Timeouts and memory limits appear
as trial statuses. When GRAANK has an empty search space, the trial completes
with zero patterns and records `empty_search_space: true`.

Resume a run with `--resume --output <same-run-folder>` and the original
options, source, and data. The runner rebuilds the CSVs from `trials.jsonl`,
skips saved trials, and retries any interrupted trial without a record.

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

## Full BCR APRIORI run at support 0.5

One measured pair used all 116 BCR rows and nine features, with Glucose as the
target, support 0.5, seed 0, and no warm-up. The packed worker imported SO4GP
1.1.2 from the project interpreter.

| Method | Time (ms) | Peak RSS (MiB) | Valid patterns | Invalid patterns |
| --- | ---: | ---: | ---: | ---: |
| Dense | 58.3 | 373.6 | 8 | 35 |
| Packed | 54.2 | 371.7 | 8 | 35 |

Returned itemsets agreed after sign normalization. The dense-to-packed ratios
were 1.08 for elapsed time and 1.005 for peak RSS.

## Earlier BCR comparison

The saved BCR comparison used 32 rows, four numeric attributes, support 0.7,
Glucose as the target, and one trial per method. It used the pinned dense
revision and packed source at `eb981fb`. [Saved records](benchmarking-results.json)

| Method | Time (ms) | Peak RSS (MiB) | Returned patterns |
| --- | ---: | ---: | ---: |
| Dense | 14.542 | 372.6 | 0 |
| Packed | 7.707 | 372.3 | 0 |

On this input, the dense trial took 1.89 times as long. The sampled peak RSS
differed by 0.3 MiB, and both searches returned an empty pattern set.
