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

For GRAANK, the default `study` profile uses up to 5,000 rows and 10 features
per dataset, with seeds 0 through 2. It runs all five search
types in both layouts at minimum support 0.2, with one discarded warm-up
per trial, a 300-second timeout, and a best-effort memory limit of 6,144 MiB
per worker. The `full` profile uses the full prepared datasets and one seed.
The `quick` profile uses up to 64 rows, six features, and one seed.

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
| Invalid candidates | Candidate rejections recorded by GRAANK's `Invalid Count` |
| Time and memory ratios | Baseline divided by comparison, using completed trials with matching inputs and seeds |
| Itemset overlap | Jaccard index of returned patterns, treating a pattern and its global sign reversal as equivalent |
| APRIORI output recall | Fraction of baseline APRIORI itemsets recovered by the comparison |

The CSVs report medians from completed trials. Timeouts and memory limits appear
as trial statuses. When GRAANK has an empty search space, the trial completes
with zero patterns and records `empty_search_space: true`.

Resume a run with `--resume --output <same-run-folder>` and the original
options, source, and data. The runner rebuilds the CSVs from `trials.jsonl`,
skips saved trials, and retries any interrupted trial without a record.

### Dense and packed APRIORI

The saved study ran six datasets with caps of 5,000 rows and 10 features,
three seeds, and a 10-iteration budget for each heuristic. The time and peak
RSS columns show the median for each completed method. Paired speedup is the
median of the dense-to-packed time ratios for matched seeds. [Run manifest](results/graank/manifest.json) · [Run summaries](results/graank/summary.csv) · [Paired summaries](results/graank/paired_summary.csv)

| Dataset | Rows × features | Dense time (s) or outcome | Packed time (s) or outcome | Paired speedup | Median peak RSS, dense → packed (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BCR | 116 × 9 | 21.999 | 0.371 | 58.15× | 375.1 → 374.2 |
| PWC | 5,000 × 7 | 7.306 | 5.023 | 1.45× | 1,069.1 → 476.7 |
| AQY | 5,000 × 10 | 6,144 MiB limit (3/3) | 24.067 | — | — → 1,293.8 |
| C2K | 3,942 × 10 | 12.451 | 6.297 | 1.98× | 1,088.5 → 481.7 |
| DIR | 5,000 × 10 | 280.627 (1/3 completed) | 62.883 | 4.46× (1 pair) | 5,295.1 → 1,029.8 |
| APS | 5,000 × 10 | 5.541 | 1.439 | 2.82× | 900.3 → 441.9 |

![APRIORI run time and peak memory across all six datasets, including the AQY dense memory limits and DIR's single dense completion.](results/graank/apriori_dense_vs_packed.png)

Dense and packed APRIORI returned the same itemsets on BCR, C2K, and PWC
at all three seeds after accounting for full sign reversals. On APS, packed
returned four of the five dense itemsets at each seed. The single completed
DIR pair had identical itemsets, a **4.46× speedup**, and roughly 81% lower
peak RSS for packed. Packed had a paired speedup above 1× in 27 of the 29
same-search comparisons with a completed pair. APS hill climbing (0.50×)
and BCR GA (0.84×) had the two lower ratios. Across non-BCR heuristic
comparisons, packed peak RSS was 43–58% lower by paired medians.
[Paired summaries](results/graank/paired_summary.csv)

Of the 180 measured trials, 175 completed, including all 144 heuristic trials
and all 18 packed APRIORI trials. The five incomplete runs were dense APRIORI.
Three AQY runs hit the memory limit, and two DIR runs timed out.
[Trial records](results/graank/trials.jsonl)

### Heuristic speed and pattern recovery

The table compares searches using packed bitmaps. Pattern and invalid candidate
counts are totals over the three seeds, and all returned patterns had reported
support of at least 0.2. APRIORI recall is the median fraction of its distinct
itemsets each search recovered at the same seed. [Run summaries](results/graank/summary.csv) · [Paired summaries](results/graank/paired_summary.csv)

| Dataset | Search | Median time (s) | Paired speedup vs. APRIORI | Returned patterns | Invalid candidates | APRIORI recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BCR | APRIORI | 0.371 | 1.00× | 174 | 2,791 | 100.00% |
| BCR | Genetic algorithm (GA) | 0.077 | 4.91× | 23 | 41 | 5.66% |
| BCR | Hill climbing (HC) | 0.033 | 9.79× | 7 | 1 | 1.75% |
| BCR | Particle swarm optimization (PSO) | 0.041 | 7.99× | 2 | 100 | 0.00% |
| BCR | Random search | 0.031 | 11.99× | 9 | 9 | 0.00% |
| PWC | APRIORI | 5.023 | 1.00× | 30 | 176 | 100.00% |
| PWC | Genetic algorithm (GA) | 9.529 | 0.53× | 20 | 27 | 30.00% |
| PWC | Hill climbing (HC) | 4.182 | 1.25× | 4 | 0 | 0.00% |
| PWC | Particle swarm optimization (PSO) | 9.499 | 0.55× | 11 | 67 | 10.00% |
| PWC | Random search | 3.844 | 1.36× | 16 | 6 | 20.00% |
| AQY | APRIORI | 24.067 | 1.00× | 39 | 749 | 100.00% |
| AQY | Genetic algorithm (GA) | 17.742 | 1.41× | 27 | 32 | 15.38% |
| AQY | Hill climbing (HC) | 7.113 | 3.50× | 8 | 0 | 0.00% |
| AQY | Particle swarm optimization (PSO) | 20.968 | 1.19× | 3 | 0 | 0.00% |
| AQY | Random search | 6.871 | 3.55× | 11 | 2 | 7.69% |
| C2K | APRIORI | 6.297 | 1.00× | 66 | 438 | 100.00% |
| C2K | Genetic algorithm (GA) | 9.397 | 0.67× | 22 | 113 | 10.00% |
| C2K | Hill climbing (HC) | 4.615 | 1.36× | 5 | 5 | 0.00% |
| C2K | Particle swarm optimization (PSO) | 8.669 | 0.64× | 3 | 170 | 0.00% |
| C2K | Random search | 3.772 | 1.67× | 9 | 12 | 5.00% |
| DIR | APRIORI | 62.883 | 1.00× | 76 | 6,550 | 100.00% |
| DIR | Genetic algorithm (GA) | 14.183 | 4.61× | 30 | 60 | 9.09% |
| DIR | Hill climbing (HC) | 6.725 | 9.20× | 5 | 1 | 0.00% |
| DIR | Particle swarm optimization (PSO) | 14.717 | 4.27× | 3 | 119 | 0.00% |
| DIR | Random search | 6.102 | 10.31× | 10 | 6 | 0.00% |
| APS | APRIORI | 1.439 | 1.00× | 12 | 30 | 100.00% |
| APS | Genetic algorithm (GA) | 6.572 | 0.20× | 10 | 99 | 50.00% |
| APS | Hill climbing (HC) | 1.466 | 1.03× | 4 | 0 | 25.00% |
| APS | Particle swarm optimization (PSO) | 3.726 | 0.39× | 12 | 134 | 75.00% |
| APS | Random search | 1.581 | 1.11× | 12 | 8 | 50.00% |

![Packed search run time and APRIORI itemset recall across all six datasets.](results/graank/packed_search_tradeoff.png)

On BCR, the heuristics ran **4.91–11.99× faster** by paired medians and
reached at most **5.66%** median APRIORI recall. On PWC and C2K, GA had the
highest heuristic recall at **30.00%** and **10.00%**, respectively, with
paired speedups below 1× in both cases. On AQY, all four heuristics ran
faster by paired medians, and GA reached the highest heuristic recall
at **15.38%**. On APS, PSO reached **75.00%** recall at 0.39× APRIORI speed,
while random search reached **50.00%** at 1.11×.

### DIR search results

All four DIR heuristic searches completed on all three seeds in each layout.
The paired speedup in this table compares dense and packed versions of the
same search. Packed APRIORI completed on all three seeds, providing a
baseline for the packed heuristic recall estimates. [Run summaries](results/graank/summary.csv) · [Paired summaries](results/graank/paired_summary.csv)

| Search | Dense median time (s) | Packed median time (s) | Paired speedup | Median peak RSS, dense → packed (MiB) | Returned patterns, dense → packed | Invalid candidates, dense → packed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GA | 29.972 | 14.183 | 2.23× | 1,145.9 → 482.6 | 25 → 30 | 31 → 60 |
| HC | 9.223 | 6.725 | 1.35× | 1,145.5 → 488.1 | 5 → 5 | 1 → 1 |
| PSO | 39.634 | 14.717 | 3.83× | 1,145.2 → 485.5 | 5 → 3 | 9 → 119 |
| Random search | 8.222 | 6.102 | 1.29× | 1,145.4 → 483.2 | 10 → 10 | 6 → 6 |

On DIR, the packed heuristics achieved **1.29–3.83× paired speedups** over
their dense counterparts and used 57–58% less peak RSS by paired medians.
Packed random search was fastest at **6.102 s** and had the highest paired
speedup against packed APRIORI at **10.31×**. GA recovered **9.09%** of packed
APRIORI itemsets by median recall. The other three heuristics had zero
median recall.

## TGRAANK comparison on rainfall and EVI

The temporal experiment uses the notebook's rainfall and Enhanced Vegetation
Index (EVI) CSVs. Each file has 23 annual observations from 2000 through 2022,
with NRB, KAK, MAK, and WAJ as the four targets.

```powershell
python benchmarks/optimization/benchmark.py --experiment temporal --profile full --warmups 0 --output benchmarks/optimization/results/tgraank-replication
python benchmarks/optimization/tools/analyze_tgraank.py benchmarks/optimization/results/tgraank-replication
```

Use a fresh output directory for each replication and pass the same path to
the analysis helper. The saved runs are in [`results/tgraank`](results/tgraank/).

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
- [Runtime and memory by target](results/tgraank/summary.csv)

### Replication results

The saved run completed 24 trials with SO4GP 1.1.2 and seed 0, one per
method, dataset, and site combination. The table sums confusion counts
across sites. The heatmaps show mean counts per site.
[Run manifest](results/tgraank/manifest.json) · [Per-site confusion counts](results/tgraank/confusion_matrix.csv)

| Method | True positives | False positives | False negatives | True negatives | Scored entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical TGRAANK | 34 | 26 | 26 | 229 | 315 |
| TGRAANK with random shifts | 6 | 4 | 6 | 47 | 63 |
| TGRAANK-AMI | 8 | 4 | 4 | 47 | 63 |

![Mean confusion matrices for classical TGRAANK, AMI, and random shifts. Rows are EVI reference classes and columns are rainfall prediction classes.](results/tgraank/confusion_matrix.png)

Scores are mean percentages ± sample standard deviation across the four
equally weighted sites. Each method supplies its own set of mined patterns
for scoring. [Metric summary](results/tgraank/metrics_summary.csv)

| Method | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical TGRAANK | 81.23 ± 9.26 | 56.46 ± 19.72 | 56.67 ± 20.73 | 87.65 ± 6.46 | 56.53 ± 20.17 |
| TGRAANK with random shifts | 79.54 ± 16.43 | 58.33 ± 28.87 | 50.00 ± 33.33 | 88.43 ± 12.12 | 53.33 ± 31.27 |
| TGRAANK-AMI | **84.18 ± 12.94** | **66.67 ± 27.22** | **66.67 ± 27.22** | **89.51 ± 8.62** | **66.67 ± 27.22** |

![Accuracy, precision, recall, specificity, and F1 by TGRAANK method. Points are site averages and error bars show one sample standard deviation.](results/tgraank/classification_metrics.png)

AMI had the highest mean on all five metrics. Its mean accuracy was
**2.95 percentage points above classical TGRAANK** and **4.64 points above
random shifts**. Mean F1 rose by 10.14 and 13.33 points, respectively.
Accuracy improved over classical at KAK, NRB, and WAJ. At MAK it fell
from 76.92% to 69.23%.
[Per-site scores](results/tgraank/confusion_matrix.csv)
