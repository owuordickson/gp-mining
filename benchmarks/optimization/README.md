# Optimization benchmark

This benchmark measures three related questions:

1. How do the current packed-bit GRAANK and the preceding dense-bitmap revision compare? A separate intersection experiment checks equal dense and packed results and measures their operations and array sizes.
2. How do exhaustive TGRAANK transformations compare with TGRAANK-AMI and with seeded random transformation steps?
3. How do APRIORI, ACO, GA, PSO, hill climbing, and random search compare in runtime, memory, and returned patterns?

ClusterGP is currently excluded because of an existing bug. Historical-version results include all changes since the baseline, not just packing. AMI and exhaustive temporal transformations need not return the same patterns. Itemset overlap ignores support and time-lag differences.

## Data

Place the six CSV files from the [source data directory](https://github.com/owuordickson/meso-hpc-lr/tree/master/hpc%40lr/data) in `data/`, or use `--download`. Downloads are pinned to revision `9f128eada3da0604fe18e6474d427ced7e72ffc6`. The files are kept locally and ignored by Git.

| Dataset | File | Default temporal target | Time |
| --- | --- | --- | --- |
| BCR | `breast_cancer.csv` | `Glucose` | Synthetic row order |
| C2K | `c2k.csv` | `o_dlv_e` | Synthetic row order |
| AQY | `air_quality.csv` | `CO(GT)` | Recorded date and time |
| DIR | `directio15k.csv` | `Hm0` | Recorded `date_time` |
| PWC | `power_consumption50k.csv` | `Global_active_power` | Recorded date and time |
| APS | `aps_16k.csv` | `aa_000` | Synthetic row order |

Synthetic timestamps encode original row positions; they do not establish temporal relationships among independent observations. The loader selects numeric attributes, excludes labels and IDs, handles dataset-specific missing markers, and uses complete cases. It prepares any selected subset in memory directly from the files in `data/`; no separate prepared dataset files are created. Every trial records the source and prepared-data hashes, selected columns, and retained rows. Directio sites are kept separate; use `--site` if the selected file contains multiple sites.

## Run

In PyCharm, open `benchmarks/optimization/benchmark.py` and choose **Run** with
the project's Python interpreter. The run configuration can use **Script path**
with `benchmark.py` or **Module name** with `benchmarks.optimization.benchmark`
from the repository root. With no parameters, the runner uses the study profile
and the dataset entries enabled in `tools/datasets.py`.

From the repository root, use the project's Python environment with its dependencies installed:

```powershell
python benchmarks/optimization/benchmark.py --download
python benchmarks/optimization/benchmark.py --datasets BCR AQY --experiment temporal --rows 64 --features 6 --lag 2
python benchmarks/optimization/benchmark.py --datasets BCR --experiment graank --seeds 0 1 2 --support 0.7
python benchmarks/optimization/benchmark.py --datasets DIR --profile study --supports 0.5 0.7 0.9 --timeout 600
```

Without options, the study profile uses up to 1,000 rows, 12 features, supports 0.5/0.7/0.9, five seeds, and ten lags. `--profile quick` uses at most 64 rows, six features, support 0.7, one seed and two lags. The term "smoke test" previously used here meant a small check that the pipeline ran; it did not represent a complete experiment. `--profile full` uses the selected file's full prepared size and requires `--memory-mib`. Use `--rows`, `--features`, `--support`, `--supports`, `--seeds`, `--target`, `--lag`, `--iterations`, and `--timeout` to change these settings. `--apriori-level` optionally limits APRIORI depth; by default its search is not depth-limited. `--memory-mib` applies a best-effort sampled process-tree memory limit. Each trial has a discarded warm-up by default; use `--warmups` to change the count.

Smaller source files can be selected explicitly using `C2K1K`, `AQY1K`, `DIR1K`, `DIR8K`, `PWC1K`, `PWC10K`, `PWC19K`, `APS2K`, and `APS5K`; no file is substituted silently. Use `--data-dir` for an alternative cache. The pinned dense revision is extracted from local Git objects; a checkout without that object needs the revision fetched first.

## Earlier small-sample results

The [saved records](benchmarking-results.json) contain two bounded smoke runs,
not a full performance study. Each method has **one measured trial with seed 0
and no warm-up**. There are no repeated-trial confidence intervals. All 25
trials completed successfully. The machine used Windows 11, an eight-logical-CPU
Intel processor, 32 GiB RAM, Python 3.14.0, and NumPy 2.5.3; the JSON includes
the source revisions, dependency versions, input hashes, and individual trials.

### Dense versus packed intersections

The kernel selects the first two numeric features and creates one comparison
array for each. The table reports the total time for **10 intersections and
counts**. The ratio is dense time divided by packed time, so a value below
1 means the packed operation took longer. Array sizes cover those two arrays
only; construction and packing are excluded from intersection time.

| Dataset | Retained / selected rows | Dense time (µs) | Packed time (µs) | Dense / packed time | Dense arrays (bytes) | Packed arrays (bytes) | Equal results |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| BCR | 64 / 64 | 54.9 | 236.9 | 0.232× | 8,192 | 1,024 | Yes |
| C2K | 64 / 64 | 62.0 | 184.1 | 0.337× | 8,192 | 1,024 | Yes |
| AQY | 57 / 64 | 79.6 | 236.6 | 0.336× | 6,498 | 814 | Yes |
| DIR | 64 / 64 | 55.2 | 198.6 | 0.278× | 8,192 | 1,024 | Yes |
| PWC | 64 / 64 | 30.8 | 280.8 | 0.110× | 8,192 | 1,024 | Yes |
| APS | 16 / 64 | 26.1 | 139.1 | 0.188× | 512 | 64 | Yes |

The packed arrays use about one eighth of the space in this isolated comparison;
AQY needs padding because 57² bits do not fill a whole number of bytes. All
intersections agree. Packed intersection and counting took longer in these tiny
samples, but the single, un-warmed measurements do not establish how the methods
scale. AQY and APS retained fewer than 64 rows after complete-case filtering.
Array sizes do not measure the full algorithm's peak memory.

### GRAANK versions on BCR

The remaining tables use the first 32 BCR rows and four numeric attributes
(Age, BMI, Glucose, and Insulin), support 0.7, and Glucose as the target. The
legacy revision is `affc786`; the current revision is `eb981fb`. Times cover
miner construction and mining. Peak memory is sampled process-tree resident
memory, including imports and post-mining validation.

| Version | Time (ms) | Peak memory (MiB) | Returned patterns | Legacy / version time |
| --- | ---: | ---: | ---: | ---: |
| Legacy dense bitmap | 14.542 | 372.6 | 0 | 1.00× |
| Current packed bits | 7.707 | 372.3 | 0 | 1.89× |

The current revision took less time in this single run. This compares whole
versions, including changes made after packing, and both returned no patterns;
it cannot isolate a packing speedup or establish output equivalence on a
non-empty result.

### Temporal methods on BCR

The temporal runs use synthetic row-order timestamps and a maximum lag of two.
For these independent BCR cases the timestamps support a computational test,
not a real-time or causal interpretation. The ratio is TGRAANK time divided by
the method's time.

| Method | Time (ms) | Peak memory (MiB) | Returned patterns | TGRAANK / method time |
| --- | ---: | ---: | ---: | ---: |
| TGRAANK, all transformations | 65.211 | 372.1 | 0 | 1.00× |
| TGRAANK-AMI | 213.045 | 373.8 | 0 | 0.31× |
| TGRAANK-AMI, random steps | 36.308 | 373.9 | 0 | 1.80× |

AMI took about 3.27 times as long as TGRAANK in this small run, while the random
step baseline took less time. All three returned empty outputs, so the recorded
itemset overlap of 1.0 is agreement between empty sets; it is not evidence of
pattern recovery or temporal accuracy.

### GRAANK search methods on BCR

Heuristic methods used five iterations. APRIORI had no depth limit. The ratio
is APRIORI time divided by the method's time; equal iteration counts do not imply
equal amounts of search work.

| Search method | Time (ms) | Peak memory (MiB) | Returned patterns | APRIORI / method time |
| --- | ---: | ---: | ---: | ---: |
| APRIORI | 8.692 | 372.4 | 0 | 1.00× |
| ACO | 246.811 | 372.2 | 0 | 0.04× |
| GA | 13.302 | 372.5 | 0 | 0.65× |
| PSO | 11.790 | 373.3 | 0 | 0.74× |
| Hill climbing | 19.812 | 372.7 | 0 | 0.44× |
| Random | 12.405 | 373.2 | 0 | 0.70× |

APRIORI was the fastest on this particular small input. None of the methods
returned a pattern, making APRIORI-output recall undefined; a non-empty workload
and repeated seeds are needed to compare pattern quality and search efficiency.
The approximately 372–374 MiB memory readings include the interpreter and
libraries, so they do not show bitmap storage savings on their own.

## Result files and interpretation

When too few gradual items meet the support threshold to start GRAANK's search,
the trial completes with zero patterns and records `empty_search_space: true`.

Each run creates a folder under `results/` with `manifest.json`, `trials.jsonl`,
`summary.csv`, and `paired_summary.csv`. The runner saves each trial to
`trials.jsonl` as it finishes and refreshes the cumulative summaries when each
dataset finishes. A figure is created when enough pattern-recovery data is
available. The runner passes selected data from `data/` in memory to each
isolated worker. Timeouts, memory limits, input failures, and algorithm
exceptions are recorded as failures.

The pattern columns in `summary.csv` total returned patterns across successful
trials in each row. A pattern is valid if its recomputed support meets the
threshold. Patterns whose validity could not be checked have their own count.
Blank cells indicate unavailable counts, including for temporal and kernel trials.
The summary gives median peak resident memory in bytes and MiB, where
1 MiB equals 1,048,576 bytes.

To continue a stopped run, use `--resume --output <same-run-folder>` with the
same run options, source, and data. The runner rebuilds the reports from
`trials.jsonl` and skips trials already recorded there. An interrupted trial
without a saved record runs again.

The algorithm timer includes miner construction and mining; input preparation and process startup are excluded, and process elapsed time is recorded separately. Sampled process-tree resident memory includes imports and post-mining validation, and is not a representation-only memory measure. Kernel records include separate dense and packed array sizes and intersection timings; both representations exist in each kernel process. Results from small smoke inputs check the pipeline and are not general performance claims.

See [benchmarking-results.json](benchmarking-results.json) for the saved benchmark record and its scope.
