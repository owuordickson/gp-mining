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

Synthetic timestamps encode original row positions; they do not establish temporal relationships among independent observations. The loader selects numeric attributes, excludes labels and IDs, handles dataset-specific missing markers, and uses complete cases. Every trial records the source and prepared-data hashes, selected columns, and retained rows. Directio sites are kept separate; use `--site` if the selected file contains multiple sites.

## Run

From the repository root, use the project's Python environment with its dependencies installed:

```powershell
python benchmarks/optimization/benchmark.py --download
python benchmarks/optimization/benchmark.py --datasets BCR AQY --experiment temporal --rows 64 --features 6 --lag 2
python benchmarks/optimization/benchmark.py --datasets BCR --experiment graank --seeds 0 1 2 --support 0.7
python benchmarks/optimization/benchmark.py --datasets DIR --profile study --supports 0.5 0.7 0.9 --timeout 600
```

Without options, the smoke profile uses at most 64 rows, six features, support 0.7, one seed, five heuristic iterations and two temporal lags. `--profile study` defaults to up to 1,000 rows, 12 features, supports 0.5/0.7/0.9, five seeds, and ten lags; `--profile full` uses the selected file's full prepared size and requires `--memory-mib`. Use `--rows`, `--features`, `--support`, `--supports`, `--seeds`, `--target`, `--lag`, `--iterations`, and `--timeout` to change these settings. `--apriori-level` optionally limits APRIORI depth; by default its search is not depth-limited. `--memory-mib` applies a best-effort sampled process-tree memory limit. Each trial has a discarded warm-up by default; use `--warmups` to change the count.

Smaller source files can be selected explicitly using `C2K1K`, `AQY1K`, `DIR1K`, `DIR8K`, `PWC1K`, `PWC10K`, `PWC19K`, `APS2K`, and `APS5K`; no file is substituted silently. Use `--data-dir` for an alternative cache. The pinned dense revision is extracted from local Git objects; a checkout without that object needs the revision fetched first.

## Results and interpretation

Each run creates a new folder under `results/`, with `manifest.json`, `trials.jsonl`, prepared CSVs, `summary.csv`, `paired_summary.csv`, `summary.json`, and `report.md`. A figure is created when sufficient pattern-recovery data is available. Use `--output` to choose a directory and `--resume` only with the same settings, source and prepared inputs. A timeout, memory limit, input failure, or algorithm exception is recorded as a failure rather than as a zero-duration result.

The algorithm timer includes miner construction and mining; input preparation and process startup are excluded, and process elapsed time is recorded separately. Sampled process-tree resident memory includes imports and post-mining validation, and is not a representation-only memory measure. Kernel records include separate dense and packed array sizes and intersection timings; both representations exist in each kernel process. Results from small smoke inputs check the pipeline and are not general performance claims.

See [benchmarking-results.json](benchmarking-results.json) for the saved benchmark record and its scope.
