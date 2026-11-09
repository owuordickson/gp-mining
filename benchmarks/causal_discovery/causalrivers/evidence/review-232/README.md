# PR #232 review fixes — 12 October 2026

Refs #19. This is a bounded O1 river-network graph-recovery contribution. It is not held-out weather/NDVI forecast validation or full contractual acceptance. No GPU campaign or multi-method/multi-day rerun is claimed.

## Historical source clarification (13 October 2026)

The producer source commit, source bytes and scorer version are unknown for the restored Random+1-5 original, its preserved replacement, and the older Confounder-3 run. Their reconstructed `provenance.json` records now say `unknown-historical-producer`; the former `8bb9cf0` claim is withdrawn because its later working-tree hashes do not identify that Git revision. The superseded metadata remains in Git history at `dfe65f45`. Frozen/reconstructed input hashes identify later rescoring inputs, not recovered original production inputs, commands or environments.

The separate portable `rescore.json` records retain the verified later scoring source at `dfadcd91`. For recorded source files in the pair, rescoring and validation records, `hash_bytes` now declares either exact Git blob bytes or the Git blob converted to Windows CRLF newlines. Each hash was checked against its declared commit. Dependent manifest and rescore provenance hashes have been refreshed. Prediction, configuration, runtime, score and label artifacts are unchanged; no mining or scoring rerun was needed for this metadata correction.

Tested regression and historical-rescoring source SHA: `dfadcd91410446148f1f767788d72bea37b2888e`. The scored pair source SHA is `a95c5fdc48fb74c6bdd1837e9c1496011813abdd`; the subsequent code change makes reconstructed-label rescoring portable and does not change either method or the scorer. The final child evidence commit contains these sources plus the recorded artifacts. `validation.json` links the resolved environment, actual installation report, source hashes and passing tests. Supported execution profile: Windows x64, CPython 3.14.0, CPU. The historical Conda/notebook export is not a validated alternate profile for this delivery.

| Review item | Concrete correction and evidence |
|---|---|
| 1 | Benchmark-root path/key normalization; missing/malformed/absent/mismatched digests refuse before unpickling. Verified bytes are the loaded bytes. `setup.json` independently verifies all three product graphs and the East Germany CSV against the pinned external archive. `generation.txt` records bounded confounder generation: East 24, Bavaria 11, flood 1; ordered East topology matches the frozen selection. |
| 2 | Clean pip installation from `requirements.txt` succeeded; `pip-install.json`, `install.txt`, `pip-check.txt`, `imports.txt`, `environment.json`, and `requirements-resolved.txt` provide actual installation/import/version/CPU evidence. Key Conda runtime pins were aligned; no Conda solve/install is claimed. |
| 3 | Constant columns preserve missing observations; only finite observed entries become zero. Empty, no-valid-row, short, unresolved-gap and nonfinite selections fail before any method call. Scoring rejects invalid shapes/nonfinite results. TGRAANK receives a deep copy because upstream mutates DataFrames; selection is captured before invocation. All 19 hermetic regressions pass. |
| 4 | All four original historical result files restored byte for byte from `7ff9c9f8c0b2468ddef05df5cdb82115a4ee00e1`; replacement retained under `recovered-replacement-3025328c`. Both checksummed prediction versions rescored against the same 2,790 ordered graphs reconstructed from retained prediction node IDs and verified product topology. Original serialized label bytes/production environment remain unavailable and are not claimed recovered. |
| 5 | `pair.json` links one completed T-GRAANK/VAR pair: identical frozen input hashes, ordered graph selection, row counts, time windows, preprocessing and CPU environment. Each run includes complete command, config, predictions, runtime, scoring, source/scorer hashes and provenance. |
| 6 | This report describes the current partial O1 delivery and provides concrete commands/output; no external leaderboard, install-resolution-only, forecast, superiority, speedup or full acceptance claim substitutes for local evidence. |

From `benchmarks/causal_discovery/causalrivers`, the complete setup/validation commands actually executed were:

```powershell
../../.venv_geoai/Scripts/python.exe -m venv .venv_review
.venv_review/Scripts/python.exe -m pip install -r requirements.txt --report evidence/review-232/pip-install.json
.venv_review/Scripts/python.exe -m pip check
.venv_review/Scripts/python.exe -m pip freeze
.venv_review/Scripts/python.exe environment_probe.py
../../.venv_geoai/Scripts/python.exe verify_setup.py
.venv_review/Scripts/python.exe 0_generate_datasets.py which=confounder n_vars=3 save_path=evidence/review-232/generated/
.venv_review/Scripts/python.exe -m unittest discover -s tests -v
.venv_review/Scripts/python.exe run_pair.py
```

Actual outputs: successful installation; `No broken requirements found.`; all eleven imports passed; `mode: cpu`; archive SHA-256 `4f779fa9222653af092046eacba39a955cc65638bfb70fb7e31a32d9d8373b55`; all four selected products match archive members; confounder generation counts 24/11/1; `Ran 19 tests` / `OK`. Reinstall the selected environment with `python -m pip install -r evidence/review-232/requirements-resolved.txt`.

The two full method commands actually executed by `run_pair.py` are:

```powershell
.venv_review/Scripts/python.exe benchmark.py method=var label_path=evidence/review-232/frozen/confounder_3/east.p data_path=product/rivers_ts_east_germany.csv restrict_to=-1 dt_preprocess=True data_preprocess.resolution=6h data_preprocess.normalize=True data_preprocess.interpolate=True data_preprocess.subsample=1 data_preprocess.subset_year=False data_preprocess.subset_month=False data_preprocess.remove_trailing_nans_early=False method.max_lag=3 method.var_absolute_values=True method.map_to_summary_graph=max remove_diagonal=True save_full_out=True
```

```powershell
.venv_review/Scripts/python.exe benchmark.py method=tgraank label_path=evidence/review-232/frozen/confounder_3/east.p data_path=product/rivers_ts_east_germany.csv restrict_to=-1 dt_preprocess=True data_preprocess.resolution=6h data_preprocess.normalize=True data_preprocess.interpolate=True data_preprocess.subsample=1 data_preprocess.subset_year=False data_preprocess.subset_month=False data_preprocess.remove_trailing_nans_early=False method.max_lag=3 method.var_absolute_values=True method.map_to_summary_graph=max remove_diagonal=True save_full_out=True
```

| Method | Pooled AUROC | Mean per graph AUROC | Whole command runtime (seconds) | Result directory |
|---|---:|---:|---:|---|
| var | 0.6471076093717603 | 0.7089120370370371 | 5.068525 | `results/var_confounder_3/20261012182059` |
| tgraank | 0.6093717603151566 | 0.6513310185185185 | 294.514426 | `results/tgraank_confounder_3/20261012182600` |

Runtime includes loading, preprocessing, method execution and scoring on this local CPU machine; it does not establish a qualified speedup. AUROC aggregation types are distinct.

Historical recovery command, from the parent repository root:

```powershell
benchmarks/causal_discovery/causalrivers/.venv_review/Scripts/python.exe scripts/reconstruct_causalrivers_labels.py
```

The reconstructed labels have SHA-256 `00a400cf01ab4bec46727c17d2cff13e58541638fc9a9ee531fdd5572b37ccd9`; all 2,790 prediction row/column node identities and order are validated against the product graph. The original random label file's recorded digest is not reproduced. Other random dataset digest mismatches remain listed in `input-audit.json`; this pair uses the matching confounder selection.

Checksum-aware rescoring commands, from the benchmark root:

```powershell
.venv_review/Scripts/python.exe -u rescore.py --run results/tgraank_1_random_5/20260921025013 --labels evidence/review-232/reconstructed/historical-original/east.p --output results/rescores/review-232-original-portable
.venv_review/Scripts/python.exe -u rescore.py --run results/tgraank_1_random_5/recovered-replacement-3025328c --labels evidence/review-232/reconstructed/preserved-replacement/east.p --output results/rescores/review-232-replacement-portable
```

| Predictions | Pooled AUROC | Mean per graph AUROC | Largest absolute metric/threshold difference from its stored scoring file |
|---|---:|---:|---:|
| Original | 0.6751973087091248 | 0.7342750171305081 | 0.02254128456115723 |
| Replacement | 0.6749177530064113 | 0.7340156985733362 | 0.1710000038146973 |

Per-metric differences are in each rescore directory’s `score_differences.csv`. The replacement-minus-original pooled AUROC difference is -0.0002795557027134299; mean-per-graph difference is -0.0002593185571719836. The vectorized accuracy calculation preserves the reviewed threshold grid; `scorer-equivalence.json` records exact agreement with the reviewed scalar implementation on 18 representative cases.
