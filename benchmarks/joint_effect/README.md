# Joint-effect benchmark

This benchmark tests whether gradual patterns (GPs) and descriptors of their
gradual concordance sets (GCSs) help a classifier recognize predefined classes
in unseen recordings. Each recording becomes one row in a table. The **features** are
numbers calculated from the sensor readings. The **label** is the activity
supplied by the dataset.

## Worked example with RacketSports

A smartwatch worn on the player's dominant wrist recorded a stroke for three
seconds. Each recording contains 30 readings from six channels, with three
axes of acceleration and three axes of rotation rate. The archive supplies one
of four labels for the whole recording, `Badminton_Clear`, `Badminton_Smash`,
`Squash_BackhandBoast`, or `Squash_ForehandBoast`.
[UEA RacketSports dataset](https://www.timeseriesclassification.com/description.php?Dataset=RacketSports).

### Original recording

The first recording in the published test split has the label
`Badminton_Smash`. That single label applies to all 30 readings. The
table shows the original archive values for two of those readings.

| Reading | Accel. X | Accel. Y | Accel. Z | Gyro. X | Gyro. Y | Gyro. Z |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.787509 | -0.813344 | 5.804565 | -2.540857 | 0.077238 | -1.054696 |
| 16 | 14.509458 | -11.269689 | 11.749872 | -14.949511 | -5.276141 | 2.476937 |

<details>
<summary>All 30 readings from this recording</summary>

| Reading | Accel. X | Accel. Y | Accel. Z | Gyro. X | Gyro. Y | Gyro. Z |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6661 | -0.558194 | -2.323993 | -0.058594 | -0.822982 | -0.194426 |
| 2 | -0.549034 | -0.354419 | -2.772675 | -1.558073 | 0.058594 | -2.040144 |
| 3 | 1.787509 | -0.813344 | 5.804565 | -2.540857 | 0.077238 | -1.054696 |
| 4 | 1.787509 | -0.813344 | 5.804565 | -2.540857 | 0.077238 | -1.054696 |
| 5 | 4.933549 | 1.774982 | 6.955224 | -1.099973 | 1.651291 | -0.050604 |
| 6 | 0.059652 | -7.810628 | 3.018279 | 2.27452 | 0.609912 | -0.705794 |
| 7 | 2.509221 | -4.748552 | -6.000353 | 1.478172 | 1.368974 | -1.837727 |
| 8 | -1.225921 | -0.8191 | -3.01334 | -1.424904 | 1.408924 | -0.660516 |
| 9 | 0.066535 | -2.882862 | 3.637775 | 2.034817 | 1.376964 | -0.67117 |
| 10 | 3.043326 | -4.53172 | 3.547236 | 2.215926 | 0.980121 | -1.360983 |
| 11 | 3.927304 | -3.852172 | 5.227589 | 3.41977 | 1.102636 | -1.693905 |
| 12 | 5.936533 | -7.790691 | -11.003212 | -1.241132 | 6.314857 | -2.026827 |
| 13 | 6.169348 | 2.209995 | -20.295305 | -10.66148 | 10.922491 | 4.389238 |
| 14 | 22.8746 | -17.673197 | -13.550385 | -15.183887 | -5.902034 | -12.744238 |
| 15 | 23.352537 | 23.981478 | 11.664442 | 34.871536 | -9.889102 | -15.375649 |
| 16 | 14.509458 | -11.269689 | 11.749872 | -14.949511 | -5.276141 | 2.476937 |
| 17 | 12.477253 | 11.081854 | 4.704073 | 2.170649 | -0.335585 | 0.034624 |
| 18 | 7.187539 | 12.789593 | -2.402317 | -0.689814 | 0.359555 | 0.103872 |
| 19 | -1.773086 | 3.603779 | -4.403036 | -0.596595 | 0.697804 | 1.667271 |
| 20 | -3.327036 | 3.079627 | -5.912917 | 2.285174 | 1.036052 | 3.118809 |
| 21 | -1.743321 | -3.054036 | -1.313099 | 2.71664 | 1.984213 | 2.229243 |
| 22 | 3.34082 | -2.211271 | 2.846099 | 1.661945 | 0.985448 | 0.748408 |
| 23 | 3.322938 | 2.191704 | 0.022982 | 1.001428 | 0.279654 | 1.078666 |
| 24 | 5.100965 | -0.55755 | -3.837673 | 1.502142 | 0.564635 | 1.060022 |
| 25 | 3.872064 | 1.730436 | -1.444935 | 1.323696 | 0.308951 | 0.119852 |
| 26 | 4.015299 | -0.727289 | 1.474412 | 1.235805 | -0.780368 | 0.545991 |
| 27 | 1.813128 | 0.99165 | -1.31174 | 1.827074 | 0.029297 | 0.65519 |
| 28 | 2.506096 | -1.619691 | -3.268069 | 0.724437 | 0.141159 | 0.652526 |
| 29 | 2.88671 | -1.948051 | -1.902785 | 0.396843 | 0.833636 | 0.202416 |
| 30 | -0.828796 | -2.499378 | -2.126675 | 0.657853 | 0.359555 | 0.24503 |

</details>

### Encoded recording

A pattern mined from this recording says
**Z acceleration increases while Y rotation rate decreases**. From reading 3
to reading 16 in this recording, Z acceleration rises from 5.80 to 11.75 and Y
rotation rate falls from 0.08 to −5.28. The pair `(3, 16)` therefore supports
the pattern.

The benchmark compares every earlier reading with every later one in that
recording. Reading 1 gives 29 pairs, reading 2 gives 28, and so on. This gives
30 × 29 / 2 = 435 possible pairs. The pattern holds for 122 of them. Those
122 pairs form its GCS and give a directional density of 122 / 435 ≈ 0.28.

The benchmark turns the set of pairs into numbers. Connectivity counts the
groups of readings linked by the pairs, including isolated readings.
Singularity is the largest fraction of pairs that touch any one reading.
Rank dispersion measures how much the gaps between paired readings vary.
Average separation is the mean gap, measured in observation steps.
Full support also counts pairs that match when read in reverse, giving
263 / 435 ≈ 0.6046.

With six channels, the `gcs` method reserves 30 two-item pattern slots,
giving 150 feature columns. Each slot contributes five descriptors. Unused slots contain
zeros. This table shows the five GCS values for the illustrated pattern, with
the recording's original label alongside them.

| Directional density | Connected groups | Singularity | Rank dispersion | Average separation | Known label |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.2805 | 1 | 0.1393 | 33.82 | 9.57 | `Badminton_Smash` |

Each recording supplies its own GRAANK results in the miner's output order.
A slot can therefore describe different channel combinations in different
recordings. The classifier learns from these descriptor vectors and the known
stroke labels. For a test recording, it predicts one stroke label and the
benchmark compares that prediction with the label supplied by the archive.


## Run from an IDE

Open [benchmark.py](benchmark.py) and click **Run**. With no arguments, it runs
the UEA benchmark on **ERing** using the cached archive in `uea/data`. Edit
`RUN_EXPERIMENT` to select `uea` or `synthetic`, and edit `RUN_OPTIONS` to change
the dataset or other settings. If your Run configuration supplies arguments,
the runner uses those arguments. Direct launches load SO4GP from this checkout.

## Run from the repository root

Use the repository's Python environment with an editable install of GP-Mining.
The runner uses the shared feature extraction and evaluation code in `uea/tools`.
Dataset access uses the Python standard library.

```powershell
python -m pip install -e .
python benchmarks/joint_effect/benchmark.py uea --download
python benchmarks/joint_effect/benchmark.py synthetic
```

The UEA benchmark uses the existing published train and test splits for the 18
datasets in the saved runs. The load_splits function preserves the
recording membership and labels supplied in the TRAIN and TEST files. To run offline, place the official
ZIP for each dataset or its two `.ts` files under `benchmarks/joint_effect/uea/data`.
Use `--data-dir` for another location and `--datasets` to select a subset.

- [UEA archive paper](https://arxiv.org/abs/1811.00075)
- [Multiverse expansion](https://arxiv.org/abs/2603.20352)
- [BasicMotions](https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions)
- [RacketSports](https://www.timeseriesclassification.com/description.php?Dataset=RacketSports)
- [NATOPS](https://www.timeseriesclassification.com/description.php?Dataset=NATOPS)

For a small local check:

```powershell
python benchmarks/joint_effect/benchmark.py synthetic --seeds 0 --noise 0.5 --train-size 32 --test-size 24 --max-patterns 8 --folds 2 --bootstrap 100
```

The synthetic generate_splits function generates recordings directly into
the configured training and test sets for each seed and noise level. The
existing train_size and test_size settings establish the split sizes, and
independent child seeds supply their recordings and labels. The default
sweep uses five seeds and three noise levels.

- **Shared**. Three of six channels share a changing latent signal inside a
  random interval in the positive class. The negative class has independent
  channels with the same per-channel process and interval distribution.
  Measurement noise and sparse outliers affect both classes.
- **Timing**. Both classes contain equal numbers of observations from a shared
  state and an independent state. Labels distinguish a long shared episode from
  repeated short episodes. Stable within-state order and palindromic layouts
  preserve full support and directional density when the same underlying rows
  are arranged in either layout. Sequences in the actual train and test splits
  use independent draws. Bounded perturbations preserve cross-state ordering.

## Protocol

The shared build_features function mines each recording independently using
SO4GP's GRAANK, stopping at two-item patterns. The --min-support option is
passed to the miner for each recording. The current encoder reserves all
two-item pattern slots: num_channels * (num_channels - 1). The legacy
--max-patterns argument is still accepted but does not limit these slots.

The output matrices represent separate inputs to matched classifiers.

| Method | Features |
| --- | --- |
| baseline | All original sequence values, flattened into one row per recording in observation/channel order. |
| baseline_gcs | Raw baseline columns followed by the GCS descriptor columns. |
| baseline_standard | Raw baseline columns followed by the standard summary columns. |
| standard | Channel means, population standard deviations, linear trends, and Pearson correlations between distinct channels. |
| gi | Density, connectivity, singularity, rank dispersion, and average separation for each signed channel in fixed channel/sign positions. |
| gp | Support and confidence for each retained GRAANK pattern. |
| gcs | Density, connectivity, singularity, rank dispersion, and average separation for each retained GRAANK pattern. |
| gcs_no_density | The GCS matrix with density removed from every pattern slot. |
| gcs_no_connectivity | The GCS matrix with connectivity removed from every pattern slot. |
| gcs_no_singularity | The GCS matrix with singularity removed from every pattern slot. |
| gcs_no_dispersion | The GCS matrix with rank dispersion removed from every pattern slot. |
| gcs_no_avg_separation | The GCS matrix with average separation removed from every pattern slot. |

For a recording with 30 readings and six channels, baseline supplies 180
raw features. baseline_gcs adds 150 GCS columns, giving 330 features;
baseline_standard adds 33 summary columns, giving 213 features. Each
method keeps the same single activity label and uses the same classifier,
training-only scaling, and tuning. Combined methods reuse the extracted
features without additional mining.

GI columns retain channel identities when pruning removes an item. GP and
GCS blocks use the miner's result order independently for each recording.
Pruned GI slots and unused GP/GCS slots contain zeros. For a retained item
whose canonical concordance set is empty, density is zero and connectivity
counts all observations as isolated components. Undefined statistics become
zero in the feature matrices. SO4GP computes mining scores and descriptors.

The classifier is **regularized logistic regression**
(`sklearn.linear_model.LogisticRegression`). The benchmark trains a separate
model for each feature set and standardizes the feature columns with
`StandardScaler`, fitted on training data. The models use the same stratified
folds and try the same regularization values (`C` = 0.1, 1, and 10 by default).
Each model selects its `C` by validation balanced accuracy. The requested fold
count is capped at the size of the smallest training class, so Handwriting
uses two folds with the default settings. Stratified CV requires at least two
recordings per training class. Features are built independently for each
recording. Each fold fits its scaler and classifier on the fold's training
features, and final models use all training features before scoring test cases.

## Reports

By default, each run writes to a new directory under `uea/results/` or
`synthetic/results/` within this benchmark. Use `--output` to choose another
destination. The saved reference scores and links to their detailed reports
are collected in [benchmarking-results.json](benchmarking-results.json).

- metrics.csv gives held-out balanced accuracy, macro-F1, 95% stratified
  bootstrap intervals, and model tuning, fit, and prediction times.
- paired_contrasts.csv uses the same bootstrap cases for each method pair.
  The primary contrast is gcs minus gp. Each combined baseline also has
  a paired comparison against baseline to measure the added features.
- summary.csv and paired_summary.csv give means and sample standard deviations
  across independent runs at each dataset or noise level. A single run has no
  across-run standard deviation.
- run.json records the BenchmarkConfig settings under config, dataset
  dimensions, effective fold counts, software versions, and fold-level and
  final extraction timings. The baseline stage covers reshaping raw values;
  the standard stage covers summary features.
  The gradual_patterns stage covers GRAANK mining and GI/GP/GCS extraction;
  feature_combinations covers joining the existing columns.
  Intervals for the archive describe held-out cases conditional on the
  published training split.

Classification measures the value of the representation for recognizing
multivariate behavior. CAMELS-US is a separate optional environmental case
study with its own streamflow label and time-block protocol.

Run the tests for both suites from the repository root.

```powershell
python -m pytest --import-mode=importlib benchmarks/joint_effect -q
```

## UEA results and analysis

**BA means balanced accuracy.** Compute the fraction of correctly classified
recordings in each class, then average across classes. If four classes have
recalls of 90%, 80%, 70%, and 60%, BA is 75%. Macro-F1 similarly gives every
class equal weight when averaging F1. Higher is better for both measures.
A change from 70% to 75% is **five percentage points (pp)**.

### Saved runs and statistical scope

All four runs use the same 18 published train/test splits, seed 0, support
cutoff 0.0, and logistic regression. They use the same three-value C grid,
up to three stratified training folds, and 1,000 test-case bootstrap draws.
Handwriting uses two folds because its smallest training class has two cases.

For c channels, P = c(c - 1) is the maximum number of distinct two-item
patterns after merging globally reversed directions. K denotes allocated
pattern slots: GP has 2K columns and GCS has 5K. Pruning can leave unused
slots. The recorded feature widths confirm the budgets below; the shared
manifest value max_patterns=24 does not describe the effective historical caps.

| Run | Pattern slots K | Purpose | Reports |
| --- | ---: | --- | --- |
| uea-half | P/2 | Half the two-item pattern slots | [Scores](uea/results/uea-half/metrics.csv), [paired intervals](uea/results/uea-half/paired_contrasts.csv), [settings](uea/results/uea-half/run.json) |
| uea-full | P | All possible two-item slots | [Scores](uea/results/uea-full/metrics.csv), [paired intervals](uea/results/uea-full/paired_contrasts.csv), [settings](uea/results/uea-full/run.json) |
| uea-extra | P + 100 | Full slots plus unused padding | [Scores](uea/results/uea-extra/metrics.csv), [paired intervals](uea/results/uea-extra/paired_contrasts.csv), [settings](uea/results/uea-extra/run.json) |
| uea-baseline | P | Raw values and augmented raw-value models | [Scores](uea/results/uea-baseline/metrics.csv), [paired intervals](uea/results/uea-baseline/paired_contrasts.csv), [settings](uea/results/uea-baseline/run.json) |

Every aggregate below gives each dataset equal weight. Across-dataset tests
use a two-sided exact Wilcoxon signed-rank sign-permutation calculation, with
Holm correction across **16 BA comparisons** at 0.05. None passes that
correction. This follows the paired comparison approach in
[Demšar (2006)](https://jmlr.org/papers/v7/demsar06a.html).
The [analysis summary](uea/results/analysis-summary.json) records exact and
adjusted p-values, per-dataset effects, and hashes of the input reports.

The saved 95% paired bootstrap intervals describe held-out cases conditional
on the fitted models. They are unadjusted per-dataset intervals and do not
measure variation from retraining on new splits or seeds. Wins/ties/losses
always compare the first named method with the second on BA.

### Full: GI, GP, GCS, and Standard

| Method | Mean BA (%) | Mean macro-F1 (%) | Median BA (%) | Mean rank |
| --- | ---: | ---: | ---: | ---: |
| gi | 59.67 | 58.40 | 62.54 | 2.94 |
| gp | 60.45 | 58.12 | 63.18 | 2.86 |
| gcs | 62.29 | 60.40 | 61.46 | 2.31 |
| standard | 68.02 | 66.26 | 72.00 | 1.89 |

Among the three gradual representations, GCS has the highest mean BA and
macro-F1 and the best average rank. Its BA is higher than GI on 12 datasets,
tied on two, and lower on four: **+2.63 pp on average**. Against GP, the counts
are **10/2/6**, with **+1.84 pp BA** and **+2.28 pp macro-F1**. These gains
are heterogeneous and are not significant across datasets after correction.
GP has the higher median BA, so GCS does not lead on every summary.

Standard has the highest overall means. GCS versus Standard gives
**5 wins, 3 ties, and 10 losses**, averaging **-5.73 pp BA** and
**-5.86 pp macro-F1**. GCS nevertheless improves BA on RacketSports (+5.47),
ERing (+14.07), ArticularyWordRecognition (+2.00), Handwriting (+9.96), and
IRDS-EFL (+19.05).

**GP versus GCS is retained as a representation comparison.** Here GP uses
support/confidence and GCS uses five descriptors. GCS is not GP with extra
columns appended. A direct test of added value beyond GP would require a
gp_gcs method containing both blocks, compared against GP on the same patterns.

#### Eleven illustrative datasets

These 11 cases have GCS BA above GP or Standard in uea-full. They were
selected after inspecting results and are descriptive examples, not an
unbiased 11-dataset benchmark. All means, ranks, tests, and win counts above
and below retain all 18 datasets. BA is in percent; difference intervals
come directly from the saved paired bootstrap results.

| Dataset | GI BA | GP BA | GCS BA | Standard BA | GCS - GP (pp, 95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RacketSports | 81.02 | 74.23 | 82.98 | 77.50 | +8.74 [+2.09, +16.00] |
| NATOPS | 62.22 | 73.33 | 84.44 | 90.56 | +11.11 [+3.89, +17.79] |
| AtrialFibrillation | 26.67 | 20.00 | 33.33 | 33.33 | +13.33 [-13.33, +33.33] |
| ERing | 80.74 | 70.74 | 87.78 | 73.70 | +17.04 [+11.48, +22.59] |
| ArticularyWordRecognition | 93.00 | 92.67 | 95.33 | 93.33 | +2.67 [-0.67, +5.33] |
| Libras | 67.78 | 42.22 | 51.11 | 69.44 | +8.89 [+0.00, +18.33] |
| LSST | 17.23 | 20.56 | 23.14 | 31.33 | +2.58 [+0.85, +4.43] |
| PenDigits | 75.60 | 38.52 | 66.31 | 73.41 | +27.79 [+26.03, +29.66] |
| Handwriting | 20.24 | 17.15 | 21.45 | 11.49 | +4.30 [+1.71, +6.60] |
| IRDS-EFL | 67.26 | 84.52 | 73.21 | 54.17 | -11.31 [-25.60, +0.00] |
| JapaneseVowels_eq | 75.73 | 75.79 | 79.56 | 96.72 | +3.77 [-0.21, +7.93] |

<details>
<summary>The other seven datasets, included in every aggregate and test</summary>

| Dataset | GI BA | GP BA | GCS BA | Standard BA | GCS - GP (pp, 95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BasicMotions | 52.50 | 95.00 | 87.50 | 97.50 | -7.50 [-15.00, +0.00] |
| Epilepsy | 62.85 | 55.44 | 30.16 | 92.21 | -25.27 [-35.51, -15.50] |
| FingerMovements | 53.08 | 60.18 | 49.12 | 55.12 | -11.06 [-22.17, -0.12] |
| AppliancesEnergy_disc | 55.51 | 66.18 | 56.62 | 70.59 | -9.56 [-25.39, +4.41] |
| EyesOpenShut | 54.76 | 57.14 | 54.76 | 59.52 | -2.38 [-16.67, +11.90] |
| SPHERE-WUS | 50.00 | 50.00 | 50.00 | 50.00 | +0.00 [+0.00, +0.00] |
| UIPRMD-SSE-C | 77.78 | 94.44 | 94.44 | 94.44 | +0.00 [-16.67, +16.67] |

</details>

The GCS-minus-GP 95% interval is entirely above zero on RacketSports, NATOPS,
ERing, LSST, PenDigits, and Handwriting, and entirely below zero on Epilepsy
and FingerMovements. These are unadjusted, split-specific intervals.
The large loss on Epilepsy (-25.27 pp versus GP and -62.05 pp versus Standard)
is part of the overall result.

### Pattern-budget ablation: half, full, and extra

| Method | Half BA (%) | Full BA (%) | Extra BA (%) | Full - half (pp) | Extra - full (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| gp | 54.83 | 60.45 | 58.63 | +5.62 | -1.82 |
| gcs | 57.64 | 62.29 | 62.30 | +4.66 | +0.00 |

| Comparison | BA wins/ties/losses | Exact p | Holm p |
| --- | ---: | ---: | ---: |
| Full - half, GP | 11/3/4 | 0.035 | 0.565 |
| Full - half, GCS | 11/1/6 | 0.080 | 1.000 |
| Extra - full, GP | 5/4/9 | 0.042 | 0.628 |
| Extra - full, GCS | 1/17/0 | 1.000 | 1.000 |

Half reduces mean BA, but these results do not establish a significant
overall drop after multiple-comparison correction. For GCS, full improves
mean BA by 4.66 pp, with 11 wins, one tie, and six losses. The nominal GP
p-value of 0.035 does not survive correction; its Holm-adjusted value is 0.565.

For GCS, extra and full have identical BA on 17 of 18 datasets. Only LSST
changes (+0.05291 pp), giving a mean change of just **+0.00294 pp**. Thus no
practical gain is observed from extra slots. This is a padding check: extra
adds 100 unused slots, not additional valid two-item or higher-order patterns.
It does not establish that P is the best budget under every selection rule.

GP changes on 14 datasets between full and extra, despite those extra slots
carrying no new pattern information. Standard and GI remain identical.
The baseline run also reproduces every full GCS BA score while GP differs
on 13 datasets. These GP results therefore cannot isolate a padding effect.

A small current-source check confirms that changing Python's hash seed can
select a different default reference GI: the same pattern has support 0.6
but confidence 0.6 or 1.0. This demonstrates a reproducibility risk; the
archived scores alone cannot establish its contribution to each difference.

Freeze the mined feature matrices, default reference choice, candidate
ordering, folds, and hash seed before using this comparison as a clean
capacity ablation. The saved manifests do not identify the historical
hash seed or source snapshot.

### GCS descriptor ablations in full

Positive differences mean that retaining the descriptor helps. Negative
differences mean that removing it improves the aggregate score. CI counts
refer to saved unadjusted per-dataset 95% intervals for full minus ablated GCS.

| Removed descriptor | BA without (%) | Full - without BA (pp) | Full - without F1 (pp) | Wins/ties/losses | CI above/below 0 | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Density | 61.27 | +1.03 | +0.81 | 12/1/5 | 4/1 | 1.000 |
| Connectivity | 60.61 | +1.68 | +0.95 | 10/2/6 | 4/1 | 1.000 |
| Singularity | 63.07 | -0.78 | -1.01 | 6/6/6 | 1/1 | 1.000 |
| Rank dispersion | 62.62 | -0.32 | -1.21 | 6/4/8 | 3/0 | 1.000 |
| Average separation | 62.00 | +0.29 | +0.37 | 7/4/7 | 3/0 | 1.000 |

Connectivity and density have the largest positive mean BA contributions.
Average separation adds a smaller positive mean. Removing singularity or
rank dispersion improves mean BA, although dispersion helps particular
datasets, including ERing, LSST, and PenDigits. No descriptor is uniformly
helpful, and no across-dataset ablation passes the Holm correction.
These observations support dataset-dependent descriptor value, rather than
a claim that all five descriptors are individually necessary.

### Raw logistic regression with added features

All three methods use the same classifier, folds, scaling procedure, and
tuning grid. baseline receives flattened raw sequences; baseline_gcs and
baseline_standard append their respective computed features.

| Inputs to logistic regression | Mean BA (%) | Mean macro-F1 (%) | BA gain over raw (pp) | Wins/ties/losses | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 69.72 | 67.77 | 0.00 | -- | -- |
| baseline_gcs | 70.53 | 68.44 | +0.81 | 10/2/6 | 1.000 |
| baseline_standard | 71.47 | 70.47 | +1.74 | 9/5/4 | 1.000 |

GCS helps 10 datasets, but the average improvement is only **+0.81 pp BA**.
Its unadjusted paired interval is above zero on BasicMotions, RacketSports,
LSST, PenDigits, Handwriting, and IRDS-EFL, and below zero on UIPRMD-SSE-C.
The mean gain is not significant across datasets after correction.

Standard features give the larger mean gain: **+1.74 pp BA** and
**+2.70 pp macro-F1**. baseline_gcs versus baseline_standard gives 8 wins,
two ties, and eight losses, with a mean BA difference of **-0.93 pp**.
Neither augmentation is an overall winner supported by the adjusted tests.

For example, BasicMotions rises from 77.50% raw BA to 90.00% with GCS,
whereas adding Standard keeps it at 77.50%. In IRDS-EFL, raw BA is 54.17%,
GCS raises it to 78.57%, and Standard raises it further to 88.10%.
There are also losses: SPHERE-WUS falls from 75.00% raw BA to 50.00% with GCS.

### Cost and analyses still needed

The saved classifier times include tuning, fitting, and prediction only.
Feature extraction is additional; the shared gradual_patterns timing mixes
GI, GP, and GCS work and cannot identify their individual extraction costs.
Timings are descriptive single-run measurements, not a controlled speed comparison.

| Representation | Mean feature columns | Mean classifier time (s) |
| --- | ---: | ---: |
| standard | 449.22 | 0.32 |
| gp | 1576.22 | 0.56 |
| gcs | 3940.56 | 1.11 |
| baseline | 1763.11 | 0.83 |
| baseline_gcs | 5703.67 | 1.60 |
| baseline_standard | 2212.33 | 0.92 |

The remaining priorities for a journal evaluation are:

- **An incremental GP test:** compare GP with GP + GCS on identical mined
  patterns. Add a density-only control to distinguish the structural
  descriptors from ordinary agreement strength.
- **A controlled budget study:** reuse one full feature extraction and keep
  pattern order fixed for several fractions of P. This removes the GP
  reproducibility issue and tests real truncation rather than unused padding.
- **Repeated training and uncertainty:** repeat fold/tuning seeds and report
  macro-F1 and per-class errors alongside BA. Save predictions to permit
  paired comparisons between any methods. Keep the published test labels
  out of pattern selection, hyperparameter selection, and dataset selection.
- **Feature count and resource controls:** compare matched feature budgets
  or random-feature controls, and measure extraction time and peak memory.
  Adding many more columns can change regularized logistic regression even
  when the classifier and C grid stay the same.
- **A defensible dataset subset:** for an 11-dataset primary benchmark,
  select by declared domain, size, or channel criteria before observing test
  scores. Keep the current favorable 11 as illustrative cases and retain
  the other seven in the complete report.

These experiments measure predictive usefulness for predefined labels.
They do not estimate physical interaction strengths or reconstruct signals.
The supported practical conclusion is that GCS offers improvements on
specific datasets and can complement raw values, with heterogeneous effects
and no adjusted evidence of overall superiority in these 18 datasets.

## Synthetic reference results

The [summaries](synthetic/results/synthetic-full/summary.csv) and [paired summaries](synthetic/results/synthetic-full/paired_summary.csv) cover 30 runs: two regimes,
three noise levels, and five seeds. Each run uses 120 training and 80 test
recordings, a support cutoff of 0.2, and at most 24 GRAANK patterns per
recording. Python's hash seed was fixed at 0; the [manifest](synthetic/results/synthetic-full/run.json)
records the settings and source hashes. These saved runs predate the raw-value
baseline and the change to all two-item pattern slots.

BA columns are means across five seeds. Difference columns give the paired
mean +/- sample standard deviation in percentage points, rather than confidence intervals.

| Regime | Noise | Standard BA (%) | GI BA (%) | GP BA (%) | GCS BA (%) | GCS - GP BA (pp) | GCS - GP macro-F1 (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 0.0 | 80.25 | 50.75 | 83.50 | 87.50 | +4.00 +/- 3.35 | +3.99 +/- 3.36 |
| shared | 0.5 | 76.25 | 48.75 | 76.50 | 81.00 | +4.50 +/- 4.01 | +4.49 +/- 3.96 |
| shared | 1.0 | 68.25 | 49.00 | 69.25 | 69.25 | +0.00 +/- 5.08 | -0.01 +/- 5.00 |
| timing | 0.0 | 51.00 | 73.75 | 47.75 | 97.25 | +49.50 +/- 8.08 | +49.53 +/- 8.07 |
| timing | 0.5 | 49.75 | 72.75 | 51.25 | 98.00 | +46.75 +/- 3.14 | +46.87 +/- 3.16 |
| timing | 1.0 | 50.00 | 75.25 | 53.00 | 99.50 | +46.50 +/- 3.35 | +46.94 +/- 3.14 |

GCS has higher mean BA than GP in five settings and ties it in the sixth;
it exceeds Standard in all six. Gains over GP are modest and vary across
seeds for shared signals, but reach 46.50--49.50 percentage points for timing.
The GI baseline also captures some timing information because the layouts
change individual-channel order.

Removing average separation lowers mean BA by 0.50--4.00 points across the
six settings. Removing rank dispersion lowers timing BA by 16.25--28.00
points. Other ablations sometimes improve scores, so every descriptor does
not help equally. These results support GCS for these synthetic tasks;
performance on other datasets requires separate evaluation.
