# Joint-effect benchmark

This benchmark tests whether gradual patterns (GPs) and descriptors of their
gradual concordance sets (GCSs) help a classifier recognize activities in unseen
recordings. Each recording becomes one row in a table. The **features** are
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

One pattern selected from the training recordings says
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
Full support also counts pairs that match when read in reverse, giving
263 / 435 ≈ 0.6046.

The encoded recording has 180 numeric features in the `gcs` method. The
table shows the five columns for this one pattern. The label stays in a
separate column and keeps its original value.

| Full support | Directional density | Connected groups | Singularity | Rank dispersion | Known label |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.6046 | 0.2805 | 1 | 0.1393 | 33.82 | `Badminton_Smash` |

Each selected pattern contributes its own columns. The classifier learns from
these numeric features and the known stroke labels in the training recordings.
For a test recording, it receives the feature values, predicts a stroke, and
the benchmark compares that prediction with the label supplied by the archive.

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

The UEA benchmark uses published train and test splits for the 11
datasets in the reference runs. To run offline, place the official
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

The synthetic suite generates independent train and test sequences in memory
for each seed and noise level. The default sweep uses five seeds and three
noise levels.

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

SO4GP's GRAANK runs its Apriori search on each training recording, stopping
at two-channel patterns. The search considers every family with positive
full support in that recording. The benchmark ranks the discovered families
by their mean full support across all training recordings, including zero
for recordings in which a family was absent. It retains both inverse forms
of each selected family, such as `A+ B-` and `A- B+`. Set an even
`--max-patterns` and use `--min-support` to filter by mean training support
(default 0.2). SO4GP computes full support from both orientations of the packed
bitmap. Directional density and the structural descriptors use earlier-to-later
pairs.

**Method** names a way to represent a recording. **Features** lists the input
columns for that method's classifier.

| Method | Features |
| --- | --- |
| `standard` | The mean, standard deviation, and overall linear trend of each channel, plus the Pearson correlation between each pair of channels at matching times. |
| `gi` | Each channel's increase and decrease described separately by density, mean gap, connectivity, singularity, and rank dispersion. |
| `gp` | The `gi` columns plus full support and directional density for each selected two-channel GP. |
| `gcs` | The `gp` columns plus connectivity, singularity, and rank dispersion for each selected two-channel GP. |
| `gcs_no_connectivity` | The `gcs` columns with GP connectivity removed. |
| `gcs_no_singularity` | The `gcs` columns with GP singularity removed. |
| `gcs_no_dispersion` | The `gcs` columns with GP rank dispersion removed. |

For an empty concordance set, connectivity is the observation count. Undefined
separation and singularity statistics use zero in the feature matrix, and
density records the empty set. The benchmark uses SO4GP to build packed
bitmaps, intersect them, extract concordance sets, and compute support,
density, and the structural descriptors.

The classifier is **regularized logistic regression**
(`sklearn.linear_model.LogisticRegression`). The benchmark trains a separate
model for each feature set and standardizes the feature columns with
`StandardScaler`, fitted on training data. The models use the same stratified
folds and try the same regularization values (`C` = 0.1, 1, and 10 by default).
Each model selects its `C` by validation balanced accuracy. The requested fold
count is capped at the size of the smallest training class, so Handwriting
uses two folds with the default settings. Stratified CV requires at least two
recordings per training class. Each fold selects patterns using its own training
cases. The final patterns are selected on the full training split and applied to
every test recording.

## Reports

By default, each run writes to a new directory under `uea/results/` or
`synthetic/results/` within this benchmark. Use `--output` to choose another
destination. The saved reference scores and links to their detailed reports
are collected in [benchmarking-results.json](benchmarking-results.json).

- metrics.csv gives held-out balanced accuracy, macro-F1, 95% stratified
  bootstrap intervals, and model tuning, fit, and prediction times.
- paired_contrasts.csv uses the same bootstrap cases for each method pair.
  The primary contrast is gcs minus gp.
- summary.csv and paired_summary.csv give means and sample standard deviations
  across independent runs at each dataset or noise level. A single run has no
  across-run standard deviation.
- run.json records dataset dimensions, the effective fold count per dataset,
  final and fold-local pattern vocabularies, mining and feature extraction times,
  software versions, and settings. Feature extraction is shared across the
  compared methods.
  GP timing covers support. GCS timing includes directional density, which
  both methods use.
  Intervals for the archive describe held-out cases conditional on the
  published training split.

Classification measures the value of the representation for recognizing
multivariate behavior. CAMELS-US is a separate optional environmental case
study with its own streamflow label and time-block protocol.

Run the tests for both suites from the repository root.

```powershell
python -m pytest --import-mode=importlib benchmarks/joint_effect -q
```

## Reference runs

The tables below report the earlier direct-search protocol, with a mean
training support cutoff of 0.4.

**BA means balanced accuracy.** For each class, divide the number of correctly
predicted recordings by the number of recordings with that true label. Average
the resulting percentages to give every class equal weight. For example, if
the four strokes are correctly classified 90%, 80%, 70%, and 60% of the time,
the balanced accuracy is 75%. Higher BA is better.

The UEA runs use the published splits and the same protocol for 11 datasets. Scores are percentages. Differences and paired 95% bootstrap intervals are in percentage points.

- 11 datasets. [Metrics](uea/results/uea-20261110T172511120121Z/metrics.csv) and [paired intervals](uea/results/uea-20261110T172511120121Z/paired_contrasts.csv).

| Dataset | Standard BA | GP BA | GCS BA | GCS - GP BA (95% interval) | GP macro-F1 | GCS macro-F1 | Elapsed (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BasicMotions | 97.5 | 92.5 | 95.0 | +2.5 [-5.0, +10.0] | 92.3 | 95.0 | 27.7 |
| RacketSports | 77.5 | 78.2 | 84.2 | +5.9 [+0.5, +11.3] | 78.4 | 84.0 | 16.4 |
| NATOPS | 90.6 | 68.3 | 70.0 | +1.7 [-1.1, +4.4] | 67.6 | 69.6 | 101.9 |
| ArticularyWordRecognition | 93.3 | 96.0 | 96.7 | +0.7 [-0.7, +2.0] | 95.9 | 96.6 | 249.0 |
| AtrialFibrillation | 33.3 | 33.3 | 13.3 | -20.0 [-40.0, +0.0] | 31.6 | 12.1 | 49.6 |
| ERing | 73.7 | 86.7 | 86.3 | -0.4 [-3.3, +2.6] | 86.7 | 86.4 | 7.6 |
| Epilepsy | 92.2 | 67.6 | 45.1 | -22.5 [-32.0, -12.9] | 67.5 | 45.0 | 182.5 |
| FingerMovements | 55.1 | 51.0 | 52.2 | +1.1 [-7.9, +10.2] | 51.0 | 51.7 | 100.1 |
| LSST | 31.3 | 23.9 | 23.8 | -0.1 [-2.8, +2.1] | 24.4 | 24.4 | 448.5 |
| Libras | 69.4 | 76.7 | 73.9 | -2.8 [-6.7, +1.1] | 75.9 | 73.1 | 13.8 |
| PenDigits | 73.4 | 78.0 | 81.8 | +3.8 [+2.8, +4.8] | 77.9 | 81.7 | 132.3 |

GCS scored higher than ordinary GP on 6 of 11 datasets and higher than the
standard baseline on 5 of 11. The ordinary GP comparison measures the
contribution of the added GCS descriptors with the selected patterns,
support, and density held fixed. The paired 95% interval for GCS minus GP
was entirely above zero on RacketSports and PenDigits. The paired 95%
interval was entirely below zero on Epilepsy.

The [synthetic summaries](synthetic/results/synthetic-full/summary.csv) and
[paired summaries](synthetic/results/synthetic-full/paired_summary.csv) cover five
seeds per regime and noise level. The paired columns report the mean and
sample standard deviation of the differences across seeds.

| Regime | Noise | GP BA (%) | GCS BA (%) | GCS - GP BA (pp, mean +/- SD) | GCS - GP macro-F1 (pp, mean +/- SD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared | 0.0 | 84.0 | 81.7 | -2.3 +/- 2.7 | -2.2 +/- 2.7 |
| shared | 0.5 | 78.5 | 79.8 | +1.2 +/- 5.1 | +1.3 +/- 5.2 |
| shared | 1.0 | 71.8 | 68.0 | -3.8 +/- 7.1 | -3.7 +/- 7.2 |
| timing | 0.0 | 66.5 | 100.0 | +33.5 +/- 10.7 | +33.8 +/- 10.7 |
| timing | 0.5 | 63.3 | 100.0 | +36.8 +/- 6.8 | +36.8 +/- 6.8 |
| timing | 1.0 | 56.5 | 100.0 | +43.5 +/- 9.0 | +43.8 +/- 9.4 |

The timing layouts also change univariate temporal order. The GI baseline
measures what the individual-item descriptors capture, and the ordinary GP
baseline controls for their contribution together with support and density.
