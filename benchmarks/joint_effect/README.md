# Joint-effect benchmark

This benchmark measures whether composed gradual patterns (GPs) and their gradual
concordance set (GCS) descriptors help classify multivariate sequences. Each
case is an entire sequence. The target is its independent class label.

## Run from the repository root

Use the repository's Python environment with an editable install of GP-Mining.
The runner uses NumPy, scikit-learn, and the descriptor implementation on this
branch. Dataset access uses the Python standard library.

```powershell
python -m pip install -e .
python -m benchmarks.joint_effect uea --download
python -m benchmarks.joint_effect synthetic
```

The UEA run uses the published train and test files for BasicMotions,
RacketSports, and NATOPS. No fork is needed. To run offline, place the official
ZIP for each dataset or its two .ts files under benchmarks/joint_effect/data.
Use --data-dir for another location and --datasets to select a subset.

- [UEA archive paper](https://arxiv.org/abs/1811.00075)
- [Multiverse expansion](https://arxiv.org/abs/2603.20352)
- [BasicMotions](https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions)
- [RacketSports](https://www.timeseriesclassification.com/description.php?Dataset=RacketSports)
- [NATOPS](https://www.timeseriesclassification.com/description.php?Dataset=NATOPS)

For a small local check:

```powershell
python -m benchmarks.joint_effect synthetic --seeds 0 --noise 0.5 --train-size 32 --test-size 24 --max-patterns 8 --folds 2 --bootstrap 100
```

The synthetic experiment uses independent train and test sequences for each
seed and noise level. The default sweep uses five seeds and three noise levels.

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

Train-only mining ranks signed two-channel pattern families by mean full support
over training cases. Each retained family contributes both inverse directions,
so --max-patterns must be even. --min-support filters mean training support.
Full support uses both packed bitmap orientations. Directional density and
the GCS descriptors use the upper triangle in the original sequence order.

| Method | Features |
| --- | --- |
| standard | Channel means, standard deviations, slopes, and Pearson correlations |
| gi | Both directions of each GI, represented by density, mean gap, connectivity, singularity, and dispersion |
| gp | GI features plus full support and directional density of the selected GPs |
| gcs | The same GP features plus GP connectivity, singularity, and dispersion |
| gcs_no_connectivity | GCS features with only the GP connectivity block removed |
| gcs_no_singularity | GCS features with only the GP singularity block removed |
| gcs_no_dispersion | GCS features with only the GP dispersion block removed |

For an empty concordance set, connectivity is the observation count. Undefined
separation and singularity statistics use zero in the feature matrix, and
density records the empty set. The implementation calls
GP.gen_gradual_concordance_set and GP.compute_descriptors for the descriptors.

Every method uses the same stratified training folds and regularization grid
for standardized logistic regression. Each fold mines its vocabulary on its
own training cases. The final vocabulary is mined on the full training split
and held fixed on test. Feature standardization is fitted inside each model's
training pipeline. Validation balanced accuracy selects regularization.

## Reports

Each run writes to a new directory under benchmarks/joint_effect/results by
default. Pass --output to choose a new directory.

- metrics.csv gives held-out balanced accuracy, macro-F1, 95% stratified
  bootstrap intervals, and model tuning, fit, and prediction times.
- paired_contrasts.csv uses the same bootstrap cases for each method pair.
  The primary contrast is gcs minus gp.
- summary.csv and paired_summary.csv give means and sample standard deviations
  across independent runs at each dataset or noise level. A single run has no
  across-run standard deviation.
- run.json records dataset dimensions, final and fold-local pattern
  vocabularies, mining and feature extraction times, software versions, and
  settings. Feature extraction is shared across the compared methods.
  Intervals for the archive describe held-out cases conditional on the
  published training split.

Classification measures the value of the representation for recognizing
multivariate behavior. CAMELS-US is a separate optional environmental case
study with its own streamflow label and time-block protocol.

Run the focused checks with:

```powershell
python -m pytest benchmarks/joint_effect/tests -q
```

## Reference runs

The [UEA metrics](results/uea-20261109T133818769079Z/metrics.csv) and
[paired intervals](results/uea-20261109T133818769079Z/paired_contrasts.csv)
use the published splits and the default protocol. Scores are percentages.
Differences and paired 95% bootstrap intervals are in percentage points.

| Dataset | Standard BA | GP BA | GCS BA | GCS - GP BA (95% interval) | GP macro-F1 | GCS macro-F1 | Elapsed (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BasicMotions | 97.5 | 92.5 | 95.0 | +2.5 [-5.0, +10.0] | 92.3 | 95.0 | 27.7 |
| RacketSports | 77.5 | 78.2 | 84.2 | +5.9 [+0.5, +11.3] | 78.4 | 84.0 | 16.4 |
| NATOPS | 90.6 | 68.3 | 70.0 | +1.7 [-1.1, +4.4] | 67.6 | 69.6 | 101.9 |

The [synthetic summaries](results/synthetic-full/summary.csv) and
[paired summaries](results/synthetic-full/paired_summary.csv) cover five
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
