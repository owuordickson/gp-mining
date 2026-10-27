---
title: Causal Discovery Using FTGPs
---

# Causal Discovery Using FTGPs

Fuzzy temporal gradual patterns (FTGPs) combine **how attributes change together** with **when those relationships occur**. Under a temporal structural causal model, each variable depends on past parent variables and disturbances. A pattern such as

$$
X_{t-\tau}^{+}\rightarrow Y_t^{+}
$$

is a candidate positive relationship from past X to current Y. Temporal precedence and co-variation provide causal hypotheses; interpretation also requires assumptions about confounding, temporal order and the selected lags.

## 1. Create a controlled example

With SO4GP installed, run the Python blocks in order. Here, `Y[t] = 1.5 * X[t-1] + noise`: the causal link is known because we generate it. There is no reverse link in this model. The observations are daily and the random seed makes the example reproducible.

```python
import numpy as np
import pandas as pd
from so4gp.algorithms import TGRAANK

rng = np.random.default_rng(7)
n = 80
x = rng.normal(size=n)
y = np.empty(n)
y[0] = rng.normal()
y[1:] = 1.5 * x[:-1] + 0.05 * rng.normal(size=n - 1)

data = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
    "X": x,
    "Y": y,
})
```
## 2. Discover and read candidate relationships

Use `TGRAANK` to compute the temporal dependency matrix:

```python
miner = TGRAANK(data, device="cpu")
scores = miner.discover_causal_relationships(max_lag=2)
print(scores.round(3).to_string())
```
**Rows are effects; columns are causes.** Expected weights for this example are:

| Effect / Cause | X | Y |
|---|---:|---:|
| X | 0.000 | 0.545 |
| Y | 0.986 | 0.000 |

Read `scores.loc["Y", "X"]` as the candidate `X -> Y` direction. It is stronger than the reverse in this example. A nonzero weight is not proof of an edge, and these support-derived weights are not causal probabilities or significance levels.

For this example, `max_lag=2` visits the one-row shift in the current implementation. With daily data this is one day. The matrix summarizes candidate strengths; individual FTGP queries provide signs and estimated lags.

## 3. Use your own data

Replace the example DataFrame with your date column and numerical attributes, then rerun the discovery step:

```python
raw = pd.read_csv("my_timeseries.csv")
raw["Date"] = pd.to_datetime(raw["Date"])
data = raw.sort_values("Date")[["Date", "temperature", "flow"]].reset_index(drop=True)
```

Resolve missing values before mining and choose a lag search appropriate to the data and plausible mechanisms. Row shifts represent equal elapsed times only on regularly sampled data. Account for shared drivers and seasonality before interpreting candidate directions causally.

For sign/lag inspection, a real Deposium GeoAI case study with its saved tests and cross-validation evidence, and more detailed interpretation, see [causal_discovery.ipynb](https://github.com/owuordickson/gp-mining/blob/main/examples/causal_discovery.ipynb).

## Archived benchmark results

The following CausalRivers results reproduce the [saved benchmark report](https://github.com/owuordickson/gp-mining/blob/main/benchmarks/benchmarking-results.md), updated 26 October 2026. The metric is **mean per graph AUROC**, excluding autoregressive diagonal entries; higher values indicate better graph-edge ranking on that setting.

| Graph set | TGRAANK | PCMCI | PC | VARLiNGAM | DYNOTEARS |
|---|---:|---:|---:|---:|---:|
| Close 3 | 0.677891 | 0.639 | 0.501 | 0.786 | 0.501 |
| Close 5 | 0.654067 | 0.617 | 0.514 | 0.773 | 0.499 |
| Root cause 3 | 0.700800 | 0.696 | 0.501 | 0.772 | 0.503 |
| Root cause 5 | 0.704923 | 0.737 | 0.517 | 0.769 | 0.564 |
| Random+1 3 | 0.828725 | 0.831 | 0.498 | 0.840 | 0.519 |
| Random+1 5* | 0.734016 | 0.735 | 0.513 | 0.788 | 0.608 |
| Confounder 3 | 0.651331 | 0.663 | 0.490 | 0.675 | 0.525 |
| Confounder 5 | 0.586420 | 0.640 | 0.514 | 0.698 | 0.532 |
| Random 3 | 0.669843 | 0.655 | 0.501 | 0.787 | 0.503 |
| Random 5 | 0.640697 | 0.653 | 0.506 | 0.755 | 0.608 |
| Disjoint 10 | 0.815098 | 0.802 | Unavailable | 0.827 | 0.606 |

*Random+1-5 retains the saved TGRAANK value `0.734016`; the restored original artifact gives `0.734275`. The report documents this difference and all scoring sources. PC has no reported Disjoint-10 result.

These are archived comparisons, not new runs. Comparator preprocessing and search budgets have not been established as identical across the panel. Use the table as descriptive benchmark evidence, separate from the controlled example and Deposium's cross-validation results.
