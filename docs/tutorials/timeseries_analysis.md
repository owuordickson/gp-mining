---
title: Studying Joint Effects with GP Descriptors
---

# Studying Joint Effects with GP Descriptors

A gradual item (GI) describes an attribute and a direction, such as `A+` for increasing values. A gradual pattern (GP) combines several GIs by retaining the observation pairs that satisfy all their directions. Comparing the individual GIs with their GP lets us study how attributes behave jointly.

The idea is to look for a more coherent, DTW-like progression of index relationships: **one connected component**, **low singularity**, and **low rank-gap variance**.

$$
\kappa_g \rightarrow 1,\qquad Si_g \rightarrow 0,\qquad \sigma_g^2 \rightarrow 0.
$$

| Descriptor | What it tells us |
|---|---|
| Connectivity | The number of connected components; the target is **1**. |
| Singularity | How unevenly observations participate in the relationships; smaller means more even participation. |
| Rank-gap variance | How much the absolute index distances vary; smaller means more consistent separations. |

These are structural indicators to examine together. They do not establish that the relationships form a valid DTW alignment.

## 1. Prepare a small example

With SO4GP installed, run the following Python blocks in order. The rows represent six observations already arranged in time order.

```python
import numpy as np
import pandas as pd
from so4gp import GP
from so4gp.algorithms import GRAANK

data = pd.DataFrame({
    "A": [0, 2, 1, 4, 3, 5],
    "B": [4, 5, 2, 3, 0, 1],
})

miner = GRAANK(data, min_sup=0.1, eq=False, device="cpu")
_ = miner.discover(compute_descriptors=True)
engine = miner.mining_engine
```

`eq=False` uses strict comparisons. For example, `A+` retains `(i, j)` when `A[j] > A[i]`. The low support threshold keeps this small example's joint pattern available.

## 2. Combine the GIs and compare their descriptors

Select `A+` and `B+` using their zero-based column keys, `0+` and `1+`. Their joint GP is the intersection of their pairwise matrices, computed with bitwise AND.

```python
keys = ("0+", "1+")
labels = [f"{data.columns[int(key[:-1])]}{key[-1]}" for key in keys]
matrices = {
    label: engine.valid_bins[key].packed_bin_mat
    for label, key in zip(labels, keys)
}
matrices["Joint GP"] = np.bitwise_and.reduce(list(matrices.values()))

rows = []
for label, matrix in matrices.items():
    pairs = np.asarray(
        GP.gen_gradual_warping_set(matrix, engine.row_count)
    ).reshape(-1, 2)
    gp = GP()
    if not gp.compute_descriptors(pairs, obj_count=engine.row_count):
        print(f"{label}: no pairs to describe")
        continue
    rows.append({
        "Pattern": label,
        "Pairs": len(pairs),
        "Connectivity": gp.graph_connectivity,
        "Singularity": gp.singularity_score,
        "Rank-gap variance": np.var(np.abs(pairs[:, 0] - pairs[:, 1])),
    })

result = pd.DataFrame(rows)
print(result.round(3).to_string(index=False))
```

SO4GP's `rank_dispersion` reports the standard deviation of the absolute index gaps. Here, `np.var` computes the variance directly to match the hypothesis's squared notation.

Expected results:

| Pattern | Pairs | Connectivity | Singularity | Rank-gap variance |
|---|---:|---:|---:|---:|
| A+ | 15 | 1 | 0.000 | 1.556 |
| B+ | 15 | 1 | 0.000 | 1.556 |
| Joint GP | 5 | 1 | 0.133 | 0.000 |

The joint pairs are `(0, 1), (2, 1), (2, 3), (4, 3), (4, 5)`. Every pair links neighboring observations, so the variance becomes zero while the observation graph remains connected. Singularity increases because the chain's endpoints have fewer neighbors than its interior observations. Composition improves the consistency of index gaps in this example, while participation becomes less uniform.

Connectivity counts only observations appearing in the pairs. When using your own data, also check which observations remain in the joint set.

## 3. Use your own data

Replace the example DataFrame with your numerical attributes, sorted by date:

```python
raw = pd.read_csv("my_timeseries.csv")
raw["Date"] = pd.to_datetime(raw["Date"])
data = raw.sort_values("Date")[["temperature", "humidity"]].reset_index(drop=True)
```

Rerun the mining and descriptor steps. Use `keys = ("0+", "1+")` to study increasing temperature with increasing humidity, or `keys = ("0+", "1-")` for increasing temperature with decreasing humidity. The keys refer to column positions in `data`; each selected key must be present in `engine.valid_bins`.

Handle missing values before mining and choose a support threshold appropriate to your data. Index gaps measure row separations; they represent equal time intervals only when observations are regularly spaced.

For plots, matched GI-to-GP comparisons, and the rainfall/EVI analysis, see [ftgp_composition.ipynb](https://github.com/owuordickson/gp-mining/blob/main/applications/ftgp_composition.ipynb).
