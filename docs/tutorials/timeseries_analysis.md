---
title: Studying Joint Effects with GP Descriptors
---

# Analyzing Joint Effects with GPs

A gradual pattern (GP) retains observation pairs that satisfy every constituent gradual item (GI). The gradual concordance set contains the supporting pairs $(i,j)$ with $i<j$ in the given observation order. Compare the GP with each constituent to describe how composition changes the retained pairs.

| Descriptor | Calculation |
|---|---|
| Density | Number of concordance pairs divided by the number of possible unordered observation pairs. |
| Mean index separation | Mean of the gaps $j-i$. |
| Rank dispersion | Population variance of the gaps $j-i$. |
| Connectivity | Number of connected components of the undirected graph on all observations, including isolated ones. |
| Singularity | Largest vertex degree divided by the number of concordance pairs. |

The full packed bitmap retains both orientations for pattern support. Descriptor calculations use the upper-triangular concordance set.

## Compare individual items and their joint pattern

The rows represent six observations in time order.

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
engine = miner.mining_engine
engine.fit_bitmap()

matrices = {
    label: engine.valid_bins[key].packed_bin_mat
    for label, key in [("A+", "0+"), ("B+", "1+")]
}
matrices["Joint GP"] = np.bitwise_and.reduce(list(matrices.values()))

rows = []
for label, matrix in matrices.items():
    pairs = GP.gen_gradual_concordance_set(matrix, engine.attr_size)
    gp = GP()
    gp.compute_descriptors(pairs, obj_count=engine.attr_size)
    rows.append({
        "Pattern": label,
        "Pairs": len(pairs),
        "Density": gp.density,
        "Connectivity": gp.graph_connectivity,
        "Singularity": gp.singularity_score,
        "Index-gap variance": gp.rank_dispersion,
    })

print(pd.DataFrame(rows).round(3).to_string(index=False))
```

| Pattern | Pairs | Density | Connectivity | Singularity | Index-gap variance |
|---|---:|---:|---:|---:|---:|
| A+ | 13 | 0.867 | 1 | 0.385 | 1.479 |
| B+ | 3 | 0.200 | 3 | 0.333 | 0.000 |
| Joint GP | 3 | 0.200 | 3 | 0.333 | 0.000 |

The joint concordance set contains $(0,1)$, $(2,3)$, and $(4,5)$. Each pair connects two neighboring observations, yielding three components and zero gap variance. The joint set equals the concordance set of B+, so their descriptors agree.

An empty concordance set has zero density and one component per observation. Its mean separation, dispersion, and singularity are undefined and appear as `NaN`.

## Use your own data

Sort observations by time and select the numerical attributes. The keys `0+` and `1-` refer to the first attribute increasing and the second decreasing. Index gaps count observation steps and represent equal elapsed times when sampling is regular.

For plots and a rainfall and EVI case study, see [the composition notebook](https://github.com/owuordickson/gp-mining/blob/main/examples/ftgp_composition.ipynb).
