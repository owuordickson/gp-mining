---
title: Joint-Effect Features with GRAANK
---

# Joint-effect features with GRAANK

A gradual item (GI) describes one attribute increasing or decreasing. A gradual
pattern (GP) combines items: `0+ 1+` means the first two columns increase
together; `0+ 1-` means the first increases while the second decreases.

## 1. Compare individual and joint changes

With SO4GP installed, run these examples in order.

```python
import numpy as np
import pandas as pd
from so4gp.algorithms import GRAANK

# Six ordered observations; columns 0 and 1 are A and B.
data = np.array([
    [0, 4], [2, 5], [1, 2],
    [4, 3], [3, 0], [5, 1],
], dtype=float)

items = GRAANK.build_gradual_features(data, pattern_level=1)
joint = GRAANK.build_gradual_features(data, pattern_level=2)

comparison = pd.concat([items.loc[["0+", "1+"]], joint.loc[["0+ 1+"]]])
print(comparison.round(3))
```

Each row describes a pattern using numerical column identifiers. Here, `1+` and
`0+ 1+` retain the same three forward observation pairs, so their structural
descriptors agree. Support can differ from density.

| Column | Meaning |
| --- | --- |
| `support` | SO4GP's existing pair-support score. |
| `density` | Fraction of possible forward pairs satisfying the pattern. |
| `connectivity` | Number of linked groups, including isolated observations. |
| `avg_separation` | Mean gap between paired observation indices. |
| `rank_dispersion` | Variance of those gaps. |
| `singularity` | Largest fraction of pairs that share one observation. |

## 2. Encode several recordings

```python
recordings = np.stack([data, data[::-1]])
features = GRAANK.build_gradual_features(recordings, max_patterns=2)

print(features.shape)  # (2, 12)
print(features["0+ 1+"].round(3))
```

A 3D input has shape `(recordings, observations, channels)`. The result has
**one row per recording** and `(pattern, descriptor)` columns: two patterns
give 12 features. Comparisons stay within each recording. Keep the dataset's
classification labels separately, aligned with row order.

## Options and your own data

- `pattern_level=1` describes individual GIs; `2` describes pairs; `3`
  describes triples when at least three channels are available.
- `max_patterns=None` includes all combinations. A positive cap takes the
  first patterns in deterministic order; it does not select by support.
  Zero-support patterns are included.
- GIs keep both signs. Multi-item patterns keep the lowest-numbered
  participating column positive, representing each globally reversed pair once.

Sort observations by time and select finite numerical columns, excluding dates
and labels. Index gaps measure observation steps, not necessarily elapsed time.
For empty sets, `avg_separation`, `rank_dispersion`, and `singularity` are
`NaN` in 2D tables and zero in 3D features. Connectivity still counts isolated
observations.

See the [worked notebook](https://github.com/owuordickson/gp-mining/blob/main/examples/joint_effect.ipynb)
for classification, plots, and the rainfall/EVI example.
