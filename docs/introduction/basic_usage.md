---
layout: "contents"
firstpage:
---

# Quick Start

A **gradual item (GI)** pairs an attribute with a direction, such as increasing `Age` (`Age+`) or decreasing `Salary` (`Salary-`). A **gradual pattern (GP)** combines GIs whose directions hold together across pairs of observations. Its **support** is the fraction of all unordered observation pairs that satisfy those directions in a consistent orientation.

## Illustrative Example
Consider a dataset containing 6 objects across 3 features (`Age`, `Salary`, and `Cars`):

| Object | Age | Salary | Cars |
|:---:|:---:|:---:|:---:|
| o<sub>1</sub> | 23 | 52,000 | 0 |
| o<sub>2</sub> | 27 | 51,000 | 1 |
| o<sub>3</sub> | 31 | 50,000 | 1 |
| o<sub>4</sub> | 36 | 48,000 | 1 |
| o<sub>5</sub> | 40 | 47,000 | 2 |
| o<sub>6</sub> | 40 | 45,000 | 2 |

An extracted GP might take the following form:

> **{ Age<sup>+</sup>, Salary<sup>-</sup> }** &nbsp;&nbsp;&nbsp;&nbsp; `[Support ≈ 0.933]`

This pattern holds for **14 of the 15 unordered observation pairs** (approximately 93.3%): orienting each pair toward increasing `Age` also gives decreasing `Salary`.

### Checking the support

There are `6 × 5 / 2 = 15` unordered pairs, including nonadjacent observations. For example, both o<sub>1</sub> → o<sub>2</sub> and o<sub>1</sub> → o<sub>6</sub> satisfy the pattern. The only exception is o<sub>5</sub> → o<sub>6</sub>: `Age` stays at 40 while `Salary` decreases. With strict comparisons (`eq=False`, the default), this tie does not satisfy `Age+`, so the support is `14 / 15 ≈ 0.933`.


## Installation

`so4gp` requires **Python 3.14 or later** and is available on **PyPI**. Install the latest stable release using:

```shell
python -m pip install so4gp
```

You can verify the installation by importing the package:

```python
import so4gp
```

---

## Basic Usage

### Importing Algorithms

The `so4gp` package exposes its public algorithms through the
`so4gp.algorithms` module.

```python
from so4gp.algorithms import GRAANK, TGRAANK
```

Alternatively, you can import the package namespace directly:

```python
import so4gp as sgp
```

---

### Input Data

Most `so4gp` algorithms accept either:

- a `pandas.DataFrame`
- the path to a CSV file containing numerical data

Datasets should contain numerical attributes. For temporal gradual pattern mining (`TGRAANK`), the dataset should additionally contain a timestamp or ordered temporal attribute.

---

#### Example 01: Gradual Pattern Mining

The example below extracts gradual patterns using the classical APRIORI-based
GRAANK algorithm.

```python
import pandas as pd
from so4gp.algorithms import GRAANK

df = pd.DataFrame(
    [
        [30, 3, 1, 10],
        [35, 2, 2, 8],
        [40, 4, 2, 7],
        [50, 1, 1, 6],
        [52, 7, 1, 2],
    ],
    columns=["Age", "Salary", "Cars", "Expenses"],
)

miner_gp = GRAANK(
    data_source=df,
    min_sup=0.5,
)

results = miner_gp.discover()

print(results)
```

The `discover()` method returns a JSON-formatted string containing the mined
gradual patterns, their support values, and additional metadata.
Here, `min_sup=0.5` retains patterns supported by at least half of all unordered observation pairs.

- **Choosing a Mining Algorithm:**

    `GRAANK` supports multiple search strategies through the `discover()` method.

    ```python
    results = miner_gp.discover(search_type="ga")
    ```

    Supported search algorithms include:
    
    | Algorithm | Description |
    |-----------|-------------|
    | `apriori` | Classical exhaustive level-wise search |
    | `ga` | Genetic Algorithm |
    | `aco` | Ant Colony Optimization |
    | `pso` | Particle Swarm Optimization |
    | `hc` | Hill Climbing |
    | `random` | Random Search |

---

#### Example 02: Temporal Gradual Pattern Mining

The following example discovers fuzzy temporal gradual patterns using the
default Average Mutual Information (AMI) transformation algorithm.

```python
import pandas as pd
from so4gp.algorithms import TGRAANK

df = pd.DataFrame(
    [
        ["2021-03", 30, 3, 1, 10],
        ["2021-04", 35, 2, 2, 8],
        ["2021-05", 40, 4, 2, 7],
        ["2021-06", 50, 1, 1, 6],
        ["2021-07", 52, 7, 1, 2],
    ],
    columns=["Date", "Age", "Salary", "Cars", "Expenses"],
)

miner_tgp = TGRAANK(
    data_source=df,
    min_sup=0.5,
    min_rep=0.5,
)

results = miner_tgp.discover(target_col=1)

print(results)
```

`target_col=1` selects `Age` using zero-based column positions: `Date` is column 0 and `Age` is column 1. The target remains the reference attribute while the other attributes are shifted. `min_rep=0.5` sets the minimum fraction of rows retained after a temporal shift; larger shifts leave fewer aligned observations. The AMI transformation estimates which shifts to use.

---

`TGRAANK` supports multiple temporal transformation algorithms.

```python
results = miner_tgp.discover(target_col=1, transformations="ami")
```

Supported transformation algorithms include:

| Algorithm | Description |
|-----------|-------------|
| `ami` | Average Mutual Information-based transformation (recommended) |
| `all` | Classical TGrad fuzzy temporal transformation |

---

Use `TGRAANK.discover_causal_relationships()` to score candidate temporal causal relationships:

```python
res_df = miner_tgp.discover_causal_relationships(max_lag=3)
```

These scores require causal assumptions and validation. See the [causal discovery tutorial](../tutorials/causal_discovery.md) for a controlled example and guidance on interpretation.

## Learn More

Continue with a practical tutorial:

- [Feature selection using GPs](../tutorials/feature_selection.md)
- [Studying joint effects with GP descriptors](../tutorials/timeseries_analysis.md)
- [Causal discovery using FTGPs](../tutorials/causal_discovery.md)

Use the [Algorithm Reference](../api/algorithms.rst), [Class Reference](../api/classes.rst), and [Function Reference](../api/functions.rst) for parameter and return-value documentation. More detailed workflows are available in the [example notebooks](https://github.com/owuordickson/gp-mining/tree/main/examples).

## References
* Owuor, D., Runkler T., Laurent A., Menya E., Orero J (2021), Ant Colony Optimization for Mining Gradual Patterns. International Journal of Machine Learning and Cybernetics. [https://doi.org/10.1007/s13042-021-01390-w](https://doi.org/10.1007/s13042-021-01390-w)
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. [https://doi.org/10.1109/FUZZ-IEEE.2019.8858883](https://doi.org/10.1109/FUZZ-IEEE.2019.8858883)
* Laurent A., Lesot MJ., Rifqi M. (2009) GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In: Andreasen T., Yager R.R., Bulskov H., Christiansen H., Larsen H.L. (eds) Flexible Query Answering Systems. FQAS 2009. Lecture Notes in Computer Science, vol 5822. Springer, Berlin, Heidelberg. [https://doi.org/10.1007/978-3-642-04957-6_33](https://doi.org/10.1007/978-3-642-04957-6_33)
