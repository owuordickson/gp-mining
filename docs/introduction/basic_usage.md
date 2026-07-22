---
layout: "contents"
firstpage:
---

# Introduction

A **Gradual Pattern (GP)** is a co-occurring set of gradual items (GI) that captures covariations between attributes. A pattern's quality is measured quantitatively by its computed **support value**.

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

> **{ Age<sup>+</sup>, Salary<sup>-</sup> }** &nbsp;&nbsp;&nbsp;&nbsp; `[Support ≈ 0.83]`

This expression reveals that in **83.3% of the dataset** (5 out of 6 objects), a strict increase in `Age` (<sup>+</sup>) strongly correlates with a simultaneous decrease in `Salary` (<sup>-</sup>). 

### Step-by-Step Validation:
* Comparing o<sub>1</sub> → o<sub>2</sub>: Age increases (23 → 27), Salary decreases (52k → 51k). **(Valid)**
* Comparing o<sub>2</sub> → o<sub>3</sub>: Age increases (27 → 31), Salary decreases (51k → 50k). **(Valid)**
* Comparing o<sub>3</sub> → o<sub>4</sub>: Age increases (31 → 36), Salary decreases (50k → 48k). **(Valid)**
* Comparing o<sub>4</sub> → o<sub>5</sub>: Age increases (36 → 40), Salary decreases (48k → 47k). **(Valid)**
* Comparing o<sub>5</sub> → o<sub>6</sub>: Age stays the same (40 → 40), but Salary decreases (47k → 45k). Depending on your variation definition (strict vs. non-strict inequality), this step sequence validates 5 out of 6 objects.



# Quick Start

## Installation

`so4gp` is available on **PyPI**.

Install the latest stable release using:

```shell
pip install so4gp
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

Datasets should contain numerical attributes. For temporal gradual pattern
mining (`TGRAANK`), the dataset should additionally contain a timestamp or
ordered temporal attribute.

---

#### Example: Gradual Pattern Mining

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

miner = GRAANK(
    data_source=df,
    min_sup=0.5,
)

results = miner.discover()

print(results)
```

The `discover()` method returns a JSON-formatted string containing the mined
gradual patterns, their support values, and additional metadata.

---

#### Example: Temporal Gradual Pattern Mining

The following example discovers fuzzy temporal gradual patterns using the
default Mutual Information (AMI) transformation algorithm.

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

miner = TGRAANK(
    data_source=df,
    target_col=1,
    min_sup=0.5,
    min_rep=0.5,
)

results = miner.discover()

print(results)
```

---

## Choosing a Mining Algorithm

`GRAANK` supports multiple search strategies through the `discover()` method.

```python
results = miner.discover(search_type="ga")
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

Similarly, `TGRAANK` supports multiple temporal transformation algorithms.

```python
results = miner.discover(transformations="ami")
```

Supported transformation algorithms include:

| Algorithm | Description |
|-----------|-------------|
| `ami` | Average Mutual Information-based transformation (recommended) |
| `all` | Classical TGrad fuzzy temporal transformation |

---

## Learn More

The complete documentation includes:

- **Quick Start** — installation and introductory examples.
- **Algorithm Reference** — algorithm descriptions and usage.
- **API Reference** — complete documentation of all public classes, methods, and functions.
- **Examples** — practical mining workflows and advanced use cases.

Refer to the **Algorithm Reference** for detailed descriptions of each mining algorithm and the **API Reference** for complete parameter documentation.

## References
* Owuor, D., Runkler T., Laurent A., Menya E., Orero J (2021), Ant Colony Optimization for Mining Gradual Patterns. International Journal of Machine Learning and Cybernetics. [https://doi.org/10.1007/s13042-021-01390-w](https://doi.org/10.1007/s13042-021-01390-w)
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. [https://doi.org/10.1109/FUZZ-IEEE.2019.8858883](https://doi.org/10.1109/FUZZ-IEEE.2019.8858883)
* Laurent A., Lesot MJ., Rifqi M. (2009) GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In: Andreasen T., Yager R.R., Bulskov H., Christiansen H., Larsen H.L. (eds) Flexible Query Answering Systems. FQAS 2009. Lecture Notes in Computer Science, vol 5822. Springer, Berlin, Heidelberg. [https://doi.org/10.1007/978-3-642-04957-6_33](https://doi.org/10.1007/978-3-642-04957-6_33)
