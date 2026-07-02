---
layout: "contents"
firstpage:
---

# Introduction

A **Gradual Pattern (GP)** is a co-occurring set of gradual items (GI) that captures covariations between attributes. A pattern's quality is measured quantitatively by its computed **support value**.

### Illustrative Example
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

#### Step-by-Step Validation:
* Comparing o<sub>1</sub> → o<sub>2</sub>: Age increases (23 → 27), Salary decreases (52k → 51k). **(Valid)**
* Comparing o<sub>2</sub> → o<sub>3</sub>: Age increases (27 → 31), Salary decreases (51k → 50k). **(Valid)**
* Comparing o<sub>3</sub> → o<sub>4</sub>: Age increases (31 → 36), Salary decreases (50k → 48k). **(Valid)**
* Comparing o<sub>4</sub> → o<sub>5</sub>: Age increases (36 → 40), Salary decreases (48k → 47k). **(Valid)**
* Comparing o<sub>5</sub> → o<sub>6</sub>: Age stays the same (40 → 40), but Salary decreases (47k → 45k). Depending on your variation definition (strict vs. non-strict inequality), this step sequence validates 5 out of 6 objects.



## Installation
The library is available on **PyPI**. To install it, run the following command in your terminal:

```shell
pip install so4gp
```

## Basic Usage
After installing the ```so4gp``` package, you can import it as follows:

```{code-block} python
import so4gp as sgp
```

The ```sgp``` namespace contains all necessary classes, functions, and algorithms. Classes and functions are accessible via ```sgp.ClassName``` or ```sgp.function_name```.

To use the algorithms, import them via:

```{code-block python}
from so4gp.algorithms import GRAANK, AntGRAANK, GeneticGRAANK, ClusterGP, TGradAMI
```

The ```so4gp``` algorithms require a numeric dataset provided as either a ```pandas.DataFrame``` or a path to a ```CSV``` file.

All ```so4gp``` functions and classes are documented in the **API Section**.

## References
* Owuor, D., Runkler T., Laurent A., Menya E., Orero J (2021), Ant Colony Optimization for Mining Gradual Patterns. International Journal of Machine Learning and Cybernetics. [https://doi.org/10.1007/s13042-021-01390-w](https://doi.org/10.1007/s13042-021-01390-w)
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. [https://doi.org/10.1109/FUZZ-IEEE.2019.8858883](https://doi.org/10.1109/FUZZ-IEEE.2019.8858883)
* Laurent A., Lesot MJ., Rifqi M. (2009) GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In: Andreasen T., Yager R.R., Bulskov H., Christiansen H., Larsen H.L. (eds) Flexible Query Answering Systems. FQAS 2009. Lecture Notes in Computer Science, vol 5822. Springer, Berlin, Heidelberg. [https://doi.org/10.1007/978-3-642-04957-6_33](https://doi.org/10.1007/978-3-642-04957-6_33)
