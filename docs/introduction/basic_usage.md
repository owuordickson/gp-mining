---
layout: "contents"
firstpage:
---


# Introduction

A **Gradual Pattern (GP)** is a co-occurring set of gradual items (GI) that captures covariations between attributes. 
A pattern's quality is measured quantitatively by its computed **support value**.

### Illustrative Example
Consider a dataset containing 6 objects across 3 features (`Age`, `Salary`, and `Cars`):

| Object ID | Age | Salary | Cars |
|:---------:|:---:|:---:|:---:|
|   $o_1$   | 23 | 52,000 | 0 |
|   $o_2$   | 27 | 51,000 | 1 |
|   $o_3$   | 31 | 50,000 | 1 |
|   $o_4$   | 36 | 48,000 | 1 |
|   $o_5$   | 40 | 47,000 | 2 |
|   $o_6$   | 40 | 45,000 | 2 |

An extracted GP might take the following form:

$$\{\text{Age}^+, \text{Salary}^-\} \quad [\text{Support} \approx 0.83]$$

This mathematical expression reveals that in **83.3% of the dataset** (5 out of 6 objects), a strict increase in `Age` ($^+$) 
strongly correlates with a simultaneous decrease in `Salary` ($^-$). 

#### Step-by-Step Validation:
* Comparing $o_1 \rightarrow o_2$: Age increases ($23 \rightarrow 27$), Salary decreases ($52\text{k} \rightarrow 51\text{k}$). **(Valid)**
* Comparing $o_2 \rightarrow o_3$: Age increases ($27 \rightarrow 31$), Salary decreases ($51\text{k} \rightarrow 50\text{k}$). **(Valid)**
* Comparing $o_3 \rightarrow o_4$: Age increases ($31 \rightarrow 36$), Salary decreases ($50\text{k} \rightarrow 48\text{k}$). **(Valid)**
* Comparing $o_4 \rightarrow o_5$: Age increases ($36 \rightarrow 40$), Salary decreases ($48\text{k} \rightarrow 47\text{k}$). **(Valid)**
* Comparing $o_5 \rightarrow o_6$: Age stays the same ($40 \rightarrow 40$), but Salary decreases ($47\text{k} \rightarrow 45\text{k}$). Depending on your variation definition (strict vs. non-strict inequality), this step sequence validates 5 out of 6 objects.



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
