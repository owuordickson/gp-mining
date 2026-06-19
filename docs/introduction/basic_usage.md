---
layout: "contents"
firstpage:
---

# Introduction


A GP (Gradual Pattern) is a set of gradual items (GI), and its quality is measured by its computed support value. For example, given a data set with 3 features (age, salary, cars) and 6 objects. A GP may take the form: {age+, salary-} with a support of 0.83. This implies that 5 out of 6 objects have the values of **age** *'increasing'* and **salary** *'decreasing'*.

<p align="center">

| Age | Salary | Cars |
|-----|------|--------|
| 23  | 52000 | 0 |
| 27  | 51000 | 1 |
| 31  | 50000 | 1 |
| 36  | 48000 | 1 |
| 40  | 47000 | 2 |
| 40  | 45000 | 2 |

</p>



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
