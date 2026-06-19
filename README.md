[![Downloads](https://pepy.tech/badge/so4gp)](https://pepy.tech/project/so4gp) [![Downloads](https://pepy.tech/badge/so4gp/week)](https://pepy.tech/project/so4gp)
![Dependents](https://badgen.net/github/dependents-repo/owuordickson/sogp_pypi/?icon=github)
[![DOI](https://zenodo.org/badge/388183952.svg)](https://doi.org/10.5281/zenodo.16281808)
![Dependents](https://badgen.net/github/license/owuordickson/sogp_pypi/?icon=github)


**SO4GP** stands for: "Some Optimizations for Gradual Patterns". SO4GP applies optimizations such as swarm intelligence, HDF5 chunks, cluster analysis and many others in order to improve the efficiency of extracting gradual patterns. It provides Python algorithm implementations for these optimization techniques. The algorithm implementations include:

* (Classical) GRAANK algorithm for extracting GPs
* Ant Colony Optimization algorithm for extracting GPs
* Genetic Algorithm for extracting GPs
* Particle Swarm Optimization algorithm for extracting GPs
* Random Search algorithm for extracting GPs
* Local Search algorithm for extracting GPs
* Clustering-based algorithm for extracting GPs

A GP (Gradual Pattern) is a set of gradual items (GI) and its quality is measured by its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and column 'salary' decreasing.

## Installation

```shell
pip install so4gp
```

## Usage
In order to any algorithm for the purpose of extracting GPs, follow the instructions that follow.

First and foremost, import the **so4gp** python package via:

```python
import so4gp as sgp
# OR 
from so4gp.algorithms import GRAANK
```

### GRAdual rANKing Algorithm for GPs (GRAANK)

This is the classical approach (initially proposed by Anne Laurent) for mining gradual patterns. All the remaining algorithms are variants of this algorithm.

```python
import pandas as pd
from so4gp.algorithms import GRAANK

dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-04", 35, 2, 2, 8], ["2021-05", 40, 4, 2, 7], ["2021-06", 50, 1, 1, 6], ["2021-07", 52, 7, 1, 2]]
dummy_df = pd.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
    
mine_obj = GRAANK(data_source=dummy_df, min_sup=0.5, eq=False)
gp_json = mine_obj.discover()
print(gp_json)

```

where you specify the parameters as follows:

* **data_source** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **eq** - *[optional]* encode equal values as gradual ```default = False```


### Sample Output
The default output is the format of JSON:

```json
{
	"Algorithm": "GRAANK",
	"Best Patterns": [
            [["Age+", "Salary+"], 0.6], 
            [["Expenses-", "Age+", "Salary+"], 0.6]
	],
	"Iterations": 20
}
```

## Contributors ✨

Thanks go to these incredible people:

<a href="https://github.com/owuordickson/sogp_pypi/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=owuordickson/sogp_pypi" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## References
* Owuor, D., Runkler T., Laurent A., Menya E., Orero J (2021), Ant Colony Optimization for Mining Gradual Patterns. International Journal of Machine Learning and Cybernetics. https://doi.org/10.1007/s13042-021-01390-w
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. https://doi.org/10.1109/FUZZ-IEEE.2019.8858883.
* Laurent A., Lesot MJ., Rifqi M. (2009) GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In: Andreasen T., Yager R.R., Bulskov H., Christiansen H., Larsen H.L. (eds) Flexible Query Answering Systems. FQAS 2009. Lecture Notes in Computer Science, vol 5822. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-04957-6_33


**See Docs for more details**
