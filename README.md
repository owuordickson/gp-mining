
<div align="center">

<strong>Python implementation of Gradual Pattern (GP) mining algorithms</strong>

[![PyPI][pypi-badge]][pypi-url]
[![GNU licensed][gnu-badge]][gnu-url]
[![Python Version][python-badge]][python-url]
[![Documentation][docs-badge]][docs-url]

</div>

<div align="center">

[![Downloads][downloads-badge]][downloads-url] 
[![Downloads][weekly-downloads-badge]][weekly-downloads-url] 
![Dependents][dependents-badge]
[![DOI][doi-badge]][doi-url]

</div>

**SO4GP** is a high-performance Python library designed to optimize the extraction of gradual patterns from large-scale 
datasets. By integrating advanced computation techniques and data management strategies, the library significantly 
reduces processing time and memory overhead during knowledge discovery. 

## Implemented Extraction Algorithms
The library provides native Python implementations for the following core and meta-heuristic gradual pattern mining algorithms:

* **GRAANK**: The foundational classical approach for mining gradual patterns.
* **Ant Colony Optimization (AntGRAANK)**: Meta-heuristic swarm optimization for search-space pruning.
* **Genetic Algorithm (GeneticGRAANK)**: Evolutionary search strategy for optimized pattern extraction.
* **Particle Swarm Optimization (ParticleGRAANK)**: Swarm intelligence framework for fast convergence.
* **Random Search (HillClimbingGRAANK)**: Baseline stochastic search variant.
* **Clustering-based Mining (ClusterGP)**: Data partitioning to accelerate pattern discovery.

### What are Gradual Patterns?
A **Gradual Pattern (GP)** is a co-occurring set of **gradual items (GI)** that captures covariations between attributes. 
A pattern's quality is measured quantitatively by its computed **support value**.

#### Example
Consider a dataset containing 10 objects with 3 attributes: `age`, `salary`, and `cars`. An extracted GP might look like:

$$\{\text{age}^+, \text{salary}^-\} \quad [\text{Support} = 0.8]$$

This output explicitly reveals that in **80% of the dataset** (8 out of 10 objects), an increase in `age` ($^+$) strongly 
correlates with a simultaneous decrease in `salary` ($^-$).


## Installation

```shell
pip install so4gp
```

## Usage
To use any algorithm to mine GPs, follow the instructions that follow.

First and foremost, import the **so4gp** python package via:

```python
import so4gp as sgp
# OR 
from so4gp.algorithms import GRAANKAlg
```

### GRAdual rANKing Algorithm for GPs (GRAANK)

This is the classical approach (initially proposed by Anne Laurent) for mining gradual patterns. All the remaining algorithms 
are variants of this algorithm.

```python
import pandas as pd
from so4gp.algorithms import GRAANKAlg

dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-04", 35, 2, 2, 8], ["2021-05", 40, 4, 2, 7], ["2021-06", 50, 1, 1, 6],
              ["2021-07", 52, 7, 1, 2]]
dummy_df = pd.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])

mine_obj = GRAANKAlg(data_source=dummy_df, min_sup=0.5, eq=False)
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

<a href="https://github.com/owuordickson/gp-mining/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=owuordickson/gp-mining" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## References
* Owuor, D., Runkler T., Laurent A., Menya E., Orero J (2021), Ant Colony Optimization for Mining Gradual Patterns. International Journal of Machine Learning and Cybernetics. https://doi.org/10.1007/s13042-021-01390-w
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. https://doi.org/10.1109/FUZZ-IEEE.2019.8858883.
* Laurent A., Lesot MJ., Rifqi M. (2009) GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In: Andreasen T., Yager R.R., Bulskov H., Christiansen H., Larsen H.L. (eds) Flexible Query Answering Systems. FQAS 2009. Lecture Notes in Computer Science, vol 5822. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-04957-6_33


**See Docs for more details**

[pypi-badge]: https://img.shields.io/pypi/v/so4gp.svg
[pypi-url]: https://pypi.org/project/so4gp/
[gnu-badge]: https://img.shields.io/pypi/l/so4gp.svg
[gnu-url]: https://github.com/owuordickson/gp-mining/blob/main/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/so4gp.svg
[python-url]: https://www.python.org/downloads/
[docs-badge]: https://img.shields.io/badge/docs-so4gp-blue.svg
[docs-url]: http://so4gp.readthedocs.io

[downloads-badge]: https://pepy.tech/badge/so4gp
[downloads-url]: https://pepy.tech/project/so4gp
[weekly-downloads-badge]: https://pepy.tech/badge/so4gp/week
[weekly-downloads-url]: https://pepy.tech/project/so4gp
[dependents-badge]: https://badgen.net/github/dependents-repo/owuordickson/gp-mining/?icon=github
[dependents-url]: https://github.com/owuordickson/gp-mining/network/dependents
[doi-badge]: https://zenodo.org/badge/388183952.svg
[doi-url]: https://doi.org/10.5281/zenodo.16281808

