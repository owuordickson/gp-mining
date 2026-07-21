---
hide-toc: true
firstpage:
lastpage:
---

```{figure} _static/img/logo_ol-g.png
:alt: SO4GP Logo
:height: 110px
:align: center
```

# SO4GP

**SO4GP** (**S**ome **O**ptimizations for **G**radual **P**atterns) is an open-source Python library for discovering **Gradual Patterns (GPs)** and **Fuzzy Temporal Gradual Patterns (FTGPs)** from numerical datasets.

The library combines classical gradual pattern mining algorithms with modern optimization techniques to accelerate pattern discovery while reducing computational cost. It also provides feature selection, clustering, and temporal data transformation algorithms for knowledge discovery from large-scale datasets.

---

## Key Features

- 🚀 High-performance gradual pattern mining
- 📈 Temporal gradual pattern mining
- 🧬 Metaheuristic optimization algorithms
- 🎯 Gradual pattern-based feature selection
- 🔍 Pattern descriptors and statistical analysis
- 📊 Support for CSV files and Pandas DataFrames
- 🐍 Simple Python API
- 📖 Fully documented API and tutorials

---

## Available Algorithms

SO4GP currently implements several gradual pattern mining algorithms.

| Category | Algorithms |
|----------|------------|
| Classical Mining | GRAANK |
| Temporal Mining | TGRAANK |
| Optimization | Genetic Algorithm (GA), Ant Colony Optimization (ACO), Particle Swarm Optimization (PSO), Hill Climbing (HC), Random Search |
| Feature Selection | GradPFS |
| Clustering | ClusterGP |

---

## Supported Data Sources

SO4GP accepts datasets in either of the following formats:

- **Pandas DataFrame**
- **CSV file**

Datasets should primarily contain numerical attributes. Temporal algorithms additionally require a timestamp or ordered temporal attribute.

---

## Documentation Guide

If you are new to SO4GP, follow the documentation in the following order:

1. **Quick Start** – Install the package and run your first mining example.
2. **API Reference** – Explore the available classes, methods, and functions.
3. **Tutorials** – Learn practical workflows for feature selection and temporal mining.

---

## Citation

If you use **SO4GP** in your research, please cite the software using the following reference:

> Owuor, D. (2025). *owuordickson/sogp_pypi: v0.7.1* (Version v0.7.1) [Computer software]. Zenodo. DOI: <https://doi.org/10.5281/zenodo.16764372>

**BibTeX**

```bibtex
@software{owuor2025so4gp,
  author       = {Owuor, Dickson},
  title        = {owuordickson/sogp_pypi: v0.7.1},
  version      = {v0.7.1},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16764372},
  url          = {https://doi.org/10.5281/zenodo.16764372}
}
```

---

```{toctree}
:hidden:

introduction/basic_usage
```

```{toctree}
:hidden:
:caption: API

api/algorithms
api/classes
api/functions
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/feature_selection
tutorials/timeseries_analysis
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/owuordickson/gp-mining>
release_notes/index
```