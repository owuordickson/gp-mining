## WP0 — Competitive Benchmark

To evaluate the performance of our proposed framework, we establish a competitive benchmark comparing our core methods against existing approaches. Our approach relies on **[GRAANK](https://pypi.org)** (Gradual Relation Analysis and Mining Association Rules) for discovering gradual patterns in multivariate numerical and time-series data, and **[T-GRAANK](https://pypi.org)**, which improves upon GRAANK by explicitly accounting for time lags. We compare our approach against direct competitors in causal discovery, graph learning, and statistical analysis, while filtering out unrelated pattern mining and prediction baselines.

| Approach Type | Method / Tool | Methodology & Focus | Benchmarking Role |
| :--- | :--- | :--- | :--- |
| **Our Approach** | **[GRAANK](https://pypi.org)** | Discovers gradual patterns and effects in multivariate numerical or time-series data. | Core baseline framework. |
| **Our Approach** | **[T-GRAANK](https://pypi.org)** | Enhances GRAANK by incorporating time lags for time-series data. | Advanced proposed framework. |
| **Competitor** | **[Tigrams / PCMCI](https://pypi.org)** | Discovers conditional independence and causal effects in multivariate time series. | Direct causal benchmark. |
| **Competitor** | **Granger Causality ([Statsmodels](https://statsmodels.org))** | Identifies predictive causality using vector autoregressive (VAR) modeling. | Baseline causal benchmark. |
| **Competitor** | **PC Algorithm ([causal-learn](https://github.com))** | Uses conditional independence tests to learn causal graph structures (DAGs). | Graph-based causal benchmark. |
| **Competitor** | **BSTS ([CausalImpact](https://github.com))** | Models causal impact and intervention effects using Bayesian structural time-series. | Intervention-based causal benchmark. |
| **Competitor** | **Classical Statistics ([SciPy](https://scipy.org))** | Measures linear relationships using the Pearson correlation coefficient. | Standard statistical baseline. |
| **Excluded** | **[PAMI](https://github.io)** | Mines frequent patterns in univariate transactional data. | Not useful for this benchmark. |
| **Excluded** | **[TabPFN-TS](https://github.com)** | Serves as a time-series prediction baseline for univariate data. | Not useful for this benchmark. |


