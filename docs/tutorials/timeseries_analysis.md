---
title: Analyzing Joint Effects in Time Series Using FTGPs
---

# Analyzing Joint Effects in Time Series Using FTGPs

## 1. Objective

This experiment explores the structural characteristics of gradual patterns (GPs) identified in two related time-series datasets: rainfall and the Enhanced Vegetation Index (EVI). It uses gradual warping sets to represent relationships between ordered observations and examines whether **gradual-pattern composition** can reveal joint effects while retaining desirable structural properties.

The motivating hypothesis is that combining gradual relationships might preserve the connectivity of the resulting gradual structure while reducing its **singularity** and **rank dispersion**. The experiment evaluates this expectation using the structural descriptors calculated by SO4GP. The results are interpreted in terms of the transformations induced by composition, rather than assuming that composition must improve every descriptor.

## 2. Datasets and experimental configuration

The notebook `ftgp_composition.ipynb` loads two datasets with the same location-based layout:

- **Rainfall:** `../datasets/ke_rain_data_2k.csv`, containing the columns `Date (Jul-Sep)`, `NRB (mm)`, `KAK (mm)`, `MAK (mm)` and `WAJ (mm)`.
- **Vegetation:** `../datasets/ke_evi_data_2k.csv`, containing `Date (Jul-Sep)`, `NRB (EVI)`, `KAK (EVI)`, `MAK (EVI)` and `WAJ (EVI)`.

The locations represented by the feature abbreviations are NRB, KAK, MAK and WAJ. The notebook prints the first five records from each dataset for inspection. No additional preprocessing or resampling step is shown in the saved notebook.

The GP mining configuration is:

```python
f_path_evi = "../datasets/ke_evi_data_2k.csv"
f_path_rain = "../datasets/ke_rain_data_2k.csv"

eq = False
min_sup = 0.5
tgt_cols = [1, 2, 3, 4]
compute_descriptors = True
```

Although the notebook introduces its configuration as TGradAMI hyperparameters, the executed mining procedure shown here uses **GRAANK** for both datasets. The `tgt_cols` variable is defined but is not passed into the displayed GRAANK calls.

## 3. Gradual-pattern discovery

Each dataset is processed independently with SO4GP's `GRAANK` implementation. The minimum support threshold is set to `0.5`, equality comparisons are disabled, and descriptor computation is enabled:

```python
from so4gp.algorithms import GRAANK

rain_grad = GRAANK(rain_data, min_sup, eq)
evi_grad = GRAANK(evi_data, min_sup, eq)

rain_grad.discover(compute_descriptors=True)
evi_grad.discover(compute_descriptors=True)
```

A GP identifies a co-variation relationship among attributes, with `+` and `-` indicating the corresponding directions of gradual change. In the notebook outputs, six two-item GPs are displayed for rainfall and six for EVI.

### 3.1 Rainfall gradual patterns

| Gradual pattern | Support | Density | Average diagonal deviation | Rank dispersion | Connectivity | Singularity |
|---|---:|---:|---:|---:|---:|---:|
| KAK (mm)+, NRB (mm)+ | 0.688 | 0.688 | 8.011 | 5.381 | 1 | 0.525 |
| NRB (mm)+, MAK (mm)+ | 0.557 | 0.557 | 7.291 | 4.926 | 1 | 1.902 |
| NRB (mm)+, WAJ (mm)+ | 0.581 | 0.581 | 7.769 | 5.154 | 1 | 1.333 |
| KAK (mm)+, MAK (mm)+ | 0.506 | 0.506 | 8.281 | 5.404 | 1 | 1.206 |
| KAK (mm)+, WAJ (mm)+ | 0.506 | 0.506 | 7.992 | 5.332 | 1 | 1.518 |
| WAJ (mm)+, MAK (mm)+ | 0.585 | 0.585 | 8.095 | 5.515 | 1 | 1.556 |
| **Mean** | **0.5705** | **0.5705** | **7.9065** | **5.2853** | **1** | **1.3400** |

The rainfall patterns all have graph connectivity equal to **1**. Their singularity scores nevertheless vary substantially, from **0.525** to **1.902**, showing that a connected gradual structure need not distribute index participation uniformly.

### 3.2 EVI gradual patterns

| Gradual pattern | Support | Density | Average diagonal deviation | Rank dispersion | Connectivity | Singularity |
|---|---:|---:|---:|---:|---:|---:|
| KAK (EVI)+, NRB (EVI)+ | 0.545 | 0.545 | 7.877 | 5.216 | 1 | 1.159 |
| NRB (EVI)+, MAK (EVI)+ | 0.534 | 0.534 | 8.156 | 5.446 | 1 | 1.357 |
| WAJ (EVI)-, NRB (EVI)+ | 0.526 | 0.526 | 7.774 | 5.322 | 1 | 1.172 |
| KAK (EVI)+, MAK (EVI)- | 0.636 | 0.636 | 8.311 | 5.706 | 1 | 0.826 |
| WAJ (EVI)-, KAK (EVI)+ | 0.549 | 0.549 | 8.273 | 5.213 | 1 | 1.086 |
| WAJ (EVI)-, MAK (EVI)+ | 0.518 | 0.518 | 7.611 | 5.239 | 1 | 1.189 |
| **Mean** | **0.5513** | **0.5513** | **8.0003** | **5.3570** | **1** | **1.1315** |

The EVI patterns are also connected (connectivity **1** in every displayed row). Their mean singularity (**1.1315**) is lower than that of the rainfall patterns (**1.3400**), while their mean rank dispersion (**5.3570**) is slightly higher than the rainfall mean (**5.2853**). These are **between-dataset descriptive comparisons**, not before-and-after composition effects.

### 3.3 Descriptor visualization

The notebook compares rainfall and EVI GP descriptors using a spider plot generated with `gp_descriptor_spider_plot`. It passes the numerical GP descriptor tables, the dataset labels `Rain` and `EVI`, and these seven dimensions:

`Support`, `Confidence`, `Density`, `Avg. Deviation from Diagonal`, `Rank Dispersion`, `Graph Connectivity`, and `Singularity Score`.

The plot provides an overview of how the two sets of discovered GPs differ across these measurements. Because the descriptors are reported on different numerical scales, their raw magnitudes should not be interpreted as directly comparable notions of quality.

## 4. Composition of gradual relationships

### 4.1 Gradual warping sets

A gradual warping set represents the index pairs associated with a gradual item or pattern. A set of directed index relationships can be visualized as a path or graph in the plane of observation indices `(i, j)`.

The notebook obtains gradual-item warping sets by fitting the mining engines:

```python
rain_grad.mining_engine.fit_warpingset()
evi_grad.mining_engine.fit_warpingset()

rain_warping_set = rain_grad.mining_engine.warping_set
evi_warping_set = evi_grad.mining_engine.warping_set
```

The intended composition combines the relationships represented by multiple GIs or GPs to investigate their joint temporal or gradual behavior. In structural terms, the resulting relationship set can be assessed using the same descriptors as its constituents. Whether its graph becomes denser, more concentrated, or less dispersed is an empirical question.

### 4.2 Individual gradual-item descriptors

The notebook constructs a `GP` object for each gradual item, assigns the support and confidence recorded in `valid_bins`, computes graph descriptors on that item's warping set, and combines the rainfall and EVI tables into `combined_df`.

There are **16 displayed GI rows**: four locations × two directions × two datasets. The notebook sets the GI-level `Confidence` column to `NaN` intentionally.

| Dataset | Gradual items | Support range | Density range | Rank dispersion range | Connectivity | Singularity range |
|---|---|---:|---:|---:|---:|---:|
| Rainfall | NRB±, KAK±, MAK±, WAJ± | 0.502–0.617 | 0.992–1.000 | 5.292–5.308 | 1 | 0.000–0.007 |
| EVI | NRB±, KAK±, MAK±, WAJ± | 0.538–0.660 | 0.996–1.000 | 5.284–5.292 | 1 | 0.000–0.004 |

The individual GI warping sets are almost fully dense according to the stored descriptor output, with very small singularity scores and connectivity **1** throughout. By comparison, the displayed two-item GP descriptors have considerably larger singularity scores and lower densities. This observation is **consistent with** joint gradual constraints concentrating participating relationships, although the notebook does not include a separate paired composition-results table that would quantify a specific composition operator's effect for every input/output pair.

### 4.3 Warping-path visualization

The notebook plots the rainfall gradual-item warping sets using `i` (horizontal axis) and `j` (vertical axis), with one subplot per gradual item. The plots illustrate how the index pairs form paths or zigzagging structures.

The notebook proposes that more strongly related time series may yield denser zigzag compositions, whereas weaker associations may lead to sparser structures. This is a **qualitative visualization hypothesis**; the notebook does not compute an independent correlation metric or report a formal test of that association.

## 5. Discussion: structural effects of GP composition

The central question is whether combining gradual relationships both preserves structural connectivity and systematically improves dispersion-related descriptors.

**Connectivity.** The displayed rainfall and EVI GP results, as well as the individual gradual-item descriptor table, all report graph connectivity equal to **1**. Within these reported results, the gradual structures remain connected. This supports connectivity preservation in the observed examples, without establishing a universal guarantee for all compositions.

**Singularity.** The comparatively larger singularity scores observed for multi-item GPs indicate a more uneven distribution of index participation than is seen in the individual GI warping sets. An increase in singularity suggests that the combined gradual constraints concentrate the resulting structure around a subset of indices, rather than distributing participation more uniformly. Thus, composition cannot be assumed to reduce singularity.

**Rank dispersion.** Dispersion changes are comparatively modest and not uniformly directed. In the displayed GP tables, rainfall and EVI have mean rank dispersion values of **5.2853** and **5.3570**, respectively; the individual GI values cluster close to **5.29**. The observed differences do not demonstrate a systematic improvement in rank dispersion through composition.

**Density and geometric structure.** Individual GI warping sets have densities close to **1**, whereas the discovered two-item GP structures have mean densities of **0.5705** for rainfall and **0.5513** for EVI. Their diagonal-deviation values also differ from the GI-level descriptors. Taken together, these measurements indicate that imposing multiple gradual relationships changes the structure of the retained index pairs. Such structural change should not automatically be interpreted as an improvement in all measured characteristics.

### Assessment of the initial hypothesis

The experiment's interpretation is that **GP composition preserves connectivity but does not systematically improve the other structural characteristics**. In particular, the increase in singularity suggests that combining gradual relationships concentrates the resulting structure around a subset of indices, rather than distributing index participation more uniformly. Rank dispersion exhibits comparatively small and inconsistent changes across compositions. Therefore, the initial hypothesis that composition systematically produces lower singularity and rank dispersion **is not supported** by these results. Instead, the experiment demonstrates that composition induces a measurable structural transformation while maintaining a connected gradual structure in the reported cases.

This conclusion is deliberately restricted to the available evidence. The saved notebook reports the original GP and GI descriptor tables and visualizations, but **does not show an explicit, separately evaluated table of composed outputs with matched before/after descriptor differences**. Consequently, these findings are descriptive: they motivate the proposed interpretation but do not establish statistical significance or a general property of the composition operation.

## 6. Conclusion

The rainfall and EVI experiments demonstrate how FTGP-oriented analysis can use gradual warping sets and structural descriptors to examine joint relationships in time-series data. The mined gradual patterns remain connected in the displayed cases, but their structure differs from that of individual gradual items, particularly in density and singularity. Composition should therefore be understood as a means of **transforming and characterizing joint gradual relationships**, not as an operation guaranteed to decrease singularity or rank dispersion.

A stronger follow-up analysis would explicitly record each composition's input warping sets and output warping set, compute paired descriptor changes, and test whether those changes are consistent across datasets and composition choices.

---

*Source: `ftgp_composition.ipynb` (saved code cells, output tables and plot descriptions). All reported numerical values are taken from the notebook's saved outputs; the hypothesis assessment incorporates the interpretation supplied for this experiment.*
