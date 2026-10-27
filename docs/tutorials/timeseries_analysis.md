---
title: Studying Joint Effects with GP Descriptors
---

# Studying Joint Effects with GP Descriptors

Use this tutorial to study how the gradual relationships of individual attributes change when combined into a gradual pattern (GP). You will compare each gradual item (GI) with its joint GP using the same index-pair representation, descriptor calculations and plots.

The proposed hypothesis is that a coherent composition should show a well-organized progression of index relationships and be structurally more coherent than its individual GIs. Its desired profile is **high connectivity, low singularity and low rank dispersion**:

```{math}
\kappa_g \rightarrow 1,\qquad Si_g \rightarrow 0,\qquad \sigma_g^2 \rightarrow 0.
```

Here, connectivity is a component count, so the desired value is **one**. SO4GP reports rank dispersion as the population standard deviation; this tutorial also computes the population variance to match the hypothesis's squared notation. These descriptors provide evidence about DTW-like structural coherence. Their joint values do not establish that a composition is a valid DTW alignment.

The full executable walkthrough is [ftgp_composition.ipynb on GitHub](https://github.com/owuordickson/gp-mining/blob/main/applications/ftgp_composition.ipynb). Run the Python blocks below in order in a notebook, or follow the corresponding notebook sections.

## 1. Prepare your environment and data

Use Python 3.14 or later, as required by this checkout. From the repository root, install the package and notebook tools into your selected environment:

```sh
python -m pip install -e .
python -m pip install jupyterlab ipykernel
python -m jupyterlab applications/ftgp_composition.ipynb
```

Choose the same environment's kernel. The following imports and settings are shared by all steps:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from so4gp import GP, GI
from so4gp.algorithms import GRAANK

pd.set_option("display.max_rows", 40)
min_sup = 0.5
eq = False
```
The repository includes `applications/datasets/ke_rain_data_2k.csv` and `applications/datasets/ke_evi_data_2k.csv`. Each contains 23 July observations from 2000 through 2022, with four locations: NRB, KAK, MAK and WAJ. Rainfall columns use mm and EVI columns contain vegetation-index values.

Load the files from either the repository root or `applications/`. Strip header whitespace, parse dates and sort chronologically. Keep dates as the index so that only numerical attributes enter mining. The loader rejects missing values, non-finite values and duplicate dates; it does not impute or resample observations.

```python
data_candidates = [Path.cwd() / "datasets", Path.cwd() / "applications" / "datasets"]
data_dir = next(
    (path for path in data_candidates if (path / "ke_rain_data_2k.csv").is_file()),
    None,
)
if data_dir is None:
    raise FileNotFoundError("Run from the repository root or applications/ directory.")


def load_series(filename):
    frame = pd.read_csv(data_dir / filename)
    frame.columns = frame.columns.str.strip()
    dates = pd.to_datetime(frame.iloc[:, 0], format="%m-%Y", errors="raise")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Resolve missing or non-finite values before comparing pair sets.")
    values.index = pd.DatetimeIndex(dates, name="Date")
    if values.index.has_duplicates:
        raise ValueError("Resolve duplicate dates before interpreting index distances.")
    return values.sort_index()


datasets = {
    "Rainfall": load_series("ke_rain_data_2k.csv"),
    "EVI": load_series("ke_evi_data_2k.csv"),
}
for name, frame in datasets.items():
    print(f"{name}: {len(frame)} observations, {frame.index.min():%Y-%m} to {frame.index.max():%Y-%m}")
    display(frame.head())
```
Each index pair `(i, j)` refers to two observations **within the same dataset**. In these annual records, one row step represents one year. With irregular sampling, row-index distances do not represent uniform elapsed time. Rainfall and EVI are analyzed independently; the following intersections do not join the two datasets.

## 2. Understand the joint-effect representation

A GI combines an attribute and a direction. Under `eq=False`, `A+` retains pairs for which `A[j] > A[i]`, and `A-` retains pairs for which `A[j] < A[i]`. An increase describes value ordering and may point backward in chronological order.

The gradual warping set (GWS) contains these directed pairs. A GP retains only comparisons satisfying **all** its constituent GIs:

```{math}
W_{\{A^+,B^-\}}=W_{A^+}\cap W_{B^-},\qquad
W_{\{g_1,\ldots,g_m\}}=\bigcap_{r=1}^{m}W_{g_r}.
```

This intersection is the joint gradual behavior being studied: which relationships survive when the attributes must agree on a pair? SO4GP represents the individual constraints as packed pairwise matrices; bitwise AND performs their intersection. Adding a constraint can only remove pairs, so density cannot increase. Other structural descriptors need to be evaluated.

“DTW-like” refers here to the organization of pairs in index space. A standard DTW path is an ordered alignment satisfying endpoint, monotonicity and local-step constraints, optimized against an alignment cost; see the [tslearn DTW definition](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html#optimization-problem). A GWS is a set of directed comparisons, and strict comparisons exclude self-pairs. The workflow measures structural coherence without calculating a DTW alignment or distance.

## 3. Read the descriptors as a profile

| Measurement | Calculation | How to interpret it |
|---|---|---|
| Density | Retained pairs divided by `n * (n - 1) / 2` | How many comparisons survive; lower density alone does not mean more path-like. |
| Avg. Deviation from Diagonal | Mean absolute index gap `abs(i - j)` | Typical separation; low variance can also occur at a large constant offset. |
| Rank Dispersion | Population standard deviation of absolute index gaps | SO4GP's reported dispersion, rounded to three decimals. |
| Rank variance | Population variance of absolute index gaps | The hypothesis's squared dispersion; smaller means more consistent separations. |
| Graph Connectivity | Number of components in the undirected observation graph | A connected composition has **one** component. SO4GP counts participating observations only. |
| Singularity Score | Degree variance divided by mean degree, over all `n` observations | Smaller means more even participation; larger means comparisons concentrate on fewer observations. |
| Active coverage | Fraction of observations appearing in the pair set | Check alongside connectivity to detect excluded observations. |

Connectivity is measured on the graph whose vertices are observations and whose edges are the retained pairs. It does not test whether neighboring points on the `(i, j)` grid form a continuous monotone path. Low singularity does not rule out branching, and low rank variance does not establish DTW step constraints. The proposed profile is an operational hypothesis to test using the descriptors and plots together.

## 4. Mine candidate compositions

Use GRAANK with minimum support `0.5`, strict comparisons and descriptor computation enabled. This is gradual-pattern mining over ordered observations; no temporal lag estimator runs in this tutorial.

```python
miners = {}
mined_tables = []
for name, frame in datasets.items():
    miner = GRAANK(frame, min_sup=min_sup, eq=eq, device="cpu")
    _ = miner.discover(compute_descriptors=True, save_results=False)
    miners[name] = miner
    table = miner.mining_engine.display_patterns_as_df.copy()
    numeric_columns = table.columns.drop("Pattern")
    table[numeric_columns] = table[numeric_columns].apply(pd.to_numeric)
    table.insert(0, "Dataset", name)
    mined_tables.append(table)

mined_df = pd.concat(mined_tables, ignore_index=True)
display(mined_df.drop(columns="Confidence"))
```
The supplied data produce six two-item GPs per dataset. Keep the `miners` mapping: each engine's `valid_bins` supplies the constituent matrices for the next step. Confidence is omitted from the structural comparisons because this analysis selects no consequent.

## 5. Reconstruct and measure any GI or GP

Use the helpers below for both constituents and intersections. `compose_pattern(engine, gi_keys)` returns a descriptor dictionary and an array of directed pairs. Keys such as `0+` refer to the zero-based numerical attribute columns, with dates excluded. `item_key` looks up keys by attribute name.

All measurements use the same observation count. `Rank variance` is computed directly from the gaps, avoiding squaring the already rounded standard deviation. An empty intersection has zero density and coverage, with undefined geometry recorded as `NaN`.

In strict mode, density is the retained-pair fraction and equals a composed GP's pair-count support. Stored GI mining support in `valid_bins` instead uses the larger chronological triangle of its bitmap and can differ from GI density. Compare the actual pair sets and their descriptors consistently.

```python
STRUCTURAL = [
    "Density", "Avg. Deviation from Diagonal", "Rank Dispersion",
    "Rank variance", "Graph Connectivity", "Singularity Score",
]


def describe_pairs(pairs, n):
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    result = {column: np.nan for column in STRUCTURAL}
    result.update({
        "Pairs": len(pairs),
        "Active coverage": len(np.unique(pairs)) / n,
        "Density": len(pairs) / GP.pair_count(n),
    })
    if len(pairs):
        gp = GP()
        if not gp.compute_descriptors(pairs, obj_count=n):
            raise ValueError("SO4GP could not describe the pair set.")
        result.update({
            "Density": gp.density,
            "Avg. Deviation from Diagonal": gp.avg_deviation_from_diagonal,
            "Rank Dispersion": gp.rank_dispersion,
            "Rank variance": float(np.var(np.abs(pairs[:, 0] - pairs[:, 1]))),
            "Graph Connectivity": gp.graph_connectivity,
            "Singularity Score": gp.singularity_score,
        })
    return result


def item_key(engine, attribute, symbol="+"):
    return f"{engine.titles.index(attribute)}{symbol}"


def compose_pattern(engine, gi_keys):
    if not gi_keys:
        raise ValueError("Select at least one gradual item.")
    items = [GI.from_string(key) for key in gi_keys]
    if len({gi.attribute_col for gi in items}) != len(items):
        raise ValueError("Use one direction per distinct attribute.")
    packed = np.bitwise_and.reduce([
        engine.valid_bins[key].packed_bin_mat for key in gi_keys
    ])
    pairs = np.asarray(
        GP.gen_gradual_warping_set(packed, engine.row_count), dtype=int
    ).reshape(-1, 2)
    label = ", ".join(f"{engine.titles[gi.attribute_col]}{gi.symbol}" for gi in items)
    row = {"Pattern": label, **describe_pairs(pairs, engine.row_count)}
    return row, pairs
```
## 6. Check a controlled composition

Before interpreting the real data, construct two toy GIs whose joint set contains only adjacent-index relationships. The lower toy threshold permits inspecting the five retained pairs out of fifteen possible pairs.

```python
toy_data = pd.DataFrame({"A": [0, 2, 1, 4, 3, 5], "B": [4, 5, 2, 3, 0, 1]})
toy_miner = GRAANK(toy_data, min_sup=0.1, eq=False, device="cpu")
toy_engine = toy_miner.mining_engine
toy_engine.fit_bitmap()
toy_keys = ("0+", "1+")
toy_rows = [compose_pattern(toy_engine, (key,))[0] for key in toy_keys]
toy_joint, toy_pairs = compose_pattern(toy_engine, toy_keys)
toy_rows.append(toy_joint)
display(pd.DataFrame(toy_rows).round(3))
print("Joint pairs:", toy_pairs.tolist())

assert set(map(tuple, toy_pairs)) == {(0, 1), (2, 1), (2, 3), (4, 3), (4, 5)}
assert toy_joint["Graph Connectivity"] == 1
assert toy_joint["Rank variance"] == 0
```
The joint pairs are `(0, 1), (2, 1), (2, 3), (4, 3), (4, 5)`. They form one observation-graph component with full coverage and zero gap variance. Singularity is approximately `0.133`, compared with `0` for each GI: endpoints participate less than interior observations. This is a connected, narrow zigzag progression, but it does not improve singularity over the individual tie-free GIs, whose complete undirected graphs already have uniform degree.

This example makes the hypothesis testable without defining path-like appearance as automatic improvement on every descriptor.

## 7. Compare every discovered GP with its own GIs

Build the GI and GP descriptor tables, then record the GP-minus-GI changes for every constituent. The assertions verify that the reconstructed GP descriptors match the mining results and that the joint pairs are subsets of every input set.

```python
gi_rows, gp_rows = [], []
comparisons, hypothesis_rows = [], []
for name, miner in miners.items():
    engine = miner.mining_engine
    constituents = {}
    for key in sorted(engine.valid_bins):
        row, pairs = compose_pattern(engine, (key,))
        constituents[key] = (row, pairs)
        gi_rows.append({"Dataset": name, **row})

    for pattern in engine.gradual_patterns or []:
        keys = tuple(sorted(pattern.to_string(), key=lambda key: GI.from_string(key).attribute_col))
        joint, pairs = compose_pattern(engine, keys)
        gp_rows.append({"Dataset": name, **joint})
        assert np.isclose(joint["Density"], pattern.density)
        assert np.isclose(joint["Singularity Score"], pattern.singularity_score)
        assert np.isclose(joint["Rank Dispersion"], pattern.rank_dispersion)
        joint_set = set(map(tuple, pairs))
        baseline_rows = []
        for key in keys:
            baseline, gi_pairs = constituents[key]
            baseline_rows.append(baseline)
            assert joint_set <= set(map(tuple, gi_pairs))
            comparisons.append({
                "Dataset": name, "GP": joint["Pattern"], "GI": baseline["Pattern"],
                "Pair retention": len(pairs) / len(gi_pairs),
                "GP coverage": joint["Active coverage"],
                "GP kappa": joint["Graph Connectivity"],
                "Delta kappa": joint["Graph Connectivity"] - baseline["Graph Connectivity"],
                "Delta Si": joint["Singularity Score"] - baseline["Singularity Score"],
                "Delta variance": joint["Rank variance"] - baseline["Rank variance"],
            })
        hypothesis_rows.append({
            "Dataset": name, "GP": joint["Pattern"],
            "Connected, full coverage": joint["Graph Connectivity"] == 1 and joint["Active coverage"] == 1,
            "Lower Si than every GI": all(joint["Singularity Score"] < row["Singularity Score"] for row in baseline_rows),
            "Lower variance than every GI": all(joint["Rank variance"] < row["Rank variance"] for row in baseline_rows),
        })

gi_df = pd.DataFrame(gi_rows)
gp_df = pd.DataFrame(gp_rows)
comparison_df = pd.DataFrame(comparisons)
hypothesis_df = pd.DataFrame(hypothesis_rows)
display(gi_df.round(3))
display(gp_df.round(3))
```
The tables contain 16 individual GI rows and 12 joint GP rows. They form the matched evidence for the hypothesis:

- `Pair retention` gives the fraction of the constituent GI's pairs surviving composition.
- Negative `Delta Si` and `Delta variance` support improvement on those respective dimensions.
- `GP kappa = 1` together with `GP coverage = 1` means the joint observation graph remains connected and includes all observations.
- `Delta kappa = 0` when both GI and GP have one component indicates preservation of connectivity.

Evaluate the strict relative hypothesis using separate flags: connected with full coverage, lower singularity than every GI, and lower variance than every GI. Equality at an ideal value is preservation rather than strict improvement. This avoids hiding a tradeoff behind an averaged score.

```python
display(comparison_df.round(3))
criteria = [
    "Connected, full coverage", "Lower Si than every GI", "Lower variance than every GI",
]
hypothesis_df["All three criteria"] = hypothesis_df[criteria].all(axis=1)
display(hypothesis_df)
display(hypothesis_df.groupby("Dataset")[criteria + ["All three criteria"]].sum())
```
For example, rainfall `NRB+, KAK+` retains 174 pairs (`Density = 0.688`), keeps one component and full coverage, and has singularity `0.525`. Its rank dispersion is `5.381` and its unrounded rank variance is about `28.954`. Read those values alongside **both** GI baselines in `comparison_df`; a comparison of rainfall means with EVI means would answer a different question.

## 8. Plot the individual and joint pair sets

Use equal axes and scatter points so that each plotted mark corresponds to a retained comparison. Drawing lines through the full array would invent segments between pairs that are merely adjacent in storage order. The dashed diagonal provides an index reference.

```python
def plot_composition(engine, gi_keys, title):
    selections = [(key,) for key in gi_keys] + [tuple(gi_keys)]
    fig, axes = plt.subplots(1, len(selections), figsize=(4.5 * len(selections), 4.5))
    n = engine.row_count
    for ax, keys in zip(axes, selections):
        row, pairs = compose_pattern(engine, keys)
        ax.scatter(pairs[:, 0], pairs[:, 1], s=13, alpha=0.7)
        ax.plot([0, n - 1], [0, n - 1], "--", color="0.65", linewidth=1)
        ax.set(xlim=(-0.5, n - 0.5), ylim=(-0.5, n - 0.5),
               xlabel="Observation index i", ylabel="Observation index j",
               title=f"{row['Pattern']}\n{row['Pairs']} retained pairs")
        ax.set_aspect("equal")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


plot_composition(toy_engine, toy_keys, "Controlled example: individual GIs and their intersection")
```

```python
rain_engine = miners["Rainfall"].mining_engine
evi_engine = miners["EVI"].mining_engine
examples = {
    "Rainfall": (rain_engine, (
        item_key(rain_engine, "NRB (mm)", "+"), item_key(rain_engine, "KAK (mm)", "+"),
    )),
    "EVI": (evi_engine, (
        item_key(evi_engine, "KAK (EVI)", "+"), item_key(evi_engine, "MAK (EVI)", "-"),
    )),
}
for name, (engine, keys) in examples.items():
    plot_composition(engine, keys, f"{name}: constituent GIs and their joint GP")
```
Inspect which parts of the GI pair sets remain in the joint panel. Then plot the three hypothesis descriptors on separate axes so that their different scales remain interpretable:

```python
metrics = ["Graph Connectivity", "Singularity Score", "Rank variance"]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), squeeze=False)
for row_index, (name, (engine, keys)) in enumerate(examples.items()):
    records = [compose_pattern(engine, (key,))[0] for key in keys]
    records.append(compose_pattern(engine, keys)[0])
    for ax, metric in zip(axes[row_index], metrics):
        ax.bar(["GI 1", "GI 2", "Joint GP"], [record[metric] for record in records],
               color=["#4777A8", "#7EA6CE", "#D07B35"])
        ax.set_title(f"{name}: {metric}")
        ax.set_ylim(bottom=0)
        ax.axhline(1 if metric == "Graph Connectivity" else 0, color="0.4", linestyle="--", linewidth=1)
fig.suptitle("Hypothesis targets: one component, low singularity, low rank-gap variance")
fig.tight_layout()
plt.show()
```
A sparser GP represents stricter joint constraints. Its visual density alone does not establish weaker correlation or better path-like structure. Read coverage, connectivity, singularity and gap variance together.

## 9. Add another attribute

The same helper composes a GP with a third GI. This exploratory sequence can include patterns below the mining support threshold; structural inspection and frequent-pattern selection answer different questions.

```python
sequence = [item_key(rain_engine, name, "+") for name in ("NRB (mm)", "KAK (mm)", "MAK (mm)")]
progression_rows = [compose_pattern(rain_engine, sequence[:size])[0] for size in range(1, 4)]
progression_df = pd.DataFrame(progression_rows)
display(progression_df.round(3))
assert progression_df["Pairs"].is_monotonic_decreasing
```
Pair counts cannot increase along this sequence. Changes in singularity, variance and connectivity remain empirical findings.

## 10. State what the evidence supports

For the supplied rainfall and EVI data, all 12 discovered GPs retain one component among participating observations, and 11 retain full coverage. Rainfall `KAK+, WAJ+` includes only 22 of the 23 observations, demonstrating why connectivity needs the coverage check. Singularity is higher than for the constituent GIs, while gap variance changes in both directions. Two rainfall GPs and three EVI GPs have lower variance than every constituent. Consequently, these examples **do not support the strict hypothesis that composition improves all three structural properties relative to every individual GI**. The paired tables show exactly which dimensions are preserved, improved or worsened.

The analysis still demonstrates the intended use of GP descriptors: studying a joint effect as a change in the geometry and participation of the relationships retained by several attributes. The ideal profile of one component, low singularity and low variance provides the question; descriptor changes and plots provide the evidence. These are descriptive results rather than statistical significance tests, causal effects or time-lag estimates.

For your own data, preserve a meaningful observation order, select GIs by name and direction, call `compose_pattern`, and compare the returned GP against each constituent. Check active coverage and empty intersections before interpreting low dispersion. If you require a literal DTW alignment, separately enforce its boundary, monotonicity and local-step constraints and calculate an appropriate alignment cost.
