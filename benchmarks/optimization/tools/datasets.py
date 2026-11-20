"""Deterministic preparation of the six optimization benchmark datasets."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import pandas as pd


# Files can be explicitly overridden via the variants, never silently substituted.
DATASETS = {
    "BCR": ("breast_cancer.csv", "Glucose", "medical"),
    "C2K": ("c2k.csv", "o_dlv_e", "transport"),
    "AQY": ("air_quality.csv", "CO(GT)", "ecology"),
    "DIR": ("directio15k.csv", "Hm0", "coastline"),
    "PWC": ("power_consumption50k.csv", "Global_active_power", "power"),
    "APS": ("aps_16k.csv", "aa_000", "vehicle"),
}


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset {path} not found; enable download explicitly"
        )
    return path.read_bytes()



def prepare(
    dataset_code: str,
    data_dir: str | Path,
    *,
    rows: int | None = None,
    features: int | None = None,
    target: str | None = None,
    site: str | int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load and prepare one dataset for all methods in a comparison.

    The synthetic datetime is an encoding of original row position, not a
    measurement time. DIR requires a site when the file contains multiple sites.
    The returned frame is ordered with time first and has a fresh RangeIndex.
    """
    d_set = dataset_code.upper()
    if d_set not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_code!r}; choose from {sorted(DATASETS)}"
        )
    if rows is not None and rows < 2:
        raise ValueError("rows must be at least 2")
    if features is not None and features < 2:
        raise ValueError("features must be at least 2")

    name, default_target, kind = DATASETS[d_set]
    chosen_target = target or default_target
    raw = _load(Path(data_dir) / name)
    separator = ";" if d_set == "PWC" else ","
    frame = pd.read_csv(io.BytesIO(raw), sep=separator, na_values=["?", "na", "NA"])
    raw_rows = len(frame)
    frame.columns = frame.columns.astype(str).str.strip()
    available_sites = []
    selected_site = None
    if d_set == "DIR" and "id_site" in frame.columns:
        available_sites = sorted(
            frame["id_site"].dropna().astype(str).unique().tolist()
        )
        if site is None and len(available_sites) > 1:
            raise ValueError(
                f"DIR contains multiple sites; choose site from {available_sites}"
            )
        selected_site = (
            str(site)
            if site is not None
            else (available_sites[0] if available_sites else None)
        )
        if selected_site is not None:
            frame = frame.loc[frame["id_site"].astype(str) == selected_site].copy()
            if frame.empty:
                raise ValueError(
                    f"Unknown DIR site {selected_site!r}; available: {available_sites}"
                )

    source_row = pd.Series(frame.index.to_numpy(), index=frame.index)
    try:
        if d_set == "AQY":
            time = pd.to_datetime(
                frame["Date"].astype(str) + " " + frame["Time"].astype(str),
                format="%m/%d/%Y %H:%M:%S",
                errors="raise",
            )
        elif d_set == "PWC":
            time = pd.to_datetime(
                frame["Date"].astype(str) + " " + frame["Time"].astype(str),
                format="%d/%m/%Y %H:%M:%S",
                errors="raise",
            )
        elif d_set == "DIR":
            time = pd.to_datetime(
                frame["date_time"], format="%Y-%m-%d %H:%M:%S", errors="raise"
            )
        else:
            time = pd.Timestamp("2000-01-01") + pd.to_timedelta(source_row, unit="s")
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid timestamp in {name}: {exc}") from exc

    if time.isna().any():
        raise ValueError(f"Missing timestamp in {name}")

    frame.insert(0, "__benchmark_time", time)
    frame = frame.sort_values("__benchmark_time", kind="stable")
    if rows is not None:
        frame = frame.iloc[:rows].copy()
    selected_rows = len(frame)

    excluded = {
        "Classification",
        "class",
        "Date",
        "Time",
        "date_time",
        "timestamp",
        "id_site",
    }
    if d_set.startswith("C2K"):
        excluded.add("nr")
        excluded.update(
            c for c in frame if c.endswith("_legid") or c.endswith("_place")
        )
    removed = [str(c) for c in frame.columns if c in excluded]
    frame = frame.drop(columns=removed)
    if chosen_target not in frame.columns:
        raise ValueError(f"Target {chosen_target!r} is unavailable in {name}")
    numeric = frame.drop(columns="__benchmark_time").apply(
        pd.to_numeric, errors="coerce"
    )

    unusable = [c for c in numeric if numeric[c].nunique(dropna=True) < 2]
    if chosen_target in unusable:
        raise ValueError(f"Target {chosen_target!r} is missing or constant")
    removed.extend(unusable)
    numeric = numeric.drop(columns=unusable)
    columns = list(numeric.columns)

    if features is not None:
        columns = columns[:features]
        if chosen_target not in columns:
            columns[-1] = chosen_target
        removed.extend(c for c in numeric if c not in columns)
    if len(columns) < 2:
        raise ValueError(f"Dataset {name} has fewer than two usable numeric attributes")
    prepared = pd.concat(
        [frame[["__benchmark_time"]], numeric[columns]], axis=1
    ).dropna()
    if len(prepared) < 2:
        raise ValueError(f"Dataset {name} has fewer than two complete rows")
    prepared = prepared.reset_index(drop=True)
    prepared["__benchmark_time"] = prepared["__benchmark_time"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    prepared_bytes = prepared.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    metadata = {
        "dataset": d_set,
        "filename": name,
        "raw_sha256": _hash(raw),
        "prepared_sha256": _hash(prepared_bytes),
        "raw_rows": raw_rows,
        "selected_rows": selected_rows,
        "retained_rows": len(prepared),
        "removed_columns": removed,
        "columns": list(prepared.columns),
        "target_name": chosen_target,
        "target_index": int(prepared.columns.get_loc(chosen_target)),
        "missing_policy": "complete_case",
    }
    return prepared, metadata
