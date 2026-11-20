"""Deterministic preparation of the six optimization benchmark datasets."""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

REVISION = "9f128eada3da0604fe18e6474d427ced7e72ffc6"
BASE_URL = (
    "https://raw.githubusercontent.com/owuordickson/meso-hpc-lr/"
    f"{REVISION}/hpc%40lr/data/"
)
# Files can be explicitly overridden via the variants, never silently substituted.
DATASETS = {
    "BCR": ("breast_cancer.csv", "Glucose", "synthetic"),
    "C2K": ("c2k.csv", "o_dlv_e", "synthetic"),
    "AQY": ("air_quality.csv", "CO(GT)", "air"),
    "DIR": ("directio15k.csv", "Hm0", "directio"),
    "PWC": ("power_consumption50k.csv", "Global_active_power", "power"),
    "APS": ("aps_16k.csv", "aa_000", "synthetic"),
    "C2K1K": ("c2k_1k.csv", "o_dlv_e", "synthetic"),
    "AQY1K": ("air_quality1k.csv", "CO(GT)", "air"),
    "DIR1K": ("directio1k.csv", "Hm0", "directio"),
    "DIR8K": ("directio8k.csv", "Hm0", "directio"),
    "PWC1K": ("power_consumption1k.csv", "Global_active_power", "power"),
    "PWC10K": ("power_consumption10k.csv", "Global_active_power", "power"),
    "PWC19K": ("power_consumption19k.csv", "Global_active_power", "power"),
    "APS2K": ("aps_2k.csv", "aa_000", "synthetic"),
    "APS5K": ("aps_5k.csv", "aa_000", "synthetic"),
}


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path, name: str, download: bool) -> bytes:
    if not path.is_file():
        if not download:
            raise FileNotFoundError(
                f"Dataset {path} not found; enable download explicitly"
            )
        request = urllib.request.Request(
            BASE_URL + name, headers={"User-Agent": "so4gp-benchmark"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data
    return path.read_bytes()


def available_sites(
    data_dir: str | Path, *, dataset_code: str = "DIR", download: bool = False
) -> list[str]:
    """Return the distinct DIR site identifiers without combining their series."""
    code = dataset_code.upper()
    if code not in DATASETS or DATASETS[code][2] != "directio":
        raise ValueError(f"Not a Directio dataset: {dataset_code!r}")
    name = DATASETS[code][0]
    raw = _load(Path(data_dir) / name, name, download)
    sites = pd.read_csv(io.BytesIO(raw), usecols=["id_site"])["id_site"]
    return sorted(sites.dropna().astype(str).unique().tolist())


def prepare(
    dataset_code: str,
    data_dir: str | Path,
    *,
    download: bool = False,
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
    code = dataset_code.upper()
    if code not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_code!r}; choose from {sorted(DATASETS)}"
        )
    if rows is not None and rows < 2:
        raise ValueError("rows must be at least 2")
    if features is not None and features < 2:
        raise ValueError("features must be at least 2")
    name, default_target, kind = DATASETS[code]
    chosen_target = target or default_target
    raw = _load(Path(data_dir) / name, name, download)
    separator = ";" if kind == "power" else ","
    frame = pd.read_csv(io.BytesIO(raw), sep=separator, na_values=["?", "na", "NA"])
    raw_rows = len(frame)
    frame.columns = frame.columns.astype(str).str.strip()
    available_sites = []
    selected_site = None
    if kind == "directio" and "id_site" in frame.columns:
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
        if kind == "air":
            time = pd.to_datetime(
                frame["Date"].astype(str) + " " + frame["Time"].astype(str),
                format="%m/%d/%Y %H:%M:%S",
                errors="raise",
            )
        elif kind == "power":
            time = pd.to_datetime(
                frame["Date"].astype(str) + " " + frame["Time"].astype(str),
                format="%d/%m/%Y %H:%M:%S",
                errors="raise",
            )
        elif kind == "directio":
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
    if code.startswith("C2K"):
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
    if code.startswith("AQY"):
        numeric = numeric.replace(-200, float("nan"))
    if code.startswith("APS"):
        numeric = numeric.replace(-1, float("nan"))
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
        "dataset": code,
        "filename": name,
        "source_revision": REVISION,
        "source_url": BASE_URL + name,
        "raw_sha256": _hash(raw),
        "prepared_sha256": _hash(prepared_bytes),
        "raw_rows": raw_rows,
        "selected_rows": selected_rows,
        "retained_rows": len(prepared),
        "removed_columns": removed,
        "columns": list(prepared.columns),
        "target_name": chosen_target,
        "target_index": int(prepared.columns.get_loc(chosen_target)),
        "time_source": "synthetic_row_order" if kind == "synthetic" else "real",
        "site": selected_site,
        "available_sites": available_sites,
        "missing_policy": "complete_case",
    }
    return prepared, metadata
