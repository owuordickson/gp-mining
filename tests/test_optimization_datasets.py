"""Offline checks for reproducible benchmark dataset preparation."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE = (
    Path(__file__).resolve().parents[1] / "benchmarks/optimization/tools/datasets.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("optimization_datasets", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_time_keeps_original_row_gaps_and_target(tmp_path):
    pd.DataFrame(
        {
            "Age": [1, 2, 3],
            "Glucose": [5, None, 7],
            "BMI": [4, 5, 6],
            "Classification": [1, 2, 1],
        }
    ).to_csv(tmp_path / "breast_cancer.csv", index=False)
    data, info = load_module().prepare("BCR", tmp_path, features=2)
    assert info["time_source"] == "synthetic_row_order"
    assert info["target_name"] == "Glucose"
    assert data.columns[info["target_index"]] == "Glucose"
    assert "Classification" not in data.columns
    delta = pd.to_datetime(data.iloc[1, 0]) - pd.to_datetime(data.iloc[0, 0])
    assert delta.total_seconds() == 2
    assert info["raw_rows"] == 3 and info["retained_rows"] == 2
    assert len(info["raw_sha256"]) == len(info["prepared_sha256"]) == 64


def test_real_time_sorted_and_missing_sentinel_removed(tmp_path):
    pd.DataFrame(
        {
            "Date": ["3/11/2004", "3/10/2004", "3/12/2004"],
            "Time": ["00:00:00"] * 3,
            "CO(GT)": [2, 1, -200],
            "T": [2, 1, 3],
        }
    ).to_csv(tmp_path / "air_quality.csv", index=False)
    data, info = load_module().prepare("AQY", tmp_path)
    assert info["time_source"] == "real"
    assert data["CO(GT)"].tolist() == [1, 2]
    assert data.iloc[0, 0].startswith("2004-03-10")


def test_directio_multiple_sites_are_not_mixed(tmp_path):
    pd.DataFrame(
        {
            "date_time": [
                "2019-01-01 00:00:00",
                "2019-01-01 00:01:00",
                "2019-01-01 00:02:00",
                "2019-01-01 00:03:00",
            ],
            "id_site": [1, 2, 1, 2],
            "Hm0": [1, 2, 3, 4],
            "Tz": [4, 3, 2, 1],
        }
    ).to_csv(tmp_path / "directio15k.csv", index=False)
    module = load_module()
    assert module.available_sites(tmp_path) == ["1", "2"]
    with pytest.raises(ValueError, match="site"):
        module.prepare("DIR", tmp_path)
    data, info = module.prepare("DIR", tmp_path, site=1)
    assert data["Hm0"].tolist() == [1, 3]
    assert info["site"] == "1"
