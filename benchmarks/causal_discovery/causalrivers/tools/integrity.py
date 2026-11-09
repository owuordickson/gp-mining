"""Benchmark-root paths and checksum verification before deserialization."""

import hashlib
import io
import json
import pickle
import re
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = BENCHMARK_ROOT / "run_manifest.json"


def artifact_path(file_path: str | Path) -> Path:
    """Resolve both relative and absolute artifacts within the benchmark root."""
    path = Path(file_path)
    path = (BENCHMARK_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(BENCHMARK_ROOT.resolve()):
        raise ValueError(f"Artifact is outside benchmark root: {file_path}")
    return path


def artifact_key(file_path: str | Path) -> str:
    return artifact_path(file_path).relative_to(BENCHMARK_ROOT.resolve()).as_posix()


def verified_bytes(file_path: str | Path, expected_hash: str | None = None) -> bytes:
    """Verify the exact bytes subsequently loaded; fail closed on missing digests."""
    path = artifact_path(file_path)
    path_key = artifact_key(path)
    if expected_hash is None:
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            runs = manifest["data_runs"]
            if not isinstance(runs, list):
                raise TypeError("data_runs must be a list")
            matches = [
                run
                for run in runs
                if artifact_key(run.get("filepath", run.get("filename", ""))) == path_key
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one manifest entry for {path_key}")
            expected_hash = matches[0].get("sha256")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid or missing integrity manifest: {MANIFEST_PATH}") from exc
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise ValueError(f"Missing or invalid SHA-256 for {path_key}")
    payload = path.read_bytes()
    computed_hash = hashlib.sha256(payload).hexdigest()
    if computed_hash != expected_hash.lower():
        raise ValueError(
            f"Integrity check failed for {path_key}!\nExpected: {expected_hash}\nGot: {computed_hash}"
        )
    return payload


def secure_load_pickle(file_path: str | Path, expected_hash: str | None = None):
    """Load only bytes verified against a manifest or an explicit recorded digest."""
    return pickle.load(io.BytesIO(verified_bytes(file_path, expected_hash)))
