"""Run reproducible optimization comparisons in isolated subprocesses."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = "affc7860397002fba26af96dcb2f0461f2cdda9d"

if __package__:
    from .tools.datasets import DATASETS, prepare
    from .tools.process_control import kill_tree, sample_tree
    from .tools.report import write_reports
else:
    # PyCharm invokes the script without a parent package.
    sys.path.insert(0, str(HERE))
    from tools.datasets import DATASETS, prepare
    from tools.process_control import kill_tree, sample_tree
    from tools.report import write_reports


def run_child(config, timeout, command=None, memory_limit_bytes=None):
    """Execute a JSON worker and distinguish timeout, memory limit, and errors."""
    command = command or [sys.executable, str(HERE / "tools" / "experiments.py")]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(config.get("seed", 0)),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    peak = [None]
    killed = threading.Event()
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.05):
            rss = sample_tree(process)
            if rss is not None:
                peak[0] = max(peak[0] or 0, rss)
                if memory_limit_bytes and rss > memory_limit_bytes:
                    killed.set()
                    kill_tree(process)
                    return

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        stdout, stderr = process.communicate(json.dumps(config), timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(process)
        process.communicate()
        return {
            "status": "timeout",
            "process_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak[0],
            "memory_sampling_seconds": 0.05,
        }
    finally:
        stop.set()
        thread.join()
    common = {
        "process_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak[0],
        "memory_sampling_seconds": 0.05,
    }
    if killed.is_set():
        return {**common, "status": "memory_limit"}
    if process.returncode:
        return {**common, "status": "error", "error": stderr[-8000:] or stdout[-8000:]}
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            **common,
            "status": "error",
            "error": f"Invalid worker output: {stdout[-4000:]} {stderr[-4000:]}",
        }
    if "error" in result or "Error" in result:
        return {**common, **result, "status": "error"}
    return {**common, **result, "status": "ok"}


def _legacy_source():
    """Extract the pinned historical package from local Git objects."""
    cache = ROOT / ".superpowers" / "optimization" / LEGACY
    source = cache / "src" / "so4gp" / "__init__.py"
    cache.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", LEGACY, "src/so4gp"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    import io

    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        expected = {
            member.name: hashlib.sha256(tree.extractfile(member).read()).hexdigest()
            for member in tree.getmembers()
            if member.isfile()
        }
        if not source.exists():
            tree.extractall(cache, filter="data")
        actual = {
            path.relative_to(cache).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (cache / "src" / "so4gp").rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        if actual != expected:
            raise RuntimeError(
                "Historical source cache does not match its pinned Git revision"
            )
    return str(cache / "src")


def _total_ram_bytes():
    if sys.platform == "win32":
        import ctypes

        total_kib = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(
            ctypes.byref(total_kib)
        ):
            return total_kib.value * 1024
    elif sys.platform.startswith("linux"):
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return None


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--experiment",
        choices=("all", "graank", "kernel", "version"),
        default="all",
    )
    parser.add_argument(
        "--profile", choices=("quick", "study", "full"), default="full"
    )
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / f"run-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--features", type=int)
    parser.add_argument("--target", help="Override target column name")
    parser.add_argument(
        "--site", help="Select a Directio site when a file contains several"
    )
    parser.add_argument("--support", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--apriori-level",
        type=int,
        help="Optional APRIORI depth bound; default is exhaustive",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--warmups", type=int, default=1, help="Discarded warm-up trials per workload"
    )
    parser.add_argument(
        "--memory-mib", type=float, default=6144, help="Best-effort sampled working-set limit"
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    seeds = (
        args.seeds
        if args.seeds is not None
        else (list(range(3)) if args.profile == "study" else [0])
    )
    if (
        args.timeout <= 0
        or args.iterations < 1
        or args.warmups < 0
        or (args.memory_mib is not None and args.memory_mib <= 0)
    ):
        raise ValueError("Invalid support, timeout, or iteration count")
    if args.profile == "full" and args.memory_mib is None:
        raise ValueError("The full profile requires a positive --memory-mib limit")
    defaults = {"quick": (64, 6), "study": (1000, 12), "full": (None, None)}
    rows, features = defaults[args.profile]
    rows, features = args.rows or rows, args.features or features
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("output", "resume")
    }
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", "src"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    benchmark_hashes = {
        str(file.relative_to(HERE)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in HERE.rglob("*.py")
    }
    versions = {}
    for package in ("numpy", "pandas", "torch", "scikit-learn", "scikit-fuzzy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "settings": settings,
        "revision": head,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "benchmark_sha256": benchmark_hashes,
        "legacy_revision": LEGACY,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _total_ram_bytes(),
        "dependencies": versions,
        "timing": "worker algorithm elapsed; process startup and preparation separate",
        "memory": (
            "sampled worker process-tree resident memory every 0.05s; "
            "includes imports and post-mining validation"
        ),
    }
    if manifest_path.exists():
        if (
            not args.resume
            or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest
        ):
            raise ValueError(
                "Output exists or resume configuration/source does not match"
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    trials_path = args.output / "trials.jsonl"
    records = (
        [
            json.loads(line)
            for line in trials_path.read_text(encoding="utf-8").splitlines()
        ]
        if trials_path.exists()
        else []
    )
    if args.resume and records:
        write_reports(args.output, records)
    completed = {r.get("trial_id") for r in records}
    experiments = {
        "graank": ["apriori", "aco", "ga", "pso", "hc", "random"],
        "kernel": ["dense", "packed"],
        #"version": ["legacy", "current"],
    }
    if args.experiment != "all":
        experiments = {args.experiment: experiments[args.experiment]}
    for dataset in args.datasets:
        records_before_dataset = len(records)
        try:
            frame, metadata = prepare(
                dataset,
                args.data_dir,
                rows=rows,
                features=features,
                target=args.target,
                site=args.site,
            )
            previous_hashes = {
                r.get("prepared_sha256")
                for r in records
                if r.get("dataset") == dataset and r.get("prepared_sha256")
            }
            if previous_hashes and previous_hashes != {metadata["prepared_sha256"]}:
                raise RuntimeError(
                    f"Prepared data changed for {dataset}; use a new output directory"
                )
            data_csv = frame.to_csv(
                index=False, lineterminator="\n", float_format="%.17g"
            )
            if (
                hashlib.sha256(data_csv.encode("utf-8")).hexdigest()
                != metadata["prepared_sha256"]
            ):
                raise ValueError(
                    "In-memory data hash differs from preprocessing metadata"
                )
        except RuntimeError:
            raise
        except Exception as exc:
            record = {
                "dataset": dataset,
                "experiment": "preparation",
                "method": "input",
                "status": "input_error",
                "error": str(exc),
            }
            records.append(record)
            with trials_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            write_reports(args.output, records)
            continue
        for seed in seeds:
            work = [
                (exp, method)
                for exp, methods in experiments.items()
                for method in methods
            ]
            random.Random(seed).shuffle(work)
            for exp, method in work:
                digest = metadata.get("prepared_sha256")
                trial_id = f"{dataset}:{exp}:{method}:{seed}:{digest}"
                if trial_id in completed:
                    continue
                config = {
                    "experiment": "graank" if exp == "version" else exp,
                    "method": "apriori" if exp == "version" else method,
                    "data_csv": data_csv,
                    "support": args.support,
                    "seed": seed,
                    "target": metadata["target_index"],
                    "iterations": args.iterations,
                    "apriori_level": None, #args.apriori_level,
                }
                record = {
                    "trial_id": trial_id,
                    "dataset": dataset,
                    "experiment": exp,
                    "method": method,
                    "seed": seed,
                    "support": args.support,
                    "target": metadata["target_name"],
                    **metadata,
                    "rows": metadata["retained_rows"],
                    "features": len(metadata["columns"]) - 1,
                }
                try:
                    if exp == "version" and method == "legacy":
                        config["source_path"] = _legacy_source()
                    limit = int(args.memory_mib * 1048576) if args.memory_mib else None
                    warmup_statuses = []
                    for _ in range(args.warmups):
                        warmup_statuses.append(
                            run_child(config, args.timeout, memory_limit_bytes=limit)[
                                "status"
                            ]
                        )
                    result = run_child(config, args.timeout, memory_limit_bytes=limit)
                    result["warmup_statuses"] = warmup_statuses
                except Exception as exc:
                    result = {"status": "error", "error": str(exc)}
                record.update(result)
                records.append(record)
                with trials_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, default=str) + "\n")
                print(
                    f"{dataset} {exp}/{method} seed={seed}: {record['status']}",
                    flush=True,
                )
        if len(records) > records_before_dataset:
            write_reports(args.output, records)
    return 0 if all(r["status"] == "ok" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
