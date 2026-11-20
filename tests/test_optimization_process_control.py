"""Benchmark process accounting must include worker descendants."""

import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


def load_control():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/optimization/tools/process_control.py"
    )
    spec = importlib.util.spec_from_file_location("optimization_process_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not (sys.platform == "win32" or sys.platform.startswith("linux")),
    reason="supported platforms",
)
def test_sample_and_terminate_descendant_worker():
    control = load_control()
    child_code = (
        "import os,time; data=bytearray(32*1024*1024); "
        "print(os.getpid(),flush=True); time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-u', '-c', {child_code!r}]); "
        "time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", parent_code], stdout=subprocess.PIPE, text=True
    )
    child_pid = int(process.stdout.readline())
    try:
        assert control.sample_tree(process) >= 32 * 1024 * 1024
        control.kill_tree(process)
        process.wait(timeout=5)
        assert process.stdout.read() == ""
    finally:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(child_pid)], capture_output=True
            )
        else:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
