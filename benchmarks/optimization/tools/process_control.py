"""Account for and stop benchmark workers and their child processes."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


def _linux_snapshot():
    """Return PID -> (parent PID, RSS bytes) from procfs."""
    processes = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return processes
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8")
            values = {}
            for line in status.splitlines():
                if line.startswith(("PPid:", "VmRSS:")):
                    key, value = line.split(":", 1)
                    values[key] = int(value.split()[0])
            if "PPid" in values:
                processes[int(entry.name)] = (
                    values["PPid"],
                    values.get("VmRSS", 0) * 1024,
                )
        except OSError, ValueError, IndexError:
            continue
    return processes


def _windows_snapshot():
    """Read process parent links without opening unrelated processes."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
    kernel.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        return {}
    processes = {}
    entry = ProcessEntry()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        ok = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while ok:
            pid = int(entry.th32ProcessID)
            processes[pid] = (int(entry.th32ParentProcessID), 0)
            ok = kernel.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    return processes


def _windows_rss(pid):
    """Get working-set bytes for one process, or None if it is unavailable."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    )
    process_handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    if not process_handle:
        return None
    try:
        info = Counters()
        info.cb = ctypes.sizeof(info)
        if psapi.GetProcessMemoryInfo(process_handle, ctypes.byref(info), info.cb):
            return int(info.WorkingSetSize)
        return None
    finally:
        kernel.CloseHandle(process_handle)


def _snapshot():
    if sys.platform == "win32":
        return _windows_snapshot()
    if sys.platform.startswith("linux"):
        return _linux_snapshot()
    return {}


def _tree(snapshot, root):
    children = {}
    for pid, (parent, _) in snapshot.items():
        children.setdefault(parent, []).append(pid)
    result = [root]
    seen = {root}
    for parent in result:
        for child in children.get(parent, []):
            if child not in seen:
                seen.add(child)
                result.append(child)
    return result


def sample_tree(process):
    """Sum sampled resident bytes for the worker and all live descendants.

    This intentionally counts shared resident pages once per process and is a
    best-effort snapshot rather than a peak or private-memory measurement.
    """
    snapshot = _snapshot()
    pids = _tree(snapshot, process.pid)
    if sys.platform == "win32":
        values = [value for pid in pids if (value := _windows_rss(pid)) is not None]
    else:
        values = [
            snapshot[pid][1] for pid in pids if pid in snapshot and snapshot[pid][1]
        ]
    return sum(values) if values else None


def kill_tree(process):
    """Terminate the benchmark's process and the descendants in its snapshot."""
    pids = _tree(_snapshot(), process.pid)
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        for pid in reversed(pids):
            handle = kernel.OpenProcess(0x0001, False, pid)
            if handle:
                try:
                    kernel.TerminateProcess(handle, 1)
                finally:
                    kernel.CloseHandle(handle)
    elif sys.platform.startswith("linux"):
        for pid in reversed(pids):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        process.kill()
