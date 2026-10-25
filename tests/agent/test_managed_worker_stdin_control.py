"""Regression for #142271 / #143422: a Windows kanban worker's control reader parked on fd 0
deadlocks the memory provider's first native import (mem0 -> qdrant_client -> numpy).

The worker now reads control frames from a private dup of stdin and leaves fd 0 on NUL on Windows,
so any native DLL init that probes stdin cannot wait on the parked read.
"""

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.managed_worker import WorkerControls, _claim_stdin_control, read_frame

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs in a child interpreter: mutating fd 0 in the test process would break the runner's stdin,
# and the drain must be observed from a process whose stdin really is a pipe.
_PROBE = """
import os, stat, sys
from agent.managed_worker import _claim_stdin_control, read_frame

stream = _claim_stdin_control()
mode = os.fstat(0).st_mode
print("fd0_is_pipe", stat.S_ISFIFO(mode), sep="=")
print("frame", read_frame(stream)["n"], sep="=")
"""


def _probe():
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=REPO_ROOT,
        input='{"n": 7}\n', capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


@pytest.mark.platforms("windows")
def test_stdin_control_disarms_fd0_and_keeps_frames_readable():
    """fd 0 leaves the control pipe (so a parked native import cannot block on it) while the
    frames the owner writes to that pipe still reach the reader."""
    seen = _probe()
    assert seen["fd0_is_pipe"] == "False", "fd 0 still points at the control pipe"
    assert seen["frame"] == "7", "the control channel broke when fd 0 was disarmed"


@pytest.mark.platforms("linux", "macos")
def test_stdin_control_keeps_fd0_on_posix():
    """POSIX has no per-fd loader lock to dodge, so fd 0 stays as the owner passed it."""
    seen = _probe()
    assert seen["fd0_is_pipe"] == "True"
    assert seen["frame"] == "7"


def test_worker_controls_reads_the_given_stream():
    """The reader honours the stream handed to it instead of reaching for sys.stdin itself."""
    class _Channel:
        def send(self, *args, **kwargs):
            raise AssertionError("a finish frame sends nothing")

    controls = WorkerControls(_Channel(), "route", io.BytesIO(b'{"type": "finish"}\n'))
    assert controls.finish.wait(timeout=30), "finish frame was never consumed"


class _FakeStdin:
    """Only ``fileno()`` is used by the claim, so the test can hand it its own pipe."""

    def __init__(self, fd):
        self._fd = fd

    def fileno(self):
        return self._fd


def test_the_posix_branch_leaves_fd0_alone_and_still_reads_frames(monkeypatch):
    """The fd-0 surgery is the Windows workaround; off Windows the owner's stdin stays intact."""
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", _FakeStdin(read_fd))
    monkeypatch.setattr(sys, "platform", "linux")

    stream = _claim_stdin_control()
    try:
        os.fstat(read_fd)  # still the owner's pipe, not replaced or closed
        assert stream.fileno() != read_fd, "the claim re-used the owner's descriptor"
        os.write(write_fd, b'{"n": 11}\n{"n": 12}\n')
        assert read_frame(stream)["n"] == 11
        assert read_frame(stream)["n"] == 12
    finally:
        stream.close()
        os.close(read_fd)
        os.close(write_fd)


def test_each_claim_owns_a_distinct_fd_and_releases_it(monkeypatch):
    """Two claims must not share one fd, and closing a stream must give the fd back (no leak)."""
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", _FakeStdin(read_fd))
    monkeypatch.setattr(sys, "platform", "linux")

    first = _claim_stdin_control()
    second = _claim_stdin_control()
    owned = (first.fileno(), second.fileno())
    try:
        assert len(set(owned)) == 2, "both claims share one descriptor"
        assert read_fd not in owned, "a claim handed out the owner's own descriptor"
    finally:
        first.close()
        second.close()
        os.close(read_fd)
        os.close(write_fd)

    for fd in owned:
        with pytest.raises(OSError):
            os.fstat(fd)
