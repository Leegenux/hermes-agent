"""Regression for #142271 / #143422: a Windows kanban worker's control reader parked on fd 0
deadlocks the memory provider's first native import (mem0 -> qdrant_client -> numpy).

The worker now reads control frames from a private dup of stdin and leaves fd 0 on NUL on Windows,
so any native DLL init that probes stdin cannot wait on the parked read.
"""

import io
import subprocess
import sys
from pathlib import Path

import pytest

from agent.managed_worker import WorkerControls

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
