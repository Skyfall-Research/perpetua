"""The shared simulator worker protocol and its crash diagnostics."""

import os
import sys
import threading
import time
import warnings

import numpy as np
import pytest

import perpetua.adapters.worker as worker_module
from perpetua.adapters.worker import DEFAULT_REQUEST_TIMEOUT, STDERR_TAIL, WorkerProcess

REPLY = """
import json, sys, time
import numpy as np
close_code = 0
for line in sys.stdin:
    request = json.loads(line)
    if request["command"] == "close":
        if close_code == "hang":
            time.sleep(60)
        sys.exit(close_code)
    if request["command"] in ("fail_on_close", "hang_on_close"):
        close_code = 4 if request["command"] == "fail_on_close" else "hang"
        sys.stderr.write("simulated cleanup failure\\n")
        sys.stderr.flush()
    if request["command"] == "empty":
        sys.stdout.write(json.dumps({"ok": True}) + "\\n")
        sys.stdout.flush()
        continue
    if request["command"] == "noisy":
        sys.stderr.write("x" * 2_000_000)
        sys.stderr.flush()
    if request["command"] == "hang":
        time.sleep(60)
    if request["command"] == "crash":
        sys.stderr.write("Traceback: simulated native failure\\n")
        sys.stderr.flush()
        sys.exit(3)
    array = np.arange(6, dtype=np.int32).reshape(2, 3)
    header = {"ok": True, "shape": [2, 3], "bytes": array.nbytes, "dtype": "int32"}
    sys.stdout.buffer.write((json.dumps(header) + "\\n").encode() + array.tobytes())
    sys.stdout.buffer.flush()
"""


def worker(timeout: float = DEFAULT_REQUEST_TIMEOUT) -> WorkerProcess:
    return WorkerProcess(
        "test",
        [sys.executable, "-c", REPLY],
        env=dict(os.environ),
        dtype=np.float32,
        timeout=timeout,
    )


def test_reply_array_uses_header_dtype() -> None:
    process = worker()
    try:
        header, array = process.request(command="get")
        assert header["ok"] is True
        assert array is not None
        assert array.dtype == np.int32
        np.testing.assert_array_equal(array, np.arange(6).reshape(2, 3))
    finally:
        process.close()
    assert process.process.returncode == 0


def test_crash_reports_exit_code_and_stderr_tail() -> None:
    process = worker()
    try:
        with pytest.raises(RuntimeError, match=r"(?s)exit code 3\).*simulated native failure"):
            process.request(command="crash")
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            process.request(command="get")  # writing to a dead worker is also reported
    finally:
        process.close()


def test_chatty_stderr_cannot_block_and_is_bounded() -> None:
    process = worker()
    try:
        # 2 MB of stderr would fill an undrained pipe and deadlock the worker.
        header, _ = process.request(command="noisy")
        assert header["ok"] is True
        assert len(process.stderr_tail()) <= STDERR_TAIL
    finally:
        process.close()


def test_hung_worker_is_killed_reported_and_reaped() -> None:
    process = worker(timeout=0.2)
    try:
        with pytest.raises(RuntimeError, match=r"did not reply within 0\.2 seconds.*killed"):
            process.request(command="hang")
        assert process.process.poll() is not None
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            process.request(command="get")
    finally:
        process.close()
    assert process.process.returncode is not None


def test_a_worker_that_stops_reading_a_large_request_is_timed_out() -> None:
    stall = """
import os, time
left = 70000
while left:
    left -= len(os.read(0, left))
time.sleep(20)
"""
    process = WorkerProcess(
        "test", [sys.executable, "-c", stall], env=dict(os.environ), dtype=np.float32, timeout=1.0
    )
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="did not reply within 1 seconds"):
            process.request(bytes(1 << 20), command="load")
        assert time.monotonic() - started < 10.0
        assert process.process.poll() is not None
    finally:
        process.close()


def test_per_request_timeout_override_still_guards_long_calls() -> None:
    process = worker(timeout=30.0)
    try:
        with pytest.raises(RuntimeError, match="did not reply within 0.1 seconds"):
            process.request(command="hang", timeout=0.1)
    finally:
        process.close()


def test_worker_killed_while_idle_is_reported_and_close_reaps_it() -> None:
    process = worker()
    try:
        process.process.kill()
        process.process.wait()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            process.request(command="get")
    finally:
        process.close()
    assert process.process.returncode is not None


def test_worker_killed_mid_request_reports_and_close_reaps_it() -> None:
    process = worker(timeout=30.0)
    failures: list[RuntimeError] = []

    def request() -> None:
        try:
            process.request(command="hang")
        except RuntimeError as error:
            failures.append(error)

    thread = threading.Thread(target=request)
    thread.start()
    try:
        time.sleep(0.2)
        assert process.process.poll() is None  # the worker is still in its step
        process.process.kill()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert failures
        assert "exited unexpectedly" in str(failures[0])
    finally:
        process.close()
    assert process.process.returncode is not None


def test_clean_close_reaps_the_worker_without_a_warning() -> None:
    process = worker()
    process.request(command="get")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        process.close()
    assert process.process.returncode == 0


def test_nonzero_exit_on_close_is_reported_as_a_warning() -> None:
    process = worker()
    process.request(command="fail_on_close")
    with pytest.warns(RuntimeWarning, match=r"(?s)exited with code 4.*simulated cleanup failure"):
        process.close()
    assert process.process.returncode == 4


def test_worker_that_ignores_close_is_killed_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "_CLOSE_TIMEOUT", 0.5)
    process = worker()
    process.request(command="hang_on_close")
    with pytest.warns(RuntimeWarning, match="did not exit within 0.5 seconds and was killed"):
        process.close()
    assert process.process.returncode is not None


def test_close_after_a_reported_crash_does_not_warn_again() -> None:
    process = worker()
    with pytest.raises(RuntimeError, match="exit code 3"):
        process.request(command="crash")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        process.close()


def test_request_array_refuses_a_reply_without_an_array() -> None:
    process = worker()
    try:
        with pytest.raises(RuntimeError, match="replied to 'empty' without an array"):
            process.request_array(command="empty")
    finally:
        process.close()
