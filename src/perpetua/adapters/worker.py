"""A simulator worker process speaking JSON lines plus raw array bytes.

Used by adapters whose simulator lives in its own pinned Python environment
(COOM, AgarCL). Each request is one JSON line, optionally followed by
``request["bytes"]`` raw bytes; each reply is a JSON header
line, optionally followed by ``header["bytes"]`` raw bytes of an array with
``header["shape"]`` and ``header.get("dtype", default)``.

The worker's stderr is drained continuously on a background thread, so a
chatty native library can never block on a full pipe, and only the most recent
``STDERR_TAIL`` bytes are kept. If the worker dies, the error reports its exit
code and that tail instead of a bare "exited unexpectedly".

Every request has a deadline. A native library that hangs (rather than
crashing) is killed and reported with the same stderr tail, so a lifetime
cannot block forever. ``request(..., timeout=...)`` overrides the default for
one call; long replays can therefore grant themselves more time without
weakening the guard on ordinary steps.
"""

import io
import json
import math
import os
import selectors
import subprocess
import threading
import time
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from perpetua.core import JSONValue

__all__ = [
    "STDERR_TAIL",
    "DEFAULT_REQUEST_TIMEOUT",
    "WorkerProcess",
]

STDERR_TAIL = 8192

#: Default seconds to wait for one request/reply, overridable per call.
DEFAULT_REQUEST_TIMEOUT = 600.0
_CLOSE_TIMEOUT = 30.0
_CHUNK = 65536


class _Expired(Exception):
    """Internal marker: the request deadline passed."""


class WorkerProcess:
    """One worker process and its request/reply protocol."""

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        dtype: type[np.generic],
        cwd: str | None = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Worker request timeout must be finite and positive.")
        self.name = name
        self.dtype = dtype
        self.timeout = timeout
        self._tail = bytearray()
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            cwd=cwd,
        )
        # A blocking write would outlast the deadline once the worker stops reading.
        if self.process.stdin is not None:
            os.set_blocking(self.process.stdin.fileno(), False)
        self._drain = threading.Thread(target=self._read_stderr, daemon=True)
        self._drain.start()

    def _read_stderr(self) -> None:
        stderr = cast(io.BufferedReader, self.process.stderr)
        while chunk := stderr.read1(4096):
            with self._lock:
                self._tail.extend(chunk)
                del self._tail[:-STDERR_TAIL]

    def stderr_tail(self) -> str:
        """The last ``STDERR_TAIL`` bytes the worker wrote to stderr, decoded."""
        with self._lock:
            return self._tail.decode(errors="replace")

    def _died(self) -> RuntimeError:
        try:
            code = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = None
            self._kill()  # stdout closed but the process lingers: do not leak it
        self._drain.join(timeout=1)
        tail = self.stderr_tail().strip() or "(no stderr output)"
        return RuntimeError(
            f"The {self.name} worker exited unexpectedly (exit code {code}). "
            f"Last stderr output:\n{tail}"
        )

    def _timed_out(self, timeout: float) -> RuntimeError:
        self._kill()
        tail = self.stderr_tail().strip() or "(no stderr output)"
        return RuntimeError(
            f"The {self.name} worker did not reply within {timeout:g} seconds and was killed. "
            f"Last stderr output:\n{tail}"
        )

    def _kill(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass  # already exiting
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self._drain.join(timeout=1)

    def _await(self, descriptor: int, event: int, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _Expired
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, event)
            if not selector.select(remaining):
                raise _Expired

    def _write_all(self, data: bytes, deadline: float) -> None:
        descriptor = self.process.stdin.fileno() if self.process.stdin is not None else -1
        view = memoryview(data)
        while view:
            self._await(descriptor, selectors.EVENT_WRITE, deadline)
            try:
                written = os.write(descriptor, view[:_CHUNK])
            except BlockingIOError:
                continue
            if written <= 0:
                raise OSError("worker stdin accepted no bytes")
            view = view[written:]

    def _read_line(self, deadline: float) -> bytes | None:
        descriptor = self.process.stdout.fileno() if self.process.stdout is not None else -1
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                return line
            self._await(descriptor, selectors.EVENT_READ, deadline)
            chunk = os.read(descriptor, _CHUNK)
            if not chunk:
                return None
            self._buffer.extend(chunk)

    def _read_exact(self, size: int, deadline: float) -> bytes | None:
        descriptor = self.process.stdout.fileno() if self.process.stdout is not None else -1
        data = bytearray(self._buffer[:size])
        del self._buffer[: len(data)]
        while len(data) < size:
            self._await(descriptor, selectors.EVENT_READ, deadline)
            chunk = os.read(descriptor, min(size - len(data), _CHUNK))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def request(
        self,
        _data: bytes | None = None,
        /,
        *,
        timeout: float | None = None,
        **payload: JSONValue,
    ) -> tuple[dict[str, Any], NDArray[Any] | None]:
        """Send one request (plus optional raw bytes) and read its reply.

        Raises after killing the worker if no complete reply arrives within
        ``timeout`` (the constructor's default unless given).
        """
        limit = self.timeout if timeout is None else timeout
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("Worker request timeout must be finite and positive.")
        deadline = time.monotonic() + limit
        if _data is not None:
            payload = {**payload, "bytes": len(_data)}
        message = (json.dumps(payload) + "\n").encode()
        try:
            self._write_all(message + (_data or b""), deadline)
        except _Expired:
            raise self._timed_out(limit) from None
        except (BrokenPipeError, OSError) as error:
            raise self._died() from error
        try:
            line = self._read_line(deadline)
            if line is None:
                raise self._died()
            header = cast(dict[str, Any], json.loads(line))
            if "error" in header:
                raise RuntimeError(f"{self.name} worker: {header['error']}")
            if "bytes" not in header:
                return header, None
            size = int(header["bytes"])
            data = self._read_exact(size, deadline)
        except _Expired:
            raise self._timed_out(limit) from None
        if data is None or len(data) != size:
            raise self._died()
        dtype = np.dtype(header.get("dtype", self.dtype))
        flat = cast(NDArray[Any], np.frombuffer(data, dtype=dtype))
        return header, flat.reshape(tuple(header["shape"])).copy()

    def request_array(
        self,
        _data: bytes | None = None,
        /,
        *,
        timeout: float | None = None,
        **payload: JSONValue,
    ) -> tuple[dict[str, Any], NDArray[Any]]:
        """``request`` for commands whose reply must carry an array."""
        header, array = self.request(_data, timeout=timeout, **payload)
        if array is None:
            raise RuntimeError(
                f"{self.name} worker replied to {payload.get('command')!r} without an array."
            )
        return header, array

    def close(self) -> None:
        """Ask the worker to exit, kill it if it does not, and close its pipes.

        The close command has no reply: the worker exits. A worker that exits
        with a nonzero code, or has to be killed, is reported as a
        ``RuntimeWarning`` rather than an exception, because ``close`` usually
        runs in a ``finally`` block where raising would hide the original error.
        A worker that had already died is not reported again.
        """
        problem = None
        if self.process.poll() is None:
            try:
                self._write_all(b'{"command": "close"}\n', time.monotonic() + _CLOSE_TIMEOUT)
            except (_Expired, OSError):
                pass  # a stuck or broken pipe: the wait below kills the worker
            try:
                code = self.process.wait(timeout=_CLOSE_TIMEOUT)
                if code != 0:
                    problem = f"exited with code {code} while closing"
            except subprocess.TimeoutExpired:
                self._kill()
                problem = f"did not exit within {_CLOSE_TIMEOUT:g} seconds and was killed"
        self._drain.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass  # stdin may still hold a request the dead worker never read
        if problem is not None:
            tail = self.stderr_tail().strip() or "(no stderr output)"
            warnings.warn(
                f"The {self.name} worker {problem}. Last stderr output:\n{tail}",
                RuntimeWarning,
                stacklevel=2,
            )
