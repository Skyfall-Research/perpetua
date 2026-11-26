"""Measurement tools for the framework's speed and cost claims.

The framework's speed claims are hypotheses until measured. This module holds
the small, exact instruments those measurements need, so results are
reproducible by others:

- ``count_compilations`` counts XLA compilation events over a window (the
  "no recompilation across drift" claim).
- ``xla_flops`` returns XLA's static per-call FLOP estimate, and
  ``autoreset_step``/``transition_scan`` provide the gymnax-style
  both-branch auto-reset baseline for the "removing the reset path saves
  compute" claim on identical dynamics.
- ``code_lines`` counts physical, comment, docstring and code lines for the
  integration-cost claim.

Nothing here measures wall-clock throughput by itself, and none of it
changes framework behavior: the instruments are inert wrappers around
existing pure functions.
"""

# JAX config/typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import ast
import io
import logging
import tokenize
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from perpetua.core import Environment, as_int

__all__ = [
    "CompilationCounter",
    "count_compilations",
    "xla_flops",
    "autoreset_step",
    "transition_scan",
    "LineCount",
    "code_lines",
]


class CompilationCounter:
    """Compilation events captured while a ``count_compilations`` block ran.

    ``events`` holds the raw ``Compiling ...`` messages in order; internal
    primitive compilations are included, so measurements should warm up first
    and report deltas.
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    @property
    def count(self) -> int:
        """Number of captured compilation events."""
        return len(self.events)


class _CompileHandler(logging.Handler):
    """Append ``Compiling ...`` records to a counter's event list."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Compiling "):
            self.events.append(message)


_COMPILE_LOGGER = "jax._src.interpreters.pxla"


@contextmanager
def count_compilations() -> Generator[CompilationCounter]:
    """Count XLA compilation events while the context is active.

    Enables ``jax_log_compiles`` for the duration and captures the compile
    records of the JAX logger that emits them (``jax._src.interpreters.pxla``,
    at WARNING). The count depends on that record being emitted, so the block
    lifts ``logging.disable(...)`` and any level or ``disabled`` setting that
    would suppress it, and restores all of them on exit; without that a muted
    logger would make the count 0 for a program that does compile. While
    active, JAX also logs each compile at WARNING level, which Python's default
    logging setup prints to stderr. JAX's in-memory compilation cache means a
    warmed call with the same shapes records nothing, which is exactly what a
    drift lifetime should show.
    """
    counter = CompilationCounter()
    emitter = logging.getLogger(_COMPILE_LOGGER)
    handler = _CompileHandler(counter.events)
    previous_flag = bool(jax.config.jax_log_compiles)
    previous_disable = logging.root.manager.disable
    previous_level, previous_disabled = emitter.level, emitter.disabled
    logging.disable(logging.NOTSET)
    emitter.disabled = False
    if not emitter.isEnabledFor(logging.WARNING):
        emitter.setLevel(logging.WARNING)
    emitter.addHandler(handler)
    jax.config.update("jax_log_compiles", True)
    try:
        yield counter
    finally:
        jax.config.update("jax_log_compiles", previous_flag)
        emitter.removeHandler(handler)
        emitter.setLevel(previous_level)
        emitter.disabled = previous_disabled
        logging.disable(previous_disable)


def xla_flops(fn: Any, *args: Any) -> float:
    """Return XLA's static FLOP estimate for one compiled call of ``fn``.

    This is the compiler's platform-independent cost model, not measured
    hardware throughput; treat it as an upper-bound-style estimate for
    comparing two computations with identical outputs.
    """
    compiled = jax.jit(fn).lower(*args).compile()
    analysis = compiled.cost_analysis()
    if analysis is None:
        return 0.0
    if isinstance(analysis, list):
        entries = cast(list[dict[str, float]], analysis)
        return float(sum(float(entry.get("flops", 0.0)) for entry in entries))
    table = cast(dict[str, float], analysis)
    return float(table.get("flops", 0.0))


def _select(done: jax.Array, new: jax.Array, old: jax.Array) -> jax.Array:
    """Select a leaf, carrying PRNG keys that ``jnp.where`` cannot hold."""
    if jnp.issubdtype(new.dtype, jax.dtypes.prng_key):
        return cast(
            jax.Array,
            jax.random.wrap_key_data(
                jnp.where(done, jax.random.key_data(new), jax.random.key_data(old))
            ),
        )
    return jnp.where(done, new, old)


def autoreset_step[StateT](
    env: Environment[StateT],
    state: StateT,
    action: ArrayLike,
    key: jax.Array,
    *,
    limit: int,
    ticks: jax.Array,
) -> tuple[StateT, jax.Array, jax.Array]:
    """One gymnax-style both-branch auto-reset transition (measurement baseline).

    Vectorised JAX environments typically implement auto-reset by computing
    both the next state and a fresh reset state every step and selecting per
    environment. This function reproduces that computation exactly on the
    same dynamics: every call runs ``env.step`` **and** ``env.init``, then
    selects the fresh state once the legacy episode limit is reached. Returns
    the selected state, the step reward and the updated episode counter.
    """
    limit = as_int(limit, "limit must be a positive integer.", minimum=1)
    step_key, reset_key = jax.random.split(key)
    next_state, _obs, reward, _info = env.step(state, action, step_key)
    fresh_state, _fresh_obs, _fresh_info = env.init(reset_key)
    elapsed = ticks + jnp.int32(1)
    done = elapsed >= limit

    def select(new: Any, old: Any) -> Any:
        return _select(done, new, old)

    selected = jax.tree.map(select, fresh_state, next_state)
    return selected, reward, jnp.where(done, jnp.int32(0), elapsed)


def transition_scan[StateT](
    env: Environment[StateT],
    state: StateT,
    action: ArrayLike,
    key: jax.Array,
    *,
    ticks: int,
    autoreset_limit: int | None = None,
) -> tuple[StateT, jax.Array]:
    """Compiled scan of ``ticks`` transitions of one dynamics function.

    With ``autoreset_limit`` unset this is the lifetime transition loop: one
    ``env.step`` per tick, no reset path. With ``autoreset_limit`` set every
    tick also computes a fresh ``env.init`` and selects it on the limit, the
    both-branch auto-reset baseline above. The action is held fixed so both
    paths run identical policy-free dynamics; returns the final state and the
    stacked rewards of shape ``(ticks, agents)``.
    """
    ticks = as_int(ticks, "ticks must be a positive integer.", minimum=1)
    if autoreset_limit is not None:
        autoreset_limit = as_int(
            autoreset_limit, "autoreset_limit must be None or a positive integer.", minimum=1
        )
    action = jnp.asarray(action)
    if autoreset_limit is None:

        def lifetime_body(
            carry: tuple[StateT, jax.Array], _: None
        ) -> tuple[tuple[StateT, jax.Array], jax.Array]:
            current, current_key = carry
            current_key, step_key = jax.random.split(current_key)
            next_state, _obs, reward, _info = env.step(current, action, step_key)
            return (next_state, current_key), reward

        (state, key), rewards = jax.lax.scan(lifetime_body, (state, key), None, length=ticks)
    else:
        limit = autoreset_limit

        def autoreset_body(
            carry: tuple[StateT, jax.Array, jax.Array], _: None
        ) -> tuple[tuple[StateT, jax.Array, jax.Array], jax.Array]:
            current, current_key, elapsed = carry
            current_key, step_key = jax.random.split(current_key)
            next_state, reward, next_elapsed = autoreset_step(
                env, current, action, step_key, limit=limit, ticks=elapsed
            )
            return (next_state, current_key, next_elapsed), reward

        (state, key, _), rewards = jax.lax.scan(
            autoreset_body, (state, key, jnp.int32(0)), None, length=ticks
        )
    return state, rewards


class LineCount(NamedTuple):
    """Physical line counts for one file, split by role."""

    total: int
    blank: int
    comment: int
    docstring: int
    code: int


def code_lines(path: Path) -> LineCount:
    """Count a Python file's lines, separating blanks, comments and docstrings.

    ``code`` counts every remaining physical line (including multi-line
    statements). ``docstring`` counts lines inside module, class and function
    docstrings via the parsed AST, so integration-cost reports can exclude
    prose. ``comment`` counts lines holding only a comment token, so a line
    starting with ``#`` inside a docstring is docstring and inside any other
    string is code. The file must parse as Python.
    """
    path = Path(path)
    text = path.read_text()
    tree = ast.parse(text)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                start = body[0].lineno
                end = body[0].end_lineno or start
                docstring_lines.update(range(start, end + 1))
    lines = text.splitlines()
    comment_lines = {
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type == tokenize.COMMENT
        and not lines[token.start[0] - 1][: token.start[1]].strip()
    }
    total = blank = comment = docstring = code = 0
    for number, line in enumerate(lines, start=1):
        total += 1
        if not line.strip():
            blank += 1
        elif number in docstring_lines:
            docstring += 1
        elif number in comment_lines:
            comment += 1
        else:
            code += 1
    return LineCount(total, blank, comment, docstring, code)
