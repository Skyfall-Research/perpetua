"""Long-run soak checks for native lifetimes and native-library workers.

``native`` advances vmapped ContinualGrid lifetimes through cyclic drift for a
target number of aggregate environment ticks. At each sample it records host
RSS, device memory, throughput and health, and at the end checks exact clocks,
zero recompiles after warm-up and bounded memory growth.
A check the platform cannot measure (RSS without ``/proc``, device memory on
the CPU backend, an integer projection from a single sample) is reported as
``null`` rather than as passed; only a failed check fails the soak.

``env`` soaks any JAX-native registered environment (``--list`` prints the
specs) with a few long vmapped lifetimes, uniform random actions and cyclic
drift on its default field (ForagerGrid under ``Accumulate``). Besides the
native checks it verifies that every
floating-point leaf of the state and observation stays finite, and projects
each integer leaf's growth to 10^10 ticks per lifetime: a signed counter that
would overflow fails, and an unsigned one that would wrap is listed for review
(unsigned counters may wrap by design, e.g. with modular differences).
``+scripted`` specs add drifting scripted slots to the multi-agent MPE tasks.

``host`` soaks a host-executed simulator (Meta-World, COOM, EnvPool, MORPHEUS's
mock) through ``HostRunner``, sampling parent and worker RSS, descriptors and
zombie children, then churns short-lived instances. Linux only.

``ppo``, ``dqn`` and ``sac`` run the native checks on one continuing learner
(PPO with eight agents, DQN on one agent, SAC on gymnax Pendulum, all with
cyclic drift), so optimizer, rollout, replay and metric state are soaked too.

``worker`` drives one AgarCL lifetime for a long seeded run, sampling worker
RSS, the parent's open file descriptors and zombie children; it then restores
the final state by replaying the full action log in a fresh worker, checks the
next step matches exactly, and churns many short-lived workers to check that
descriptors and child processes return to their baseline. Linux only (it reads
``/proc``).

Every mode prints one JSON object per sample and a final JSON summary; any
failed check exits non-zero.
"""

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

import perpetua as pp
from perpetua.core import Clock

# Phases switch every DRIFT_PERIOD ticks (a prime, so they never align with chunks).
DRIFT_PERIOD = 1009
# Counters an adapter clamps below their limit by design, so linear growth is
# expected and they never overflow (see the adapter and its unit test).
SATURATING = {"gymnax/": ("[0].env.env.time",)}


def _emit(record: Mapping[str, object]) -> None:
    print(json.dumps(record), flush=True)


def _rss_bytes(pid: int | str = "self") -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _growth(first: int | None, last: int | None) -> int | None:
    """Growth between two samples, or None when the platform reports neither."""
    return None if first is None or last is None else last - first


def _within(growth: int | None, limit: int) -> bool | None:
    """Whether growth stays within ``limit``; None when it was not measured."""
    return None if growth is None else growth <= limit


def _passed(checks: dict[str, Any]) -> bool:
    """A soak passes when no check failed; ``None`` marks a check not measured here."""
    return all(value is not False for value in checks.values())


def _settled_growth(samples: list[int | None]) -> int | None:
    """RSS growth over the samples, ignoring unreadable ones and the first tenth.

    Simulators and allocators warm up in the first tenth of a run, and a
    process that has exited has no readable RSS.
    """
    known = [sample for sample in samples if sample]
    settled = known[len(known) // 10 :] or known
    return settled[-1] - settled[0] if settled else None


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


def _children() -> list[tuple[int, str]]:
    """(pid, state) of every direct child of this process."""
    me = os.getpid()
    found: list[tuple[int, str]] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        # The command name is parenthesised and may contain spaces.
        fields = stat[stat.rindex(")") + 2 :].split()
        if int(fields[1]) == me:
            found.append((int(entry), fields[0]))
    return found


def _device_memory() -> dict[str, int | None]:
    stats = jax.local_devices()[0].memory_stats() or {}
    return {
        "device_bytes_in_use": stats.get("bytes_in_use"),
        "device_peak_bytes": stats.get("peak_bytes_in_use"),
    }


def _ticks(clock: Any) -> np.ndarray:
    hi = np.asarray(clock.hi, dtype=np.uint64)
    lo = np.asarray(clock.lo, dtype=np.uint64)
    return (hi << np.uint64(32)) | lo


def native(args: argparse.Namespace) -> bool:
    env = pp.make(
        "ContinualGrid-v0",
        size=8,
        num_agents=2,
        drift=pp.drift.Cyclic([(7, 7), (0, 0), (0, 7)], period=127),
    )
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=args.chunk)

    def initialize(seed: jax.Array) -> Any:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=args.window)

    run_batch = cast(Callable[[Any], Any], jax.jit(jax.vmap(runner), donate_argnums=0))
    carry = jax.jit(jax.vmap(initialize))(jnp.arange(args.batch, dtype=jnp.uint32))
    carry = jax.block_until_ready(run_batch(carry))  # warm-up chunk, counted below
    bool(jnp.all(carry.metrics.healthy))  # compile the health reduction too
    calls = max(args.ticks // (args.chunk * args.batch), 1)
    first: dict[str, Any] | None = None
    record: dict[str, Any] = {}
    started = last_time = time.perf_counter()
    last_call = 0
    with pp.measurement.count_compilations() as compiles:
        for call in range(1, calls + 1):
            carry = run_batch(carry)
            if call % args.sample_every and call != calls:
                continue
            healthy = bool(jnp.all(carry.metrics.healthy))
            now = time.perf_counter()
            record = {
                "mode": "native",
                "call": call,
                "ticks_per_lifetime": int(_ticks(carry.state.t)[0]),
                "aggregate_ticks": (call + 1) * args.chunk * args.batch,
                "elapsed_s": round(now - started, 3),
                "ticks_per_s": (call - last_call) * args.chunk * args.batch / (now - last_time),
                "healthy": healthy,
                "rss_bytes": _rss_bytes(),
                "peak_rss_bytes": _peak_rss_bytes(),
                **_device_memory(),
            }
            last_time, last_call = now, call
            first = first or record
            _emit(record)
            if not healthy:
                break
    ticks = _ticks(carry.state.t)
    metric_ticks = _ticks(carry.metrics.ticks)
    expected = (calls + 1) * args.chunk
    last = record
    assert first is not None
    rss_growth = _growth(first["rss_bytes"], last["rss_bytes"])
    device_growth = _growth(first["device_bytes_in_use"], last["device_bytes_in_use"])
    checks = {
        "healthy": bool(jnp.all(carry.metrics.healthy)),
        "exact_clocks": bool(np.all(ticks == expected)),
        "exact_metric_ticks": bool(np.all(metric_ticks == expected)),
        "no_recompiles_after_warmup": compiles.count == 0,
        "rss_growth_under_limit": _within(rss_growth, args.max_rss_growth),
        "device_memory_flat": _within(device_growth, 0),
    }
    _emit(
        {
            "mode": "native",
            "summary": True,
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "provenance": pp.reporting.provenance(),
            "batch": args.batch,
            "chunk": args.chunk,
            "window": args.window,
            "agents_per_lifetime": 2,
            "ticks_per_lifetime": int(ticks[0]),
            "aggregate_env_ticks": int(ticks.sum()),
            "wall_s": round(time.perf_counter() - started, 3),
            "compile_events_after_warmup": compiles.count,
            "compile_events": compiles.events,
            "rss_growth_bytes": rss_growth,
            "device_growth_bytes": device_growth,
            "checks": checks,
        }
    )
    return _passed(checks)


def env_specs() -> list[str]:
    """Every JAX-native registered id, plus ``+scripted`` multi-agent MPE variants."""
    specs: list[str] = []
    for env_id in pp.environments(builtin=True):
        if pp.spec(env_id).host:
            continue
        specs.append(env_id)
        if env_id.startswith("jaxmarl/") and env_id != "jaxmarl/MPE_simple_v3":
            specs.append(f"{env_id}+scripted")
    return specs


def spec_kwargs(spec: str) -> tuple[str, dict[str, Any], str]:
    """(env id, constructor kwargs, drift description) for a soak spec."""
    env_id, _, variant = spec.partition("+")
    probe = pp.make(env_id)
    inner = getattr(probe, "env", None)
    field = getattr(inner, "default_drift_field", None)
    kwargs: dict[str, Any] = {}
    if env_id == "ForagerGrid-v0":
        # Its mass only moves under an endogenous process; the default is stationary.
        kwargs["drift"] = pp.drift.Accumulate(gain=1.0, decay=0.5, low=1.0, high=8.0)
        description = "agent-driven mass (Accumulate)"
    elif field is None or inner is None:
        description = "stationary (no driftable field)"
    else:
        value = np.asarray(getattr(inner.default_params, field), dtype=np.float32)
        if field == "goal":
            other = np.zeros_like(value)
        elif field == "doors":
            other = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        elif field == "item_rewards":
            other = -value
        else:
            other = np.where(value != 0, value * 0.9, 0.1).astype(np.float32)
        kwargs["drift"] = pp.drift.Cyclic([value, other], period=DRIFT_PERIOD)
        description = f"cyclic {field} every {DRIFT_PERIOD} ticks"
    if variant == "scripted":
        kwargs["num_scripted"] = 1
        kwargs["scripted_drift"] = pp.drift.Cyclic([(1.0,), (0.0,)], period=3 * DRIFT_PERIOD)
        description += f"; one scripted slot, beta cycling 1/0 every {3 * DRIFT_PERIOD} ticks"
    elif variant:
        raise ValueError(f"Unknown soak variant {variant!r} in {spec!r}.")
    return env_id, kwargs, description


def _integer_leaves(tree: Any) -> list[tuple[str, Any]]:
    """Named integer leaves, excluding the two-word clocks (they carry by design)."""
    flat, _ = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda node: isinstance(node, Clock)
    )
    found: list[tuple[str, Any]] = []
    for path, leaf in flat:
        if isinstance(leaf, Clock) or not hasattr(leaf, "dtype"):
            continue
        if jnp.issubdtype(leaf.dtype, jnp.integer):
            found.append((jax.tree_util.keystr(path), leaf))
    return found


def _float_leaves(tree: Any) -> list[Any]:
    return [
        leaf
        for leaf in jax.tree.leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def env_soak(args: argparse.Namespace) -> bool:
    env_id, kwargs, description = spec_kwargs(args.spec)
    env = pp.make(env_id, **kwargs)
    policy = pp.policies.uniform(env.action_space)
    runner = pp.runtime.Runner(env, policy, ticks=args.chunk)

    def initialize(seed: jax.Array) -> Any:
        return pp.runtime.init_lifetime(env, seed=seed, window_size=args.window)

    @jax.jit
    def finite(carry: Any) -> jax.Array:
        leaves = _float_leaves((carry.state, carry.obs))
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))

    @jax.jit
    def integer_extremes(carry: Any) -> list[jax.Array]:
        # Unsigned leaves are compared as they are: a signed cast would wrap past 2**31.
        return [
            jnp.max(leaf)
            if jnp.issubdtype(leaf.dtype, jnp.unsignedinteger)
            else jnp.max(jnp.abs(leaf))
            for _, leaf in _integer_leaves((carry.state, carry.metrics))
        ]

    run_batch = cast(Callable[[Any], Any], jax.jit(jax.vmap(runner), donate_argnums=0))
    carry = jax.jit(jax.vmap(initialize))(jnp.arange(args.batch, dtype=jnp.uint32))
    carry = jax.block_until_ready(run_batch(carry))  # warm-up chunk, counted below
    leaves = _integer_leaves((carry.state, carry.metrics))
    names = [name for name, _ in leaves]
    limits = [int(jnp.iinfo(leaf.dtype).max) for _, leaf in leaves]
    signed = [bool(jnp.issubdtype(leaf.dtype, jnp.signedinteger)) for _, leaf in leaves]
    # Compile the host-side sample reductions before counting recompiles.
    _ = (bool(jnp.all(carry.metrics.healthy)), bool(finite(carry)), integer_extremes(carry))
    calls = max(args.ticks // args.chunk, 1)
    first: dict[str, Any] | None = None
    record: dict[str, Any] = {}
    extremes_first: list[int] = []
    extremes_last: list[int] = []
    rises: list[int] = []  # sampling intervals in which each integer leaf's maximum rose
    intervals = 0
    started = last_time = time.perf_counter()
    last_call = 0
    all_finite = True
    with pp.measurement.count_compilations() as compiles:
        for call in range(1, calls + 1):
            carry = run_batch(carry)
            if call % args.sample_every and call != calls:
                continue
            healthy = bool(jnp.all(carry.metrics.healthy))
            all_finite = bool(finite(carry))
            extremes = [int(value) for value in integer_extremes(carry)]
            now = time.perf_counter()
            record = {
                "mode": "env",
                "spec": args.spec,
                "call": call,
                "ticks_per_lifetime": int(_ticks(carry.state.t)[0]),
                "elapsed_s": round(now - started, 3),
                "ticks_per_s": (call - last_call) * args.chunk * args.batch / (now - last_time),
                "healthy": healthy,
                "finite": all_finite,
                "rss_bytes": _rss_bytes(),
                **_device_memory(),
            }
            last_time, last_call = now, call
            if first is None:
                first, extremes_first, rises = record, extremes, [0] * len(extremes)
            else:
                intervals += 1
                rises = [
                    count + (now_value > before)
                    for count, now_value, before in zip(rises, extremes, extremes_last, strict=True)
                ]
            extremes_last = extremes
            _emit(record)
            if not (healthy and all_finite):
                break
    assert first is not None
    ticks = _ticks(carry.state.t)
    expected = (calls + 1) * args.chunk
    span = max(record["ticks_per_lifetime"] - first["ticks_per_lifetime"], 1)
    horizon = 10**10
    overflowing: dict[str, dict[str, int]] = {}
    wrapping: dict[str, dict[str, int]] = {}
    for name, limit, is_signed, start, end, rose in zip(
        names, limits, signed, extremes_first, extremes_last, rises, strict=True
    ):
        # Only steadily rising leaves are counters; random-valued ones (labels,
        # positions) can rise between two samples without growing.
        if intervals == 0 or rose < 0.9 * intervals:
            continue
        if any(
            env_id.startswith(prefix) and name in leaves for prefix, leaves in SATURATING.items()
        ):
            continue
        rate = max(end - start, 0) / span
        projected = end + rate * (horizon - record["ticks_per_lifetime"])
        if projected > limit:
            target = overflowing if is_signed else wrapping
            target[name] = {"first": start, "last": end, "projected": int(projected)}
    rss_growth = _growth(first["rss_bytes"], record["rss_bytes"])
    device_growth = _growth(first["device_bytes_in_use"], record["device_bytes_in_use"])
    checks = {
        "healthy": bool(jnp.all(carry.metrics.healthy)),
        "finite_state_and_obs": all_finite,
        "exact_clocks": bool(np.all(ticks == expected)),
        "exact_metric_ticks": bool(np.all(_ticks(carry.metrics.ticks) == expected)),
        # One sample gives no growth rate to project from.
        "integers_fit_1e10_ticks": None if intervals == 0 else not overflowing,
        "no_recompiles_after_warmup": compiles.count == 0,
        "rss_growth_under_limit": _within(rss_growth, args.max_rss_growth),
        "device_memory_flat": _within(device_growth, 0),
    }
    wall = time.perf_counter() - started
    _emit(
        {
            "mode": "env",
            "summary": True,
            "spec": args.spec,
            "env_id": env_id,
            "drift": description,
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "provenance": pp.reporting.provenance(),
            "lifetimes": args.batch,
            "agents_per_lifetime": env.num_agents,
            "ticks_per_lifetime": int(ticks[0]),
            "wall_s": round(wall, 3),
            "ticks_per_s": calls * args.chunk * args.batch / wall,
            "integer_leaves": len(names),
            "integers_projected_to_overflow": overflowing,
            "unsigned_projected_to_wrap": wrapping,
            "compile_events_after_warmup": compiles.count,
            "compile_events": compiles.events,
            "rss_growth_bytes": rss_growth,
            "device_growth_bytes": device_growth,
            "checks": checks,
        }
    )
    return _passed(checks)


HOST_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "metaworld": ("metaworld/MT1-v0", {"env_name": "reach-v3"}),
    "coom": ("coom/COOM-v0", {}),
    "envpool-cartpole": ("envpool/CartPole-v1", {"num_envs": 8, "num_threads": 4}),
    "envpool-atari": ("envpool/Atari-v0", {"num_envs": 4, "num_threads": 4}),
    "morpheus-mock": ("morpheus/Morpheus-v0", {}),
}


def _process_sample(label: str, tick: int, started: float, mode: str = "host") -> dict[str, Any]:
    children = _children()
    return {
        "mode": mode,
        "phase": label,
        "tick": tick,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "children": len(children),
        "zombies": sum(state == "Z" for _, state in children),
        "worker_rss_bytes": [_rss_bytes(pid) for pid, _ in children],
        "parent_rss_bytes": _rss_bytes(),
        "open_descriptors": _open_descriptors(),
    }


def _close(env: Any) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()


def host_soak(args: argparse.Namespace) -> bool:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The host soak reads /proc and runs on Linux only.")
    env_id, kwargs = HOST_SPECS[args.spec]
    baseline_fds = _open_descriptors()
    baseline_children = len(_children())
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    finite = True
    calls = max(args.ticks // args.chunk, 1)
    env = pp.make(env_id, **kwargs)
    try:
        runner = pp.runtime.HostRunner(env, pp.policies.uniform(env.action_space), ticks=args.chunk)
        carry = pp.runtime.init_lifetime(env, seed=0, window_size=args.chunk)
        for call in range(1, calls + 1):
            carry = runner(carry)
            finite = bool(
                np.all(np.isfinite(np.asarray(carry.obs)))
                and np.all(np.isfinite(np.asarray(carry.metrics.mean)))
            )
            if call % args.sample_every == 0 or call == calls or not finite:
                samples.append(_process_sample("long", call * args.chunk, started))
                _emit(samples[-1] | {"finite": finite})
            if not finite:
                break
        clock_exact = carry.state.t.to_int() == calls * args.chunk
        step_s = time.perf_counter() - started
    finally:
        _close(env)
    after_long = _process_sample("after-long", calls * args.chunk, started)
    _emit(after_long)
    for index in range(args.churn):
        short = pp.make(env_id, **kwargs)
        try:
            policy = pp.policies.uniform(short.action_space)
            state, obs, info = short.init(jax.random.key(index))
            for tick in range(4):
                action = policy(obs, info, jax.random.key(tick))
                state, obs, _, info = short.step(state, action, jax.random.key(tick))
        finally:
            _close(short)
        if (index + 1) % max(args.churn // 10, 1) == 0:
            _emit(_process_sample("churn", index + 1, started))
    after_churn = _process_sample("after-churn", args.churn, started)
    _emit(after_churn)
    parent_growth = _settled_growth([s["parent_rss_bytes"] for s in samples])
    worker_growth = (
        _settled_growth([s["worker_rss_bytes"][0] for s in samples if s["worker_rss_bytes"]]) or 0
    )
    checks = {
        "finite": finite,
        "exact_clock": clock_exact,
        "no_zombies": all(s["zombies"] == 0 for s in [*samples, after_long, after_churn]),
        "children_reaped_after_long": after_long["children"] == baseline_children,
        "children_reaped_after_churn": after_churn["children"] == baseline_children,
        "descriptors_back_to_baseline": after_churn["open_descriptors"] <= baseline_fds,
        "parent_rss_growth_under_limit": parent_growth is not None
        and parent_growth <= args.max_rss_growth,
        "worker_rss_growth_under_limit": worker_growth <= args.max_rss_growth,
    }
    _emit(
        {
            "mode": "host",
            "summary": True,
            "spec": args.spec,
            "env_id": env_id,
            "ticks": calls * args.chunk,
            "step_s": round(step_s, 3),
            "ticks_per_s": calls * args.chunk / step_s,
            "churned": args.churn,
            "baseline_descriptors": baseline_fds,
            "final_descriptors": after_churn["open_descriptors"],
            "parent_rss_settled_growth_bytes": parent_growth,
            "worker_rss_settled_growth_bytes": worker_growth,
            "checks": checks,
        }
    )
    return _passed(checks)


def _trainer(args: argparse.Namespace) -> Any:
    """The soaked learner: PPO (eight agents), DQN (one agent) or SAC (Pendulum)."""
    if args.mode == "ppo":
        from perpetua.agents import PPO, PPOConfig, PPORunner

        env = pp.make(
            "ContinualGrid-v0",
            size=8,
            num_agents=8,
            drift=pp.drift.Cyclic([(7, 7), (0, 0), (0, 7)], period=4096),
        )
        return PPORunner(env, PPO(action_size=5, config=PPOConfig(rollout_steps=args.rollout)))
    if args.mode == "dqn":
        from perpetua.agents import DQN, DQNConfig, DQNRunner

        env = pp.make(
            "ContinualGrid-v0", size=8, drift=pp.drift.Cyclic([(7, 7), (0, 0)], period=4096)
        )
        config = DQNConfig(
            rollout_steps=args.rollout,
            buffer_capacity=65536,
            batch_size=64,
            gradient_steps=max(args.rollout // 4, 1),
            target_update_interval=64,
        )
        return DQNRunner(env, DQN(action_size=5, config=config))
    from perpetua.agents import SAC, SACConfig, SACRunner

    env = pp.make("gymnax/Pendulum-v1", drift=pp.drift.Cyclic([(10.0,), (12.0,)], period=4096))
    box = cast(Any, env.action_space)
    config = SACConfig(
        rollout_steps=args.rollout,
        buffer_capacity=65536,
        batch_size=256,
        gradient_steps=max(args.rollout // 16, 1),
    )
    return SACRunner(env, SAC(action_low=box.low[0], action_high=box.high[0], config=config))


def learner(args: argparse.Namespace) -> bool:
    trainer = _trainer(args)
    state = trainer.init(seed=0, window_size=args.window)
    state = trainer(state)  # warm-up update, counted in the clock below
    # Compile the host-side sample reductions too, so only learner recompiles count.
    _ = (bool(state.healthy), _mean_of(state.lifetime.metrics.window_mean))
    _ = (state.lifetime.state.t.to_int(), _ticks(state.lifetime.metrics.ticks))
    calls = max(args.ticks // args.rollout, 1)
    first: dict[str, Any] | None = None
    record: dict[str, Any] = {}
    started = last_time = time.perf_counter()
    last_call = 0
    with pp.measurement.count_compilations() as compiles:
        for call in range(1, calls + 1):
            state = trainer(state)
            if call % args.sample_every and call != calls:
                continue
            healthy = bool(state.healthy)
            now = time.perf_counter()
            record = {
                "mode": args.mode,
                "update": call,
                "ticks": state.lifetime.state.t.to_int(),
                "elapsed_s": round(now - started, 3),
                "ticks_per_s": (call - last_call) * args.rollout / (now - last_time),
                "healthy": healthy,
                "window_reward": _mean_of(state.lifetime.metrics.window_mean),
                "rss_bytes": _rss_bytes(),
                "peak_rss_bytes": _peak_rss_bytes(),
                **_device_memory(),
            }
            last_time, last_call = now, call
            first = first or record
            _emit(record)
            if not healthy:
                break
    assert first is not None
    expected = (calls + 1) * args.rollout
    rss_growth = _growth(first["rss_bytes"], record["rss_bytes"])
    device_growth = _growth(first["device_bytes_in_use"], record["device_bytes_in_use"])
    checks = {
        "healthy": bool(state.healthy),
        "exact_clock": state.lifetime.state.t.to_int() == expected,
        "exact_metric_ticks": bool(np.all(_ticks(state.lifetime.metrics.ticks) == expected)),
        "exact_updates": state.updates.to_int() == calls + 1,
        "no_recompiles_after_warmup": compiles.count == 0,
        "rss_growth_under_limit": _within(rss_growth, args.max_rss_growth),
        "device_memory_flat": _within(device_growth, 0),
    }
    _emit(
        {
            "mode": args.mode,
            "summary": True,
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "provenance": pp.reporting.provenance(),
            "agents": trainer.env.num_agents,
            "rollout": args.rollout,
            "ticks": state.lifetime.state.t.to_int(),
            "updates": state.updates.to_int(),
            "gradient_steps": state.agent.gradient_steps.to_int(),
            "wall_s": round(time.perf_counter() - started, 3),
            "compile_events_after_warmup": compiles.count,
            "compile_events": compiles.events,
            "rss_growth_bytes": rss_growth,
            "device_growth_bytes": device_growth,
            "checks": checks,
        }
    )
    return _passed(checks)


def _mean_of(values: Any) -> float:
    return float(np.asarray(values).mean())


def worker(args: argparse.Namespace) -> bool:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The worker soak reads /proc and runs on Linux only.")
    baseline_fds = _open_descriptors()
    baseline_children = len(_children())
    make = {
        "grid_size": args.grid_size,
        "num_bots": args.bots,
        "replay_capacity": args.worker_ticks + 1,
    }
    rng = np.random.default_rng(0)
    actions = np.concatenate(
        [
            rng.uniform(-1, 1, (args.worker_ticks + 1, 2)),
            rng.integers(0, 3, (args.worker_ticks + 1, 1)),
        ],
        axis=1,
    ).astype(np.float32)
    env = pp.make("agarcl/agario-grid-v0", **make)
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    respawns = 0
    try:
        state, _, _ = env.init(jax.random.key(0))
        for tick in range(args.worker_ticks):
            state, _, _, info = env.step(
                state, jnp.asarray(actions[tick : tick + 1]), jax.random.key(tick)
            )
            respawns += bool(info["respawn"][0])
            if (tick + 1) % args.worker_sample_every == 0 or tick + 1 == args.worker_ticks:
                samples.append(_process_sample("long", tick + 1, started, "worker"))
                _emit(samples[-1] | {"respawns": respawns})
        step_s = time.perf_counter() - started
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "long"
            pp.checkpoint.save(path, state, metadata=env.metadata)
            _, expected_obs, expected_reward, _ = env.step(
                state, jnp.asarray(actions[-1:]), jax.random.key(args.worker_ticks)
            )
            fresh = pp.make("agarcl/agario-grid-v0", **make)
            try:
                template, _, _ = fresh.init(jax.random.key(99))
                replay_started = time.perf_counter()
                restored = pp.checkpoint.restore(path, template, expected_metadata=fresh.metadata)
                _, obs, reward, _ = fresh.step(
                    restored, jnp.asarray(actions[-1:]), jax.random.key(args.worker_ticks)
                )
                replay_s = time.perf_counter() - replay_started
                replay_exact = bool(
                    np.array_equal(np.asarray(obs), np.asarray(expected_obs))
                    and np.array_equal(np.asarray(reward), np.asarray(expected_reward))
                )
            finally:
                cast(Any, fresh).close()
    finally:
        cast(Any, env).close()
    after_long = _process_sample("after-long", args.worker_ticks, started, "worker")
    _emit(after_long)
    for index in range(args.churn):
        short = pp.make("agarcl/agario-grid-v0", grid_size=16, num_bots=2, respawn_after=7)
        try:
            state, _, _ = short.init(jax.random.key(index))
            for tick in range(16):
                state, *_ = short.step(
                    state, jnp.asarray(actions[tick : tick + 1]), jax.random.key(tick)
                )
        finally:
            cast(Any, short).close()
        if (index + 1) % max(args.churn // 10, 1) == 0:
            _emit(_process_sample("churn", index + 1, started, "worker"))
    after_churn = _process_sample("after-churn", args.churn, started, "worker")
    _emit(after_churn)
    worker_rss = [
        s["worker_rss_bytes"][0]
        for s in samples
        if s["worker_rss_bytes"] and s["worker_rss_bytes"][0]
    ]
    worker_growth = _settled_growth(worker_rss)
    checks = {
        "replay_exact": replay_exact,
        "no_zombies": all(s["zombies"] == 0 for s in [*samples, after_long, after_churn]),
        "children_reaped_after_long": after_long["children"] == baseline_children,
        "children_reaped_after_churn": after_churn["children"] == baseline_children,
        "descriptors_back_to_baseline": after_churn["open_descriptors"] <= baseline_fds,
        "worker_rss_growth_under_limit": worker_growth is not None
        and worker_growth <= args.max_rss_growth,
    }
    _emit(
        {
            "mode": "worker",
            "summary": True,
            "env": "agarcl/agario-grid-v0",
            "grid_size": args.grid_size,
            "bots": args.bots,
            "ticks": args.worker_ticks,
            "respawns": respawns,
            "step_s": round(step_s, 3),
            "ticks_per_s": args.worker_ticks / step_s,
            "full_replay_s": round(replay_s, 3),
            "replay_ticks_per_s": args.worker_ticks / replay_s,
            "churned_workers": args.churn,
            "baseline_descriptors": baseline_fds,
            "final_descriptors": after_churn["open_descriptors"],
            "worker_rss_first_bytes": worker_rss[0] if worker_rss else None,
            "worker_rss_last_bytes": worker_rss[-1] if worker_rss else None,
            "worker_rss_settled_growth_bytes": worker_growth,
            "checks": checks,
        }
    )
    return _passed(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    native_parser = modes.add_parser("native")
    native_parser.add_argument("--ticks", type=int, default=1_000_000_000)
    native_parser.add_argument("--batch", type=int, default=4096)
    native_parser.add_argument("--chunk", type=int, default=4096)
    native_parser.add_argument("--window", type=int, default=128)
    native_parser.add_argument("--sample-every", type=int, default=16)
    native_parser.add_argument("--max-rss-growth", type=int, default=64 * 2**20)
    native_parser.add_argument("--require-gpu", action="store_true")
    env_parser = modes.add_parser("env")
    env_parser.add_argument("--spec", help="a registered JAX-native id, optionally +scripted")
    env_parser.add_argument("--list", action="store_true", help="print the specs and exit")
    env_parser.add_argument("--ticks", type=int, default=100_000_000, help="per lifetime")
    env_parser.add_argument("--batch", type=int, default=2)
    env_parser.add_argument("--chunk", type=int, default=8192)
    env_parser.add_argument("--window", type=int, default=1024)
    env_parser.add_argument("--sample-every", type=int, default=128)
    env_parser.add_argument("--max-rss-growth", type=int, default=64 * 2**20)
    env_parser.add_argument("--require-gpu", action="store_true")
    host_parser = modes.add_parser("host")
    host_parser.add_argument("--spec", choices=sorted(HOST_SPECS), required=True)
    host_parser.add_argument("--ticks", type=int, default=1_000_000)
    host_parser.add_argument("--chunk", type=int, default=1000)
    host_parser.add_argument("--sample-every", type=int, default=10)
    host_parser.add_argument("--churn", type=int, default=200)
    host_parser.add_argument("--max-rss-growth", type=int, default=64 * 2**20)
    for name, rollout in (("ppo", 1024), ("dqn", 32), ("sac", 128)):
        learner_parser = modes.add_parser(name)
        learner_parser.add_argument("--ticks", type=int, default=100_000_000)
        learner_parser.add_argument("--rollout", type=int, default=rollout)
        learner_parser.add_argument("--window", type=int, default=1024)
        learner_parser.add_argument("--sample-every", type=int, default=1000)
        learner_parser.add_argument("--max-rss-growth", type=int, default=64 * 2**20)
        learner_parser.add_argument("--require-gpu", action="store_true")
    worker_parser = modes.add_parser("worker")
    worker_parser.add_argument("--worker-ticks", type=int, default=1_000_000)
    worker_parser.add_argument("--worker-sample-every", type=int, default=10_000)
    worker_parser.add_argument("--grid-size", type=int, default=16)
    worker_parser.add_argument("--bots", type=int, default=8)
    worker_parser.add_argument("--churn", type=int, default=200)
    worker_parser.add_argument("--max-rss-growth", type=int, default=64 * 2**20)
    args = parser.parse_args()
    if getattr(args, "require_gpu", False) and jax.default_backend() != "gpu":
        raise RuntimeError("GPU soak requested, but JAX is not using a GPU.")
    if args.mode == "env" and (args.list or not args.spec):
        print("\n".join(env_specs()))
        sys.exit(0 if args.list else 2)
    modes_by_name = {"native": native, "env": env_soak, "host": host_soak, "worker": worker}
    passed = modes_by_name.get(args.mode, learner)(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
