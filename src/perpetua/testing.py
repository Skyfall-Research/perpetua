"""Check that an environment honours the lifetime contract.

``check_env(factory)`` is the counterpart of PettingZoo's ``api_test``: give
it a zero-argument function that builds a fresh environment and it runs every
contract check, raising :class:`ContractError` with a specific message on the
first failure:

- **reproducibility:** two instances with the same seeds and actions produce
  the same states and rewards;
- **continuity:** one chunk of ``2n`` ticks equals two chunks of ``n``, the
  clock counts every tick, and the rewards are finite (the lifetime's metrics
  stay ``healthy``);
- **privilege:** none of the environment's declared ``oracle_fields``
  (``boundary``, ``task_id``, ``phase``, ``drift_parameters`` and the per-field
  aliases) appears in public ``info``; they reach it only through
  :class:`~perpetua.Oracle`, which records the exposure;
- **shapes:** observations, masks and events keep their shapes, ``info`` carries
  the required keys, and ``info`` and the state keep one structure; for discrete
  actions the action mask is ``(agents, actions)`` and every active agent has a
  valid action; rewards are finite;
- **types:** observations have the dtype their space declares, masks and event
  flags are boolean, rewards are floating point, and a JAX world's state keeps
  the same types from ``init`` through ``step`` (a weakly typed leaf would make
  every runner compile twice);
- **compilation:** a second chunk compiles nothing;
- **checkpoints:** a saved lifetime resumes exactly, unless the environment's
  metadata declares ``checkpointable`` False (JAX worlds default to True, host
  simulators to False).

The shape, type and reproducibility checks call ``init`` and ``step``
directly: four ticks with the all-zero action, then ``ticks`` ticks with
actions sampled from a fixed key (:func:`perpetua.policies.uniform`: among the
actions each tick's ``action_mask`` allows for discrete slots, with inactive
slots sent action 0, and uniformly within bounds for a ``Box``, which must be
bounded). The other checks run uniformly random lifetimes through a runner and
also fail when a reward is not finite. A fault that random play does not reach
within ``ticks`` ticks is not seen, so raise ``ticks`` for worlds whose rewards
or masks only change after a longer walk.

Host simulators are checked through :class:`~perpetua.runtime.HostRunner`; the
type and compilation checks apply to JAX worlds only.

The checks assume a :class:`~perpetua.params.DriftEnv`-style state (with a
``t`` clock) and an environment whose Oracle can expose
``drift_parameters``. Outside that shape, or when saving or restoring a
checkpoint fails, the error surfaces as it is raised (for example
``AttributeError`` or ``ValueError``) rather than as :class:`ContractError`.

``reward_action_sensitivity(env)`` is a separate diagnostic, not a contract
check: it measures how often a tick's reward depends on that tick's action,
which is what same-state regret (``reference=`` on a runner) can see.
"""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from perpetua import checkpoint, measurement, policies, runtime
from perpetua.core import Environment, Oracle, as_int

__all__ = ["ContractError", "check_env", "reward_action_sensitivity"]

_REQUIRED_INFO = ("action_mask", "active_mask", "respawn", "space_change", "t")
_DRIFT_FIELDS = ("phase", "drift_parameters", "boundary", "task_id")
_DEPRECATED_FIELDS = ("drift_changed",)


class ContractError(AssertionError):
    """An environment broke the lifetime contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _close(env: Any) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()


def _same(left: object, right: object, what: str) -> None:
    """Compare numeric leaves, ignoring opaque host ownership handles."""
    _require(jax.tree.structure(left) == jax.tree.structure(right), f"{what}: structures differ.")
    numeric = (jax.Array, np.ndarray, int, float, bool, np.generic)
    a_leaves = [leaf for leaf in jax.tree.leaves(left) if isinstance(leaf, numeric)]
    b_leaves = [leaf for leaf in jax.tree.leaves(right) if isinstance(leaf, numeric)]
    _require(len(a_leaves) == len(b_leaves), f"{what}: leaf counts differ.")
    for a, b in zip(a_leaves, b_leaves, strict=True):
        if isinstance(a, jax.Array) and jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(cast(jax.Array, b))
        _require(
            np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True),
            f"{what}: values differ.",
        )


def _types(state: object) -> list[Any]:
    """The abstract type of every state leaf (container classes need not define ``==``)."""
    return [jax.typeof(leaf) for leaf in jax.tree.leaves(state)]


_ZERO_TICKS = 4


def _action(env: Environment[Any]) -> jax.Array:
    space = cast(Any, env.action_space)
    if hasattr(space, "nvec"):
        return jnp.zeros((env.num_agents,), dtype=jnp.int32)
    return jnp.zeros(space.shape, dtype=jnp.float32)


def _sampler(env: Environment[Any]) -> Callable[[Any, Any, int], jax.Array]:
    """The direct checks' action per tick: all-zero first, then sampled from the masks."""
    space = cast(Any, env.action_space)
    discrete = hasattr(space, "nvec")
    policy = policies.uniform(space)
    base = jax.random.key(0)

    def action(obs: Any, info: Any, tick: int) -> jax.Array:
        if tick < _ZERO_TICKS:
            return _action(env)
        sampled = policy(obs, info, jax.random.fold_in(base, tick))
        if discrete:
            return jnp.where(jnp.asarray(info["active_mask"]), sampled, 0).astype(jnp.int32)
        return sampled

    return action


def _healthy(carry: Any, what: str) -> None:
    _require(
        bool(carry.metrics.healthy),
        f"{what}: a uniformly random lifetime produced a non-finite reward.",
    )


def _host(env: Environment[Any]) -> bool:
    return env.metadata.get("execution") == "host"


def _runner(env: Environment[Any], ticks: int) -> Any:
    policy = policies.uniform(cast(Any, env.action_space))
    if _host(env):
        return runtime.HostRunner(env, policy, ticks=ticks)
    return runtime.Runner(env, policy, ticks=ticks)


def _reproducibility(factory: Callable[[], Environment[Any]], ticks: int) -> None:
    first, second = factory(), factory()
    try:
        a, b = first.init(jax.random.key(0)), second.init(jax.random.key(0))
        _same(a[0], b[0], "reproducibility: init state")
        _same(a[1], b[1], "reproducibility: init observation")
        act, act_second = _sampler(first), _sampler(second)
        for tick in range(_ZERO_TICKS + ticks):
            a = first.step(a[0], act(a[1], a[-1], tick), jax.random.key(tick))
            b = second.step(b[0], act_second(b[1], b[-1], tick), jax.random.key(tick))
            _same(a[0], b[0], f"reproducibility: state after tick {tick + 1}")
            _same(a[2], b[2], f"reproducibility: reward at tick {tick + 1}")
    finally:
        _close(first)
        _close(second)


def _continuity(factory: Callable[[], Environment[Any]], ticks: int) -> None:
    env, chunked_env = factory(), factory()
    try:
        whole = _runner(env, 2 * ticks)(runtime.init_lifetime(env, seed=1, window_size=4))
        short = _runner(chunked_env, ticks)
        chunked = short(short(runtime.init_lifetime(chunked_env, seed=1, window_size=4)))
        _same(whole.state, chunked.state, "continuity: one chunk vs two")
        _healthy(whole, "continuity")
        _healthy(chunked, "continuity")
        _require(
            whole.state.t.to_int() == 2 * ticks and chunked.info["t"].to_int() == 2 * ticks,
            "continuity: the lifetime clock must count every tick.",
        )
    finally:
        _close(env)
        _close(chunked_env)


def _privilege(factory: Callable[[], Environment[Any]]) -> None:
    env, wrapped = factory(), factory()
    try:
        state, _, info = env.init(jax.random.key(0))
        fields = tuple(env.oracle_fields)
        private = {*_DRIFT_FIELDS, *_DEPRECATED_FIELDS, *fields}
        leaked = sorted(private & set(info))
        _require(not leaked, f"privilege: public info leaks {leaked}.")
        recorded = set(env.metadata["oracle_exposure"])
        shown = set(cast(dict[str, Any], cast(Any, info).get("oracle", {})))
        _require(
            shown <= recorded,
            f"privilege: info['oracle'] shows {sorted(shown - recorded)} without recording "
            "them in metadata['oracle_exposure'].",
        )
        _require(
            set(_DRIFT_FIELDS) <= set(fields),
            f"privilege: oracle_fields must include {list(_DRIFT_FIELDS)} "
            f"(missing {sorted(set(_DRIFT_FIELDS) - set(fields))}); 'drift_changed' is only "
            "the deprecated alias of 'boundary'.",
        )
        missing = sorted(set(fields) - set(env.privileged(state)))
        _require(not missing, f"privilege: privileged() lacks declared fields {missing}.")
        oracle = Oracle(wrapped, expose=["drift_parameters"])
        _, _, exposed = oracle.init(jax.random.key(0))
        oracle_info = cast(dict[str, Any], exposed).get("oracle")
        _require(
            isinstance(oracle_info, dict) and set(oracle_info) == {"drift_parameters"} | recorded,
            "privilege: the Oracle must expose exactly the requested fields.",
        )
        _require(
            "drift_parameters" in oracle.metadata["oracle_exposure"],
            "privilege: the Oracle must record its exposure in metadata.",
        )
    finally:
        _close(env)
        _close(wrapped)


def _public(env: Environment[Any], obs: Any, info: Any, where: str) -> None:
    """The dtype and mask checks every observation and ``info`` must pass."""
    space = cast(Any, env.observation_space)
    _require(
        np.dtype(obs.dtype) == np.dtype(space.dtype),
        f"types: observation is {obs.dtype} but the space declares {space.dtype} {where}.",
    )
    flags = cast(dict[str, Any], info)
    for key in ("action_mask", "active_mask", "respawn", "space_change"):
        _require(
            np.dtype(flags[key].dtype) == np.dtype(bool),
            f"types: info[{key!r}] must be boolean, got {flags[key].dtype} {where}.",
        )
    mask = np.asarray(flags["action_mask"])
    _require(
        mask.ndim >= 1 and mask.shape[0] == env.num_agents,
        f"shapes: action_mask must lead with the agent axis ({env.num_agents},), "
        f"got {mask.shape} {where}.",
    )
    choices = getattr(env.action_space, "nvec", None)
    if choices is not None:
        expected = (env.num_agents, int(np.max(choices)))
        _require(
            mask.shape == expected,
            f"shapes: action_mask must have shape {expected}, got {mask.shape} {where}.",
        )
        active = np.asarray(flags["active_mask"])
        _require(
            bool(np.all(mask[active].any(axis=-1))),
            f"shapes: an active agent has no valid action {where}.",
        )


def _shapes_and_types(factory: Callable[[], Environment[Any]], ticks: int) -> None:
    env = factory()
    try:
        state, obs, info = env.init(jax.random.key(0))
        missing = [key for key in _REQUIRED_INFO if key not in info]
        _require(not missing, f"shapes: info lacks {missing}.")
        space = cast(Any, env.observation_space)
        _require(
            tuple(obs.shape) == tuple(space.shape),
            f"shapes: observation {obs.shape} does not match the space {space.shape}.",
        )
        for key in ("active_mask", "respawn", "space_change"):
            _require(
                tuple(cast(dict[str, Any], info)[key].shape) == (env.num_agents,),
                f"shapes: info[{key!r}] must have shape ({env.num_agents},).",
            )
        _public(env, obs, info, "at init")
        first_mask = tuple(info["action_mask"].shape)
        first_info = jax.tree.structure(info)
        first_state = jax.tree.structure(state)
        first_types = None if _host(env) else _types(state)
        act = _sampler(env)
        for tick in range(_ZERO_TICKS + ticks):
            state, obs, reward, info = env.step(state, act(obs, info, tick), jax.random.key(tick))
            _require(
                tuple(obs.shape) == tuple(space.shape),
                f"shapes: observation changed shape at tick {tick + 1}.",
            )
            _require(
                jax.tree.structure(info) == first_info,
                f"shapes: info from step differs in structure from init at tick {tick + 1}.",
            )
            _require(
                jax.tree.structure(state) == first_state,
                f"shapes: state from step differs in structure from init at tick {tick + 1}.",
            )
            _require(
                tuple(info["action_mask"].shape) == first_mask,
                f"shapes: action_mask changed shape at tick {tick + 1}.",
            )
            _public(env, obs, info, f"at tick {tick + 1}")
            _require(
                tuple(reward.shape) == (env.num_agents,),
                f"shapes: reward must have shape ({env.num_agents},) at tick {tick + 1}.",
            )
            _require(
                np.issubdtype(np.dtype(reward.dtype), np.floating),
                f"types: reward must be floating point, got {reward.dtype} at tick {tick + 1}.",
            )
            _require(
                bool(np.all(np.isfinite(np.asarray(reward)))),
                f"shapes: non-finite reward at tick {tick + 1}.",
            )
        if first_types is not None:
            _require(
                _types(state) == first_types,
                "types: state types differ between init and step (often a weakly typed "
                "leaf built from a Python scalar); every runner would compile twice.",
            )
    finally:
        _close(env)


def _compilation(factory: Callable[[], Environment[Any]], ticks: int) -> None:
    env = factory()
    try:
        if _host(env):
            return
        runner = _runner(env, ticks)
        carry = runner(runtime.init_lifetime(env, seed=3, window_size=4))
        with measurement.count_compilations() as compiles:
            carry = jax.block_until_ready(runner(carry))
        _healthy(carry, "compilation")
        _require(
            compiles.count == 0,
            f"compilation: a second chunk compiled {compiles.count} time(s); the carry "
            "must keep its shapes and types.",
        )
    finally:
        _close(env)


def _checkpoint(factory: Callable[[], Environment[Any]], ticks: int) -> None:
    env, target = factory(), factory()
    try:
        if not env.metadata.get("checkpointable", not _host(env)):
            return
        runner, target_runner = _runner(env, ticks), _runner(target, ticks)
        carry = runner(runtime.init_lifetime(env, seed=2, window_size=4))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "lifetime"
            checkpoint.save(path, carry, metadata=runner.metadata)
            restored = checkpoint.restore(
                path,
                runtime.init_lifetime(target, seed=9, window_size=4),
                expected_metadata=target_runner.metadata,
            )
        resumed = runner(carry)
        _same(resumed, target_runner(restored), "checkpoints: resumed lifetime")
        _healthy(resumed, "checkpoints")
    finally:
        _close(env)
        _close(target)


def check_env(factory: Callable[[], Environment[Any]], *, ticks: int = 4) -> None:
    """Run every lifetime-contract check on fresh environments from ``factory``.

    ``factory`` must build a new, identically configured environment on each
    call (several checks compare two instances). ``ticks`` is the chunk length
    used by the runner-based checks and the number of sampled-action ticks the
    direct checks step after their four all-zero ticks. Raises :class:`ContractError` on the first
    failure the checks detect (other errors propagate unchanged, see the
    module docstring); raises ``ValueError`` unless ``ticks`` is an integer
    >= 1; returns ``None`` when the environment passes.
    """
    ticks = as_int(ticks, "ticks must be a positive integer.", minimum=1)
    # Direct init/step checks first, so a structural fault is named before a
    # compiled runner trips over it.
    _shapes_and_types(factory, ticks)
    _privilege(factory)
    _reproducibility(factory, ticks)
    _continuity(factory, ticks)
    _compilation(factory, ticks)
    _checkpoint(factory, ticks)


def reward_action_sensitivity(
    env: Environment[Any], *, seed: int = 0, ticks: int = 1000, actions: int = 8
) -> float:
    """Fraction of ticks on which the reward depends on that tick's action.

    Follows one uniformly random lifetime for ``ticks`` ticks. At every tick it
    draws ``actions`` joint actions from the same policy (valid actions only,
    one per agent slot, as :func:`perpetua.policies.uniform` samples them),
    steps each from the same state with the same step key, and counts the tick
    when any slot's reward differs between two of them. The lifetime continues
    with the first action.

    ``0.0`` means no action changed the reward of its own tick on the states
    visited, so same-state regret (``reference=`` on a runner) is exactly zero
    there for every pair of policies. Either the reward cannot depend on the
    tick's action at all (CartPole pays for the pole's state before the action
    takes effect), or it does only in states random play did not reach. A
    value above zero is the share of visited states in which regret can tell
    two actions apart; it says nothing about how large the difference is.

    JAX worlds run compiled. Host simulators with snapshot support are stepped
    in Python, ``actions + 1`` simulator steps per tick; host backends that
    cannot step one state twice (no snapshot support, or a ``compiled=True``
    lift) raise ``ValueError``, as they do for a regret reference. So do
    unbounded ``Box`` action spaces, which have no uniform policy, and
    ``ticks`` < 1 or ``actions`` < 2.
    """
    ticks = as_int(ticks, "ticks must be a positive integer.", minimum=1)
    actions = as_int(actions, "actions must be an integer >= 2.", minimum=2)
    metadata = env.metadata
    if metadata.get("forkable", True) is False or (
        _host(env) and not metadata.get("checkpointable", False)
    ):
        raise ValueError(
            "reward_action_sensitivity steps several actions from the same state; this host "
            "backend has no snapshot support (or is a compiled=True lift with one single-use "
            "state), so it cannot. Use a snapshot-capable host backend or a JAX world."
        )
    policy = policies.uniform(cast(Any, env.action_space))
    init_key, env_key, agent_key = jax.random.split(
        runtime.seed_key(seed, "reward_action_sensitivity"), 3
    )
    state, obs, info = env.init(init_key)

    def split(env_key: jax.Array, agent_key: jax.Array) -> tuple[jax.Array, ...]:
        env_key, step_key = jax.random.split(env_key)
        agent_key, act_key = jax.random.split(agent_key)
        return env_key, agent_key, step_key, jax.random.split(act_key, actions)

    if _host(env):
        differing = 0
        for _ in range(ticks):
            env_key, agent_key, step_key, act_keys = split(env_key, agent_key)
            candidates = [policy(obs, info, key) for key in act_keys]
            rewards = [np.asarray(env.step(state, a, step_key)[2]) for a in candidates[1:]]
            state, obs, reward, info = env.step(state, candidates[0], step_key)
            differing += any(not np.array_equal(np.asarray(reward), other) for other in rewards)
        return differing / ticks

    def body(carry: Any, _: None) -> tuple[Any, jax.Array]:
        state, obs, info, env_key, agent_key = carry
        env_key, agent_key, step_key, act_keys = split(env_key, agent_key)

        def outcome(act_key: jax.Array) -> Any:
            return env.step(state, policy(obs, info, act_key), step_key)

        def first(leaf: jax.Array) -> jax.Array:
            return leaf[0]

        outcomes = jax.vmap(outcome)(act_keys)
        state, obs, _, info = jax.tree.map(first, outcomes)
        rewards = outcomes[2]
        return (state, obs, info, env_key, agent_key), jnp.any(rewards != rewards[0])

    _, differed = jax.lax.scan(body, (state, obs, info, env_key, agent_key), length=ticks)
    return float(np.mean(np.asarray(differed)))
