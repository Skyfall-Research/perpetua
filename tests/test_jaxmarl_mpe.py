"""Generic MPE tasks: discrete and continuous actions, drift and compilation."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false, reportPrivateUsage=false

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua import registration

SUPPORTED = registration._JAXMARL_MPE


def action_for(env: Any) -> jax.Array:
    if hasattr(env.action_space, "nvec"):
        return jnp.zeros(env.num_agents, dtype=jnp.int32)
    return jnp.zeros(env.action_space.shape, dtype=jnp.float32)


def policy_for(env: Any) -> Any:
    if hasattr(env.action_space, "nvec"):
        return pp.policies.random
    shape = env.action_space.shape

    def zero_action(obs: Any, info: Any, key: Any) -> jax.Array:
        return jnp.zeros(shape, dtype=jnp.float32)

    return zero_action


@pytest.mark.parametrize("name", SUPPORTED)
def test_supported_mpe_tasks_construct_and_step(name: str) -> None:
    env = pp.make(f"jaxmarl/{name}")
    state, obs, info = env.init(jax.random.key(0))
    assert obs.shape[0] == env.num_agents
    space = cast(Any, env.action_space)
    width = int(space.nvec[0]) if hasattr(space, "nvec") else int(space.shape[-1])
    assert info["action_mask"].shape == (env.num_agents, width)
    for tick in range(3):
        state, obs, reward, info = env.step(state, action_for(env), jax.random.key(tick))
        assert reward.shape == (env.num_agents,)
    assert info["t"].to_int() == 3
    assert "phase" not in info
    assert "drift_parameters" not in info


def test_mpe_dynamics_drift_changes_physics() -> None:
    env = pp.make("jaxmarl/MPE_simple_v3", drift={"damping": pp.drift.Cyclic([0.25, 0.9], 4)})
    state, _, _ = env.init(jax.random.key(0))
    for tick in range(6):
        state, _, _, _ = env.step(state, jnp.zeros(1, dtype=jnp.int32), jax.random.key(tick))
    assert abs(float(env.privileged(state)["damping_parameters"]) - 0.9) < 1e-6
    assert int(state.drift["damping"].phase) == 1


def test_mpe_masks_cover_per_agent_action_counts() -> None:
    env = pp.make("jaxmarl/MPE_simple_speaker_listener_v4")
    _, _, info = env.init(jax.random.key(0))
    np.testing.assert_array_equal(
        np.asarray(info["action_mask"]),
        [[True, True, True, False, False], [True, True, True, True, True]],
    )


def test_continuous_masks_cover_padded_action_dimensions() -> None:
    env = pp.make("jaxmarl/MPE_simple_world_comm_v3")
    _, _, info = env.init(jax.random.key(0))
    assert env.action_space.shape == (6, 9)
    np.testing.assert_array_equal(np.asarray(info["action_mask"]).sum(axis=1), [9, 5, 5, 5, 5, 5])


@pytest.mark.parametrize("name", SUPPORTED)
def test_supported_mpe_tasks_compile_chunk_and_checkpoint(name: str, tmp_path: Path) -> None:
    env = pp.make(f"jaxmarl/{name}")
    policy = policy_for(env)
    long_run = pp.runtime.Runner(env, policy, ticks=4)
    short = pp.runtime.Runner(env, policy, ticks=2)
    whole = long_run(pp.runtime.init_lifetime(env, seed=1, window_size=4))
    chunked = short(short(pp.runtime.init_lifetime(env, seed=1, window_size=4)))
    for a, b in zip(jax.tree.leaves(whole.state), jax.tree.leaves(chunked.state), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    path = tmp_path / "lifetime"
    pp.checkpoint.save(path, whole, metadata=long_run.metadata)
    restored = pp.checkpoint.restore(
        path,
        pp.runtime.init_lifetime(env, seed=9, window_size=4),
        expected_metadata=long_run.metadata,
    )
    assert restored.state.t.to_int() == 4


def test_continuous_mpe_runs_through_sac() -> None:
    from perpetua.agents import SAC, SACConfig, SACRunner

    env = pp.make("jaxmarl/MPE_simple_facmac_3a_v1")
    config = SACConfig(
        rollout_steps=2, buffer_capacity=16, batch_size=4, gradient_steps=1, hidden_size=8
    )
    space = cast(Any, env.action_space)
    agent = SAC(action_low=space.low[0], action_high=space.high[0], config=config)
    state = SACRunner(env, agent).init(seed=0)
    state = SACRunner(env, agent)(state)
    assert bool(state.healthy)
    assert state.agent.gradient_steps.to_int() == 1


def learner_mean(env: Any, ticks: int = 1024) -> float:
    policy = policy_for(env)
    carry = pp.runtime.init_lifetime(env, seed=0, window_size=ticks)
    final = pp.runtime.Runner(env, policy, ticks=ticks)(carry)
    learners = np.asarray(final.info["active_mask"])
    return float(np.asarray(final.metrics.mean)[learners].sum())


@pytest.mark.parametrize(
    "name",
    [
        "MPE_simple_speaker_listener_v4",
        "MPE_simple_push_v3",
        "MPE_simple_adversary_v3",
        "MPE_simple_reference_v3",
    ],
)
def test_scripted_partners_help_and_opponents_hurt_the_learners(name: str) -> None:
    def scripted(behavior: str, beta: float) -> Any:
        return pp.make(
            f"jaxmarl/{name}",
            num_scripted=1,
            scripted_behavior=behavior,
            scripted_drift=pp.drift.Cyclic([(beta,)], 1),  # one phase: beta held constant
        )

    partner = learner_mean(scripted("partner", 1.0))
    random = learner_mean(scripted("partner", 0.0))
    opponent = learner_mean(scripted("opponent", 1.0))
    assert partner > random > opponent, (partner, random, opponent)


def test_scripted_slots_are_inactive_and_ignore_their_actions() -> None:
    env = pp.make("jaxmarl/MPE_simple_tag_v3", num_scripted=1)
    state, obs, info = env.init(jax.random.key(0))
    assert info["active_mask"].tolist() == [True, True, True, False]
    assert info["action_mask"][3].tolist() == [True, False, False, False, False]
    np.testing.assert_array_equal(np.asarray(obs[3]), 0.0)
    assert np.asarray(obs[0]).any()
    first = env.step(state, jnp.asarray([1, 2, 3, 0]), jax.random.key(1))
    second = env.step(state, jnp.asarray([1, 2, 3, 4]), jax.random.key(1))
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    adversary = pp.make("jaxmarl/MPE_simple_tag_v3", scripted_slots=(0,))
    _, _, info = adversary.init(jax.random.key(0))
    assert info["active_mask"].tolist() == [False, True, True, True]
    assert cast(Any, adversary.metadata)["configuration"]["scripted_slots"] == [0]


def test_scripted_draw_key_is_neither_the_world_nor_the_respawn_key() -> None:
    env = cast(Any, pp.make("jaxmarl/MPE_simple_push_v3", num_scripted=1))
    state, _, _ = env.init(jax.random.key(0))
    seen: dict[str, jax.Array] = {}
    inner = env.env
    original = inner._scripted_actions

    def recording(world: Any, selected: Any, world_key: Any, draw_key: Any, *rest: Any) -> Any:
        seen["world"], seen["draw"] = world_key, draw_key
        return original(world, selected, world_key, draw_key, *rest)

    inner._scripted_actions = recording
    key = jax.random.key(5)
    env.step(state, jnp.zeros((env.num_agents,), jnp.int32), key)
    world, reset = jax.random.split(key)

    def data(k: jax.Array) -> np.ndarray:
        return np.asarray(jax.random.key_data(k))

    np.testing.assert_array_equal(data(seen["world"]), data(world))
    for other in (world, reset):
        assert not np.array_equal(data(seen["draw"]), data(other))
    for width in (2, 3, 4):
        for child in jax.random.split(key, width)[:2]:
            assert not np.array_equal(data(seen["draw"]), data(child))


def test_facmac_prey_slots_are_backend_scripted() -> None:
    env = pp.make("jaxmarl/MPE_simple_facmac_3a_v1")
    _, _, info = env.init(jax.random.key(0))
    assert info["active_mask"].tolist() == [True, True, True, False]
    assert not np.asarray(info["action_mask"][3]).any()
    assert cast(Any, env.metadata)["configuration"]["backend_scripted_slots"] == [3]
    scripted = pp.make("jaxmarl/MPE_simple_facmac_3a_v1", num_scripted=1)
    _, _, info = scripted.init(jax.random.key(0))
    assert info["active_mask"].tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="among"):
        pp.make("jaxmarl/MPE_simple_facmac_3a_v1", scripted_slots=(3,))


def test_scripted_drift_is_hidden_checkpointed_and_validated(tmp_path: Path) -> None:
    env = pp.make(
        "jaxmarl/MPE_simple_push_v3",
        num_scripted=1,
        scripted_drift=pp.drift.Cyclic([(1.0,), (0.0,)], 3),
    )
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=4)
    carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
    assert int(carry.state.drift["scripted"].phase) == 1
    assert "scripted_parameters" not in carry.info
    path = tmp_path / "scripted"
    pp.checkpoint.save(path, carry, metadata=runner.metadata)
    template = pp.runtime.init_lifetime(env, seed=9, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=runner.metadata)
    for a, b in zip(jax.tree.leaves(runner(carry)), jax.tree.leaves(runner(restored)), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    oracle = pp.Oracle(
        pp.make("jaxmarl/MPE_simple_push_v3", num_scripted=1), expose=["scripted_parameters"]
    )
    state, _, _ = oracle.init(jax.random.key(0))
    _, _, _, info = oracle.step(state, jnp.zeros(2, dtype=jnp.int32), jax.random.key(1))
    np.testing.assert_array_equal(cast(Any, info)["oracle"]["scripted_parameters"], [1.0])
    for kwargs in (
        {"num_scripted": 2},
        {"num_scripted": 1, "scripted_slots": (0,)},
        {"scripted_slots": (0, 0)},
        {"scripted_slots": (5,)},
        {"num_scripted": 1, "scripted_behavior": "neutral"},
    ):
        with pytest.raises(ValueError):  # noqa: PT011 - several messages
            pp.make("jaxmarl/MPE_simple_push_v3", **kwargs)
    with pytest.raises(ValueError, match="learner"):
        pp.make("jaxmarl/MPE_simple_v3", num_scripted=1)


def test_metadata_does_not_claim_every_action_is_valid_with_scripted_slots() -> None:
    env = pp.make("jaxmarl/MPE_simple_tag_v3", num_scripted=1)
    _, _, info = env.init(jax.random.key(0))
    assert not bool(np.asarray(info["action_mask"]).all())
    assert cast(Any, env.metadata)["configuration"]["action_masks"] != "all-valid"


def test_drifted_view_radius_reaches_the_observations() -> None:
    from perpetua.adapters.jaxmarl_mpe import JaxMARLMPE
    from perpetua.params import DriftEnv

    name = "MPE_simple_facmac_v1"
    default = np.asarray(JaxMARLMPE(name).backend.view_radius, np.float32)

    def observations(scale: float) -> np.ndarray:
        values = np.stack([default, default * scale])
        env: Any = DriftEnv(
            cast(Any, JaxMARLMPE(name)), {"view_radius": pp.drift.Cyclic(values, period=3)}
        )
        state, obs, _ = env.init(jax.random.key(0))
        step: Any = jax.jit(env.step)
        seen = [np.asarray(obs)]
        for tick in range(12):
            action = jax.random.uniform(jax.random.key(100 + tick), env.action_space.shape)
            state, obs, _, _ = step(state, action, jax.random.key(tick))
            seen.append(np.asarray(obs))
        return np.stack(seen)

    same, shrunk = observations(1.0), observations(0.1)
    np.testing.assert_array_equal(same[:3], shrunk[:3])  # the first phase is the default
    assert not np.array_equal(same, shrunk)  # identical before: get_obs used the default


def test_a_discrete_learner_on_a_continuous_mpe_task_is_refused_at_init() -> None:
    from perpetua.agents import IPPO, PPO, PPORunner

    env = pp.make("jaxmarl/MPE_simple_facmac_3a_v1")
    for agent in (PPO(action_size=5), IPPO(action_size=5)):
        with pytest.raises(ValueError, match=r"needs a discrete action space.*pp\.agents\.SAC"):
            PPORunner(env, agent).init(seed=0)
