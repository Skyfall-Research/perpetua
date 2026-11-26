"""Sequential AEC worlds must lift into synchronous ticks and match the parallel lift."""

# Optional simulator/JAX interfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo import AECEnv, EnvSpec, ParallelEnv, make  # pyright: ignore[reportMissingTypeStubs]

import perpetua as pp
from perpetua.adapters import AECLifetime, PettingZooLifetime


def aec_spread(**kwargs: Any) -> AECEnv[str, Any, Any]:
    spec = EnvSpec(id="mpe2/simple_spread-v3", entry_point="mpe2.simple_spread.simple_spread:env")
    return cast(AECEnv[str, Any, Any], make("aec", spec, N=3, max_cycles=4, **kwargs))


def parallel_spread(**kwargs: Any) -> ParallelEnv[str, Any, Any]:
    spec = EnvSpec(
        id="mpe2/simple_spread-v3", entry_point="mpe2.simple_spread.simple_spread:parallel_env"
    )
    return cast(ParallelEnv[str, Any, Any], make("parallel", spec, N=3, max_cycles=4, **kwargs))


class CountingAEC(AECEnv[str, np.ndarray, int]):
    """Minimal sequential env: each agent terminates after its own step limit."""

    metadata = {"name": "counting_v0"}

    def __init__(self, limits: tuple[int, ...] = (1, 3, 3)) -> None:
        self.possible_agents = [f"agent_{i}" for i in range(len(limits))]
        self._limits = dict(zip(self.possible_agents, limits, strict=True))

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        del agent
        return spaces.Box(0.0, 10.0, (1,), dtype=np.float32)

    def action_space(self, agent: str) -> spaces.Space[Any]:
        del agent
        return spaces.Discrete(2)

    def _next_live(self, agent: str) -> str:
        order = self.possible_agents
        index = order.index(agent)
        for offset in range(1, len(order) + 1):
            candidate = order[(index + offset) % len(order)]
            if candidate in self.agents:
                return candidate
        raise RuntimeError("no live agent remains")

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del seed, options
        self.agents = self.possible_agents[:]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[0]
        self.counts = dict.fromkeys(self.agents, 0)

    def observe(self, agent: str) -> np.ndarray:
        return np.asarray([self.counts[agent]], dtype=np.float32)

    def step(self, action: int | None) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # The AEC dead-step contract requires None; the stub types it as int.
            self._was_dead_step(cast(int, action))
            if self.agents and self.agent_selection not in self.agents:
                self.agent_selection = self._next_live(self.agent_selection)
            return
        agent = self.agent_selection
        self._clear_rewards()
        self.counts[agent] += 1
        self.rewards[agent] = float(action or 0)
        self.terminations[agent] = self.counts[agent] >= self._limits[agent]
        self.agent_selection = self._next_live(agent)
        self._accumulate_rewards()

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_aec_lifetime_public_contract_and_opaque_refusal() -> None:
    env = AECLifetime(aec_spread())
    try:
        assert env.metadata.get("checkpointable") is False
        assert env.metadata.get("env_id") == "pettingzoo-aec/simple_spread_v3"
        initial = pp.runtime.init_lifetime(env, seed=0)
        assert initial.obs.shape == (3, 18)
        assert initial.obs.dtype == jnp.float32
        np.testing.assert_array_equal(initial.info["action_mask"], np.ones((3, 5), dtype=bool))
        np.testing.assert_array_equal(initial.info["active_mask"], [True, True, True])
        assert set(initial.info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
        step = pp.runtime.tick(env, pp.policies.random, initial)
        assert step.state.t.to_int() == 1
        assert not np.any(np.asarray(step.info["respawn"]))
        configuration = env.metadata.get("configuration")
        assert isinstance(configuration, dict)
        backend = configuration["backend"]
        assert isinstance(backend, dict)
        assert backend["backend"] == "pettingzoo-aec"
        assert backend["cycle"] == "one-joint-tick-per-full-aec-cycle"
    finally:
        env.close()


def test_aec_cycle_matches_the_parallel_lift_exactly() -> None:
    sequential = AECLifetime(aec_spread())
    parallel = PettingZooLifetime(parallel_spread())
    try:
        left = pp.runtime.init_lifetime(sequential, seed=0)
        right = pp.runtime.init_lifetime(parallel, seed=0)
        np.testing.assert_array_equal(np.asarray(left.obs), np.asarray(right.obs))
        actions = [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]

        def fixed(action: int) -> pp.runtime.Policy:
            return lambda obs, info, key: jnp.full(3, action, dtype=jnp.int32)

        for tick, action in enumerate(actions):
            left = pp.runtime.tick(sequential, fixed(action), left)
            right = pp.runtime.tick(parallel, fixed(action), right)
            np.testing.assert_array_equal(np.asarray(left.obs), np.asarray(right.obs))
            np.testing.assert_array_equal(
                np.asarray(left.info["active_mask"]), np.asarray(right.info["active_mask"])
            )
            np.testing.assert_array_equal(
                np.asarray(left.info["respawn"]), np.asarray(right.info["respawn"])
            )
            assert left.state.t.to_int() == right.state.t.to_int() == tick + 1
        assert bool(jnp.any(left.info["respawn"]))  # crossed the legacy cycle limit
    finally:
        sequential.close()
        parallel.close()


def test_aec_slots_deactivate_then_respawn_without_clock_reset() -> None:
    env = AECLifetime(CountingAEC(limits=(1, 3, 3)))
    try:
        initial = pp.runtime.init_lifetime(env, seed=0)
        state, obs, reward, info = env.step(
            initial.state, jnp.asarray([1, 1, 0]), jax.random.key(0)
        )
        assert state.t.to_int() == 1
        np.testing.assert_array_equal(info["active_mask"], [False, True, True])
        np.testing.assert_array_equal(np.asarray(reward), [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(np.asarray(info["action_mask"][0]), [True, False])
        np.testing.assert_array_equal(np.asarray(info["action_mask"][1]), [True, True])
        assert float(np.asarray(obs)[0, 0]) == 0.0  # dead slot is zero-padded
        state, obs, reward, info = env.step(state, jnp.asarray([0, 1, 1]), jax.random.key(1))
        np.testing.assert_array_equal(info["active_mask"], [False, True, True])
        np.testing.assert_array_equal(np.asarray(reward), [0.0, 1.0, 1.0])
        state, obs, reward, info = env.step(state, jnp.asarray([0, 0, 0]), jax.random.key(2))
        assert state.t.to_int() == 3
        assert np.all(np.asarray(info["respawn"]))
        np.testing.assert_array_equal(info["active_mask"], [True, True, True])
        np.testing.assert_array_equal(np.asarray(obs), np.zeros((3, 1)))
    finally:
        env.close()


def test_agent_dying_while_an_earlier_agent_lives_goes_inactive_and_gets_none() -> None:
    env = AECLifetime(CountingAEC(limits=(3, 1, 3)))
    try:
        initial = pp.runtime.init_lifetime(env, seed=0)
        state, obs, reward, info = env.step(
            initial.state, jnp.asarray([1, 1, 0]), jax.random.key(0)
        )
        # agent_1 hit its limit; the selector is back on agent_0, which lives.
        np.testing.assert_array_equal(info["active_mask"], [True, False, True])
        np.testing.assert_array_equal(np.asarray(reward), [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(np.asarray(info["action_mask"][1]), [True, False])
        assert float(np.asarray(obs)[1, 0]) == 0.0
        # The dead slot's action is ignored: the agent is sent None, not 1.
        state, obs, reward, info = env.step(state, jnp.asarray([0, 1, 1]), jax.random.key(1))
        np.testing.assert_array_equal(info["active_mask"], [True, False, True])
        np.testing.assert_array_equal(np.asarray(reward), [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(np.asarray(obs), [[2.0], [0.0], [2.0]])
        assert not np.any(np.asarray(info["respawn"]))
        state, obs, reward, info = env.step(state, jnp.asarray([1, 0, 1]), jax.random.key(2))
        assert state.t.to_int() == 3
        np.testing.assert_array_equal(np.asarray(reward), [1.0, 0.0, 1.0])
        assert np.all(np.asarray(info["respawn"]))
        np.testing.assert_array_equal(info["active_mask"], [True, True, True])
    finally:
        env.close()


class KillingAEC(CountingAEC):
    """The first agent's action 1 terminates the last agent before its turn."""

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        live = not self.terminations[agent]
        super().step(cast(int, action))
        if live and agent == "agent_0" and action == 1:
            self.terminations["agent_2"] = True
            self._deads_step_first()


def test_agent_killed_before_its_turn_does_not_act_in_that_cycle() -> None:
    env = AECLifetime(KillingAEC(limits=(9, 9, 9)))
    try:
        initial = pp.runtime.init_lifetime(env, seed=0)
        state, obs, reward, info = env.step(
            initial.state, jnp.asarray([0, 1, 1]), jax.random.key(0)
        )
        np.testing.assert_array_equal(np.asarray(obs), [[1.0], [1.0], [1.0]])
        state, obs, reward, info = env.step(state, jnp.asarray([1, 1, 1]), jax.random.key(1))
        assert state.t.to_int() == 2
        np.testing.assert_array_equal(info["active_mask"], [True, True, False])
        # agent_2 never took its second step, so its submitted action earned nothing.
        np.testing.assert_array_equal(np.asarray(reward), [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(np.asarray(obs), [[2.0], [2.0], [0.0]])
        state, obs, reward, info = env.step(state, jnp.asarray([0, 1, 1]), jax.random.key(2))
        np.testing.assert_array_equal(info["active_mask"], [True, True, False])
        np.testing.assert_array_equal(np.asarray(reward), [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(np.asarray(obs), [[3.0], [3.0], [0.0]])
    finally:
        env.close()


class PausingAEC(CountingAEC):
    """The middle agent lives on but reports itself inactive after its first step."""

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        super().step(cast(int, action))
        if agent == "agent_1":
            self.infos[agent] = {"active_mask": False}


def test_action_of_a_live_but_inactive_slot_is_replaced_by_zero() -> None:
    raw = PausingAEC(limits=(9, 9, 9))
    env = AECLifetime(raw)
    try:
        initial = pp.runtime.init_lifetime(env, seed=0)
        state, _, _, info = env.step(initial.state, jnp.asarray([1, 1, 1]), jax.random.key(0))
        np.testing.assert_array_equal(info["active_mask"], [True, False, True])
        state, _, reward, _ = env.step(state, jnp.asarray([1, 1, 1]), jax.random.key(1))
        np.testing.assert_array_equal(np.asarray(reward), [1.0, 0.0, 1.0])
        assert raw.counts["agent_1"] == 2  # it still stepped, with action 0
    finally:
        env.close()


def test_selector_left_on_a_removed_agent_is_refused() -> None:
    raw = CountingAEC(limits=(9, 9, 9))
    env = AECLifetime(raw)
    try:
        initial = pp.runtime.init_lifetime(env, seed=0)
        raw.agent_selection = "agent_gone"
        with pytest.raises(ValueError, match="selector on 'agent_gone'"):
            env.step(initial.state, jnp.asarray([1, 1, 1]), jax.random.key(0))
    finally:
        env.close()


@pytest.mark.parametrize("name", ["connect_four_v3", "tictactoe_v3"])
def test_turn_based_classic_games_are_refused_at_construction(name: str) -> None:
    classic = pytest.importorskip("pettingzoo.classic")
    env = getattr(classic, name).env()
    assert env.metadata["is_parallelizable"] is False
    with pytest.raises(ValueError, match=rf"turn-based games: {name} .*is_parallelizable"):
        AECLifetime(env)


class TurnBasedAEC(CountingAEC):
    metadata = {"name": "turns_v0", "is_parallelizable": False}


class DeclaredParallelAEC(CountingAEC):
    metadata = {"name": "declared_v0", "is_parallelizable": True}


def test_is_parallelizable_false_is_refused_and_an_undeclared_key_is_accepted() -> None:
    with pytest.raises(ValueError, match=r"does not support turn-based games: turns_v0"):
        AECLifetime(cast(Any, TurnBasedAEC()))
    assert "is_parallelizable" not in CountingAEC.metadata
    for env in (CountingAEC(), DeclaredParallelAEC()):
        lifetime = AECLifetime(cast(Any, env))
        state, _, _ = lifetime.init(jax.random.key(0))
        lifetime.step(state, jnp.zeros((3,), dtype=jnp.int32), jax.random.key(1))
