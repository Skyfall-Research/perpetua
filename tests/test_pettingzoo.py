"""Both PettingZoo directions must preserve lifetimes, masks and privilege isolation."""

# Third-party/JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo import EnvSpec, ParallelEnv, make  # pyright: ignore[reportMissingTypeStubs]
from pettingzoo.utils.wrappers import BaseParallelWrapper  # pyright: ignore[reportMissingTypeStubs]

import perpetua as pp
from perpetua.adapters import PettingZooAdapter, PettingZooLifetime
from perpetua.agents import IPPO, HostPPORunner, PPOConfig
from perpetua.core import EnvInfo
from perpetua.envs.continual_grid import GridParams, GridState
from perpetua.envs.cooperative_grid import CooperativeGrid
from perpetua.params import DriftEnv


def spread() -> ParallelEnv[str, Any, Any]:
    spec = EnvSpec(
        id="mpe2/simple_spread-v3", entry_point="mpe2.simple_spread.simple_spread:parallel_env"
    )
    return cast(ParallelEnv[str, Any, Any], make("parallel", spec, N=3, max_cycles=2))


def test_parallel_view_passes_official_api_without_restarting_world() -> None:
    view = PettingZooAdapter(pp.make("CooperativeGrid-v0", size=3), horizon=8)
    try:
        cast(Any, import_module("pettingzoo.test")).parallel_api_test(view, num_cycles=20)
        _, info = view.reset()
        assert info["agent_0"]["t"].to_int() == 16
        assert view.observation_space("agent_0") is view.observation_space("agent_0")
        assert view.action_space("agent_0") is view.action_space("agent_0")
        with pytest.raises(NotImplementedError):
            view.state()
    finally:
        view.close()


def test_parallel_view_truncation_keeps_state_rng_and_rejects_reseed() -> None:
    first = PettingZooAdapter(pp.make("CooperativeGrid-v0", size=3), horizon=2)
    second = PettingZooAdapter(pp.make("CooperativeGrid-v0", size=3))
    try:
        obs_a, _ = first.reset(seed=0)
        obs_b, _ = second.reset(seed=0)
        for name in obs_a:
            np.testing.assert_array_equal(obs_a[name], obs_b[name])
        last = None
        for _ in range(2):
            actions = dict.fromkeys(first.agents, 0)
            a = first.step(actions)
            b = second.step(actions)
            last = a
            for name in a[0]:
                np.testing.assert_array_equal(a[0][name], b[0][name])
                assert a[1][name] == b[1][name]
                assert a[4][name]["t"].to_int() == b[4][name]["t"].to_int()
        assert last is not None
        assert not first.agents
        assert all(last[3].values())
        assert first.step({}) == ({}, {}, {}, {}, {})
        obs, info = first.reset(options={"legacy_setting": 1})
        for name in obs:
            np.testing.assert_array_equal(obs[name], last[0][name])
            assert info[name]["t"].to_int() == 2
        with pytest.raises(ValueError, match="seed"):
            first.reset(seed=0)
        actions = dict.fromkeys(first.agents, 0)
        a, b = first.step(actions), second.step(actions)
        for name in a[0]:
            np.testing.assert_array_equal(a[0][name], b[0][name])
            assert a[4][name]["t"].to_int() == 3
            assert a[4][name]["action_mask"].dtype == np.int8
    finally:
        first.close()
        second.close()


def test_view_checks_action_dict_and_only_exposes_explicit_oracle() -> None:
    base = pp.make("CooperativeGrid-v0", size=3)
    view = PettingZooAdapter(pp.Oracle(base, expose=["phase"]))
    try:
        with pytest.raises(RuntimeError, match="reset"):
            view.step({"agent_0": 0, "agent_1": 0})
        _, infos = view.reset(seed=0)
        assert set(infos["agent_0"]["oracle"]) == {"phase"}
        with pytest.raises(ValueError, match="agents"):
            view.step({"agent_0": 0})
        with pytest.raises(ValueError, match="space"):
            view.step({"agent_0": 99, "agent_1": 0})
        _, _, _, _, info = view.step({"agent_0": 0, "agent_1": 0})
        assert info["agent_0"]["t"].to_int() == 1
    finally:
        view.close()


def test_real_mpe_lift_respawns_without_public_done_or_partial_checkpoint(tmp_path: Path) -> None:
    first = PettingZooLifetime(spread())
    second = PettingZooLifetime(spread())
    try:
        configuration = cast(dict[str, Any], first.metadata.get("configuration"))
        assert configuration["backend"]["legacy_max_cycles"] == 2
        a = pp.runtime.init_lifetime(first, seed=0, window_size=4)
        b = pp.runtime.init_lifetime(second, seed=0, window_size=4)
        initial = a
        for _ in range(6):
            a = pp.runtime.tick(first, pp.policies.random, a)
            b = pp.runtime.tick(second, pp.policies.random, b)
            np.testing.assert_array_equal(a.obs, b.obs)
            np.testing.assert_array_equal(a.metrics.total, b.metrics.total)
            assert a.obs.shape == (3, 18)
            assert a.obs.dtype == jnp.float32
            assert set(a.info) == {"action_mask", "active_mask", "respawn", "space_change", "t"}
        assert a.state.t.to_int() == 6
        assert np.all(np.asarray(a.info["respawn"]))
        with pytest.raises(ValueError, match="stale"):
            pp.runtime.tick(first, pp.policies.random, initial)
        assert first.metadata.get("checkpointable") is False
        with pytest.raises(NotImplementedError, match="checkpoint"):
            pp.checkpoint.save(tmp_path / "unsupported", a, metadata=first.metadata)
        assert not (tmp_path / "unsupported").exists()
    finally:
        first.close()
        second.close()


def test_ippo_runs_on_a_real_pettingzoo_cpu_environment() -> None:
    env = PettingZooLifetime(spread())
    try:
        agent = IPPO(
            action_size=5, config=PPOConfig(rollout_steps=4, epochs=1, minibatches=1, hidden_size=8)
        )
        trainer = HostPPORunner(env, agent)
        state = trainer(trainer.init(seed=0))
        assert state.lifetime.state.t.to_int() == 4
        assert bool(state.healthy)
        np.testing.assert_array_equal(state.agent.gradient_steps.lo, [1, 1, 1])
        assert state.statistics.loss.shape == (3,)
    finally:
        env.close()


class MaskedObservation(BaseParallelWrapper[str, Any, Any]):
    """Expose a real native view using PettingZoo's masked-observation convention."""

    def observation_space(self, agent: str) -> spaces.Dict:
        return spaces.Dict(
            {
                "observation": self.env.observation_space(agent),
                "action_mask": spaces.MultiBinary(
                    int(cast(spaces.Discrete, self.env.action_space(agent)).n)
                ),
            }
        )

    @staticmethod
    def masked(obs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        return {
            name: {"observation": value, "action_mask": info[name]["action_mask"]}
            for name, value in obs.items()
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.masked(obs, info), info

    def step(self, actions: dict[str, int]) -> Any:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self.masked(obs, info), reward, terminated, truncated, info


def test_lift_of_masked_view_retains_masks_and_honors_lifetime_reset_ownership() -> None:
    raw = MaskedObservation(PettingZooAdapter(pp.make("CooperativeGrid-v0", size=3), horizon=2))
    env = PettingZooLifetime(raw)
    try:
        state = pp.runtime.init_lifetime(env, seed=0)
        invalid = np.asarray(state.info["action_mask"])
        slot, action = np.argwhere(~invalid)[0]
        actions = np.zeros(2, dtype=np.int32)
        actions[slot] = action
        with pytest.raises(ValueError, match="mask"):
            env.step(state.state, actions, state.env_key)
        runner = pp.runtime.HostRunner(env, pp.policies.random, ticks=4)
        state = runner(state)
        assert state.state.t.to_int() == 4
        _, raw_info = raw.reset()
        assert raw_info["agent_0"]["t"].to_int() == 4
        np.testing.assert_array_equal(
            state.info["action_mask"][0], raw_info["agent_0"]["action_mask"]
        )
    finally:
        env.close()


def test_synthetic_horizon_of_a_lifted_view_reports_no_respawn_of_its_own() -> None:
    def respawns(horizon: int | None) -> list[list[bool]]:
        view = PettingZooAdapter(pp.make("CooperativeGrid-v0", size=3), horizon=horizon)
        env = PettingZooLifetime(view)
        try:
            state = pp.runtime.init_lifetime(env, seed=0)
            flags: list[list[bool]] = []
            for _ in range(24):
                state = pp.runtime.tick(env, pp.policies.greedy, state)
                flags.append(np.asarray(state.info["respawn"]).tolist())
            return flags
        finally:
            env.close()

    truncated, continuous = respawns(2), respawns(None)
    # The horizon ends a reporting segment, not the world: the same world
    # without it defines the respawns, some of which fall on a truncating tick.
    assert truncated == continuous
    assert truncated[0] == truncated[1] == [False, False]
    assert any(all(tick) for tick in truncated[1::2])
    assert not all(all(tick) for tick in truncated[1::2])


def test_lift_initial_activity_and_legacy_privilege_are_not_confused() -> None:
    class PausedGrid(CooperativeGrid):
        def init(self, key: jax.Array, params: GridParams) -> tuple[GridState, jax.Array, EnvInfo]:
            state, _, _ = super().init(key, params)
            state = state._replace(active=jnp.asarray([True, False]))
            obs, info = self.observe(state, params)
            return state, obs, info

    raw = PettingZooAdapter(
        pp.Oracle(DriftEnv(PausedGrid(size=3), env_id="CooperativeGrid-v0"), expose=["phase"])
    )
    env = PettingZooLifetime(raw)
    try:
        state = pp.runtime.init_lifetime(env, seed=0)
        np.testing.assert_array_equal(state.state.env.active, [True, False])
        np.testing.assert_array_equal(state.info["active_mask"], [True, False])
        np.testing.assert_array_equal(state.obs[1], np.zeros(7))
        np.testing.assert_array_equal(
            state.info["action_mask"][1], [True, False, False, False, False]
        )
        assert "oracle" not in state.info
        assert env.metadata["oracle_exposure"] == ()
    finally:
        env.close()


@pytest.mark.parametrize("seed", [2**32, -1])
def test_a_parallel_reset_seed_that_jax_would_alias_is_refused(seed: int) -> None:
    view = PettingZooAdapter(pp.make("CooperativeGrid-v0", size=3))
    try:
        with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 2\*\*32\)"):
            view.reset(seed=seed)
        view.reset(seed=2**32 - 1)
    finally:
        view.close()
