"""PyTorch agents on Perpetua lifetimes through the DLPack bridge."""

# torch ships partial type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp

torch = pytest.importorskip("torch", reason="optional torch extra")

from perpetua.adapters import GymnasiumLifetime  # noqa: E402
from perpetua.torch_bridge import (  # noqa: E402
    TorchPPO,
    TorchRunner,
    from_torch,
    to_torch,
)


def runner_for(env: Any, seed: int = 0) -> TorchRunner[Any]:
    return TorchRunner(env, TorchPPO(env.observation_space, env.action_space, seed=seed))


def test_dlpack_round_trip_preserves_values_and_dtypes() -> None:
    array = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    tensor = to_torch(array)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (2, 3)
    back = from_torch(tensor * 2)
    np.testing.assert_array_equal(np.asarray(back), np.asarray(array) * 2)
    mask = from_torch(torch.tensor([True, False]))
    assert mask.dtype == jnp.bool_


def test_torch_ppo_learns_above_random_on_a_continuing_grid() -> None:
    env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
    runner = runner_for(env)
    state = runner.init(seed=0)
    for _ in range(64):
        state = runner(state)
    assert runner.healthy
    assert state.updates.to_int() == 64
    assert state.lifetime.state.t.to_int() == 64 * 128
    reference = pp.runtime.Runner(env, pp.policies.random, ticks=64 * 128)(
        pp.runtime.init_lifetime(env, seed=0)
    )
    learned = float(np.asarray(state.lifetime.metrics.window_mean).mean())
    assert learned > float(np.asarray(reference.metrics.mean).mean()) + 0.1
    assert {"loss", "policy_loss", "value_loss", "entropy"} <= set(state.statistics)


def test_checkpoint_resume_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=3, num_agents=2)
    straight = runner_for(env, seed=5)
    expected = straight.init(seed=1)
    for _ in range(4):
        expected = straight(expected)

    first = runner_for(env, seed=5)
    state = first.init(seed=1)
    for _ in range(2):
        state = first(state)
    first.save(tmp_path / "torch", state)
    with pytest.raises(FileExistsError):
        first.save(tmp_path / "torch", state)

    resumed = runner_for(env, seed=99)  # different seed: the checkpoint must win
    state = resumed.restore(tmp_path / "torch", resumed.init(seed=7))
    for _ in range(2):
        state = resumed(state)
    np.testing.assert_array_equal(
        np.asarray(state.lifetime.metrics.total), np.asarray(expected.lifetime.metrics.total)
    )
    np.testing.assert_array_equal(np.asarray(state.lifetime.obs), np.asarray(expected.lifetime.obs))
    for name, value in straight.agent.state_dict()["network"].items():
        assert torch.equal(value, resumed.agent.state_dict()["network"][name]), name


def test_generic_checkpoint_functions_refuse_a_torch_runner(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=3, num_agents=2)
    runner = runner_for(env, seed=5)
    start = runner.init(seed=1)
    state = runner(start)
    refusal = "Use TorchRunner.save and TorchRunner.restore"
    # The agent's weights are not in the carry, so each of these wrote or
    # replayed a checkpoint without them.
    for name, carry in (("start", start), ("trained", state)):
        with pytest.raises(TypeError, match=refusal):
            pp.checkpoint.save(tmp_path / name, carry, metadata=runner.metadata)
        assert not (tmp_path / name).exists()
    runner.save(tmp_path / "torch", state)
    lifetime = tmp_path / "torch" / "lifetime"
    with pytest.raises(TypeError, match=refusal):
        pp.checkpoint.restore(lifetime, state, expected_metadata=runner.metadata)
    generic = cast(Any, runner)
    for template in (state, start, (state.lifetime, state.updates)):
        with pytest.raises(TypeError, match=refusal):
            pp.checkpoint.replay(lifetime, template, generic, chunks=1)
    with pytest.raises(TypeError, match=refusal):
        pp.checkpoint.replay(
            lifetime, state, lambda carry: carry, chunks=1, expected_metadata=runner.metadata
        )


def test_masks_are_respected_and_oracle_info_flows_through() -> None:
    env = pp.Oracle(
        pp.make("DoorGrid-v0", drift=pp.drift.Cyclic([[1, 1, 1, 1], [1, 0, 1, 0]], 4)),
        expose=["boundary"],
    )
    agent = TorchPPO(env.observation_space, env.action_space, seed=0)
    lifetime = pp.runtime.init_lifetime(env, seed=0)
    for tick in range(40):
        public = {
            "action_mask": to_torch(lifetime.info["action_mask"], agent.device),
            "active_mask": to_torch(lifetime.info["active_mask"], agent.device),
        }
        with torch.no_grad():
            action, _ = agent.act(to_torch(lifetime.obs, agent.device), public)
        allowed = np.asarray(lifetime.info["action_mask"])
        assert allowed[np.arange(env.num_agents), action.numpy()].all(), tick
        lifetime, _ = pp.runtime.transition(env, lifetime, jnp.asarray(action.numpy()))
    assert "oracle" in lifetime.info

    seen: list[dict[str, Any]] = []

    class Recording(TorchPPO):
        def act(self, obs: Any, info: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            seen.append(info)
            return super().act(obs, info)

    runner = TorchRunner(
        env, Recording(env.observation_space, env.action_space, seed=0), rollout_steps=8
    )
    plain = TorchRunner(
        env, TorchPPO(env.observation_space, env.action_space, seed=0), rollout_steps=8
    )
    state, expected = runner(runner.init(seed=0)), plain(plain.init(seed=0))
    np.testing.assert_array_equal(
        np.asarray(state.lifetime.metrics.total), np.asarray(expected.lifetime.metrics.total)
    )
    assert "boundary" in cast(Any, runner.metadata["environment"])["oracle_exposure"]
    assert [int(info["t"]) for info in seen] == list(range(8))
    assert all(info["t"].dtype == torch.int64 for info in seen)
    boundaries = [bool(info["oracle"]["boundary"]) for info in seen]
    assert boundaries == [tick == 4 for tick in range(8)]
    for info in seen:
        assert set(info) == {
            "action_mask",
            "active_mask",
            "respawn",
            "space_change",
            "t",
            "oracle",
        }
        assert info["respawn"].dtype == info["space_change"].dtype == torch.bool
        assert info["respawn"].shape == info["space_change"].shape == (env.num_agents,)
    bare = TorchRunner(
        pp.make("DoorGrid-v0"), Recording(env.observation_space, env.action_space, seed=0)
    )
    assert "oracle" not in bare._public(bare.init(seed=0).lifetime.info)  # pyright: ignore[reportPrivateUsage]


def test_host_simulators_run_through_the_same_runner() -> None:
    env = GymnasiumLifetime(gym.make("CartPole-v1"))
    try:
        assert env.metadata.get("execution") == "host"
        runner = runner_for(env)
        state = runner(runner.init(seed=0))
        assert state.lifetime.state.t.to_int() == 128
        assert bool(state.lifetime.metrics.healthy)
    finally:
        env.close()


def test_invalid_settings_are_refused() -> None:
    env = pp.make("gymnax/Pendulum-v1")
    with pytest.raises(ValueError, match="discrete"):
        TorchPPO(env.observation_space, env.action_space)  # pyright: ignore[reportArgumentType]
    grid = pp.make("ContinualGrid-v0")
    with pytest.raises(ValueError, match="rollout_steps"):
        TorchRunner(grid, TorchPPO(grid.observation_space, grid.action_space), rollout_steps=0)


def test_restore_refuses_arbitrary_pickled_objects(tmp_path: Path) -> None:
    """Checkpoints load tensors and plain values only, never arbitrary Python objects."""
    env = pp.make("ContinualGrid-v0", size=3, num_agents=2)
    runner = runner_for(env)
    state = runner.init(seed=0)
    runner.save(tmp_path / "torch", state)
    torch.save({"payload": Path("not-a-tensor")}, tmp_path / "torch" / "agent.pt")
    with pytest.raises(Exception, match="(?i)weights.only|unsupported|global"):
        runner.restore(tmp_path / "torch", runner.init(seed=0))


class _StateOnly:
    """The parts of a TorchAgent that healthy reads; a custom agent's key layout."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def state_dict(self) -> dict[str, Any]:
        return self.state


def test_health_checks_every_saved_tensor_not_only_a_network_key() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    runner = runner_for(env)
    assert runner.healthy
    finite = {"actor": {"w": torch.ones(2)}, "optimizer": {"state": {0: {"m": torch.zeros(2)}}}}
    runner.agent = cast(Any, _StateOnly(finite))
    assert runner.healthy
    runner.agent = cast(Any, _StateOnly({"actor": {"w": torch.tensor([1.0, float("nan")])}}))
    assert not runner.healthy  # no "network" key: previously always healthy
    moments = [{"m": torch.tensor([float("inf")])}]
    runner.agent = cast(Any, _StateOnly({"network": {}, "optimizer": {"state": moments}}))
    assert not runner.healthy
    runner.agent = cast(Any, _StateOnly({"step": torch.tensor(3), "lr": 0.1}))
    assert runner.healthy  # integer tensors and plain numbers are not parameters


def test_torch_ppo_refuses_rollouts_that_do_not_divide_into_minibatches() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    agent = TorchPPO(env.observation_space, env.action_space, seed=0)
    with pytest.raises(ValueError, match=r"samples \(129\) must divide evenly into 4 minibatches"):
        TorchRunner(env, agent, rollout_steps=129)


def test_torch_ppo_refuses_one_sample_minibatches_at_construction() -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    agent = TorchPPO(env.observation_space, env.action_space, seed=0)
    with pytest.raises(ValueError, match=r"leave 1 sample per minibatch.*at least 2"):
        TorchRunner(env, agent, rollout_steps=4)
    TorchRunner(env, agent, rollout_steps=8)
    # The update guards itself too, for a runner built around a different agent.
    runner = TorchRunner(env, agent, rollout_steps=8)
    runner.rollout_steps = 4
    with pytest.raises(ValueError, match="at least 2"):
        runner(runner.init(seed=0))
    assert runner.healthy


def test_restoring_under_another_torch_version_names_the_framework(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    runner = runner_for(env)
    runner.save(tmp_path / "saved", runner(runner.init(seed=0)))
    other = runner_for(env)
    other.metadata["framework"] = "torch-0.0.1"
    with pytest.raises(ValueError, match=r"framework.*torch-0\.0\.1"):
        other.restore(tmp_path / "saved", other.init(seed=0))


def test_runner_accepts_an_environment_whose_info_has_no_space_change() -> None:
    class Bare:
        def __init__(self, env: Any) -> None:
            self.env = env
            for name in ("num_agents", "observation_space", "action_space", "oracle_fields"):
                setattr(self, name, getattr(env, name))
            self.metadata = env.metadata

        def _public(self, info: Any) -> Any:
            return {name: value for name, value in info.items() if name != "space_change"}

        def init(self, key: Any) -> Any:
            state, obs, info = self.env.init(key)
            return state, obs, self._public(info)

        def step(self, state: Any, action: Any, key: Any) -> Any:
            state, obs, reward, info = self.env.step(state, action, key)
            return state, obs, reward, self._public(info)

        def observe(self, state: Any) -> Any:
            obs, info = self.env.observe(state)
            return obs, self._public(info)

        def privileged(self, state: Any) -> Any:
            return self.env.privileged(state)

    seen: list[set[str]] = []

    class Spy(TorchPPO):
        def act(self, obs: Any, info: Any) -> Any:
            seen.append(set(info))
            return super().act(obs, info)

    env: Any = Bare(pp.make("ContinualGrid-v0", size=3))
    runner = TorchRunner(env, Spy(env.observation_space, env.action_space, seed=0), rollout_steps=8)
    runner(runner.init(seed=0))
    assert seen
    assert "space_change" not in seen[0]
    assert {"action_mask", "active_mask", "respawn", "t"} <= seen[0]


def _rollout_with_a_gap(placeholder: float, active: list[bool]) -> tuple[Any, Any]:
    from gymnasium import spaces

    from perpetua.torch_bridge import TorchPPOConfig, TorchRollout

    steps, features, actions = len(active), 4, 3
    agent = TorchPPO(
        spaces.Box(-10, 10, (1, features), np.float32),
        spaces.MultiDiscrete([actions]),
        config=TorchPPOConfig(hidden_size=16, minibatches=1, epochs=1),
        seed=0,
    )
    generator = torch.Generator().manual_seed(1)
    obs = torch.randn(steps, 1, features, generator=generator)
    last_obs = torch.randn(1, features, generator=generator)
    acted = torch.tensor(active).reshape(steps, 1)
    obs[~acted[:, 0]] = placeholder
    mask = torch.ones(steps, 1, actions, dtype=torch.bool)
    with torch.no_grad():
        _, values = agent.network(obs)
    rollout = TorchRollout(
        obs,
        torch.zeros(steps, 1, dtype=torch.int64),
        torch.tensor([0.3, -0.2, 0.7, 0.4, 0.5, 0.1]).reshape(steps, 1),
        mask,
        acted,
        last_obs,
        {"value": values, "log_prob": torch.zeros(steps, 1)},
    )
    return agent, rollout


def test_torch_ppo_advantages_skip_inactive_ticks_as_the_jax_gae_does() -> None:
    from perpetua.agents.ppo import compute_gae

    active = [True, True, False, False, True, True]
    results: list[np.ndarray] = []
    for placeholder in (0.0, 5.0):
        agent, rollout = _rollout_with_a_gap(placeholder, active)
        advantages, returns = agent.advantages(rollout)
        with torch.no_grad():
            _, last_value = agent.network(rollout.last_obs)
        expected, targets = compute_gae(
            rewards=jnp.asarray(rollout.reward.numpy()),
            values=jnp.asarray(rollout.extras["value"].numpy()),
            last_value=jnp.asarray(last_value.numpy()),
            gamma=agent.config.gamma,
            gae_lambda=agent.config.gae_lambda,
            active=jnp.asarray(rollout.active.numpy()),
        )
        np.testing.assert_allclose(advantages.numpy(), np.asarray(expected), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(returns.numpy(), np.asarray(targets), rtol=1e-5, atol=1e-6)
        results.append(advantages.numpy()[np.asarray(active)])
    # The placeholder observation's value differs, the active ticks' advantages do not.
    np.testing.assert_array_equal(results[0], results[1])


def test_torch_ppo_advantages_are_unchanged_when_every_tick_is_active() -> None:
    agent, rollout = _rollout_with_a_gap(0.0, [True] * 6)
    advantages, returns = agent.advantages(rollout)
    gamma, lam = agent.config.gamma, agent.config.gae_lambda
    values = rollout.extras["value"]
    with torch.no_grad():
        _, last_value = agent.network(rollout.last_obs)
    expected = torch.zeros_like(rollout.reward)
    running = torch.zeros_like(last_value)
    for step in reversed(range(6)):
        following = last_value if step == 5 else values[step + 1]
        delta = rollout.reward[step] + gamma * following - values[step]
        running = delta + gamma * lam * running
        expected[step] = running
    np.testing.assert_array_equal(advantages.numpy(), expected.numpy())
    np.testing.assert_array_equal(returns.numpy(), (expected + values).numpy())
    assert set(agent.update(rollout)) == {"loss", "policy_loss", "value_loss", "entropy"}
