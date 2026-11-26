"""CNN torso: pixel observations train through the existing learners."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import load_script, unavailable

import perpetua as pp
from perpetua.agents import DQN, IPPO, IQL, PPO, DQNConfig, HostPPORunner, PPOConfig
from perpetua.agents.dqn import DQNRunner
from perpetua.agents.encoders import CNNTorso, derive_pixel_scale
from perpetua.agents.ppo import PPORunner

MEAL = pytest.mark.skipif(
    unavailable("meal/overcooked-v0") is not None, reason=str(unavailable("meal/overcooked-v0"))
)


def test_cnn_torso_flattens_pixel_features() -> None:
    torso = CNNTorso(hidden_size=12)
    obs = jnp.zeros((2, 10, 10, 4), dtype=jnp.bool_)
    params = torso.init(jax.random.key(0), obs)
    out = cast("jax.Array", torso.apply(params, obs))
    assert out.shape == (2, 12)
    assert bool(jnp.all(jnp.isfinite(out)))


def _features(torso: CNNTorso, params: Any, obs: jax.Array) -> np.ndarray:
    return np.asarray(cast("jax.Array", torso.apply(params, obs)))


def test_cnn_torso_divides_by_its_fixed_scale_whatever_the_frame_holds() -> None:
    plain, pixels = CNNTorso(hidden_size=12), CNNTorso(hidden_size=12, pixel_scale=255.0)
    key = jax.random.key(0)
    raw = jax.random.randint(key, (2, 84, 84, 4), 0, 256).astype(jnp.float32)
    params = plain.init(key, raw)
    np.testing.assert_array_equal(
        _features(pixels, params, raw), _features(plain, params, raw / 255.0)
    )
    # Without a scale nothing is divided: not whole numbers in 0-255, binary
    # channels, normalised floats or grids with -1 cells.
    binary = jax.random.bernoulli(key, 0.3, (2, 10, 10, 4)).astype(jnp.float32)
    normalised = jax.random.normal(key, (2, 10, 10, 4)) * 30.0
    grid = jax.random.randint(key, (2, 10, 10, 4), -1, 9).astype(jnp.float32)
    for frame in (raw, binary, normalised, grid):
        small = plain.init(key, frame)
        unscaled = _features(plain, small, frame)
        assert not np.allclose(unscaled, _features(plain, small, frame / 255.0), atol=1e-3)
        np.testing.assert_allclose(_features(pixels, small, frame * 255.0), unscaled, atol=1e-5)


def test_one_cell_going_from_one_to_two_does_not_rescale_the_frame() -> None:
    torso = CNNTorso(hidden_size=12)
    key = jax.random.key(2)
    frame = jax.random.bernoulli(key, 0.2, (1, 9, 9, 26)).astype(jnp.float32).at[0, 4, 4, 16].set(1)
    params = torso.init(key, frame)
    features = _features(torso, params, frame)
    counted = _features(torso, params, frame.at[0, 4, 4, 16].set(2.0))
    rescaled = _features(torso, params, frame / 255.0)
    change = float(np.linalg.norm(counted - features))
    assert 0.0 < change < 0.2 * float(np.linalg.norm(rescaled - features))
    assert change < 0.2 * float(np.linalg.norm(features))


def _meal_learners() -> list[tuple[str, Callable[[jax.Array], jax.Array]]]:
    env = pp.make("meal/overcooked-v0")
    _, obs, _ = env.init(jax.random.key(0))
    space, key = env.observation_space, jax.random.key(0)
    ppo = PPO(action_size=6, config=PPOConfig(torso="cnn", hidden_size=16), observation_space=space)
    dqn = DQN(action_size=6, config=DQNConfig(torso="cnn", hidden_size=16), observation_space=space)
    assert ppo.pixel_scale == dqn.pixel_scale == 1.0
    ppo_state, dqn_state = ppo.init(obs, key), dqn.init(obs, key)
    return [
        ("ppo", lambda frame: jnp.concatenate(ppo.evaluate(ppo_state, frame)[0])),
        ("dqn", lambda frame: jnp.concatenate(dqn.q_values(dqn_state, frame))),
    ]


@MEAL
def test_meal_keeps_one_input_scale_when_a_pot_gets_its_second_onion() -> None:
    # A real MEAL state stepped once with `interact`; see the helper's docstring.
    frames: tuple[jax.Array, jax.Array] = load_script("record_baselines").meal_pot_fill()
    before, after = frames
    assert float(before.max()) == 1.0
    assert float(after.max()) == 2.0
    assert int(jnp.sum(before != after)) == 4  # the pot's count and the onion in hand, per cook
    for name, outputs in _meal_learners():
        step = float(np.linalg.norm(np.asarray(outputs(after) - outputs(before))))
        rescale = float(np.linalg.norm(np.asarray(outputs(before / 255.0) - outputs(before))))
        assert 0.0 < step < 0.5 * rescale, name


@pytest.mark.parametrize(
    "make",
    [
        lambda: pp.make("gymnax/Breakout-MinAtar"),
        lambda: pp.make("gymnax/Asterix-MinAtar"),
        lambda: pp.make("gymnax/Freeway-MinAtar"),
        lambda: pp.make("gymnax/SpaceInvaders-MinAtar"),
        lambda: pp.make("gymnax/Pong-misc"),
        pytest.param(lambda: pp.make("meal/overcooked-v0"), marks=MEAL),
        pytest.param(lambda: pp.presets.meal_sequence(sequence_length=2, period=2), marks=MEAL),
        lambda: pp.presets.switching_minatar(period=2),
    ],
    ids=[
        "breakout",
        "asterix",
        "freeway",
        "space-invaders",
        "pong",
        "meal",
        "meal-sequence",
        "switching-minatar",
    ],
)
def test_image_environments_that_are_not_8_bit_pixels_derive_no_scale(
    make: Callable[[], Any],
) -> None:
    space = make().observation_space
    assert len(space.shape) == 4
    assert derive_pixel_scale(space) == 1.0
    ppo, dqn = PPOConfig(torso="cnn"), DQNConfig(torso="cnn")
    assert PPO(action_size=3, config=ppo, observation_space=space).pixel_scale == 1.0
    assert IPPO(action_size=3, config=ppo, observation_space=space).pixel_scale == 1.0
    assert DQN(action_size=3, config=dqn, observation_space=space).pixel_scale == 1.0
    assert IQL(action_size=3, config=dqn, observation_space=space).pixel_scale == 1.0


def test_the_scale_is_derived_from_the_bounds_of_the_space() -> None:
    box = gym.spaces.Box
    # As declared by: a Gymnasium pixel env, EnvPool Atari after the float32
    # lift with a slot axis, and AgarCL's screen.
    assert derive_pixel_scale(box(0, 255, (84, 84, 4), dtype=np.uint8)) == 255.0
    lifted = np.broadcast_to(np.float32(255), (2, 84, 84, 4))
    assert derive_pixel_scale(box(np.zeros_like(lifted), lifted, dtype=np.float32)) == 255.0
    assert derive_pixel_scale(box(0, 255, (1, 84, 84, 3), dtype=np.float32)) == 255.0
    # AgarCL's grid, COOM's normalised frames, binary uint8 channels, mixed bounds.
    assert derive_pixel_scale(box(-1, np.inf, (1, 16, 16, 4), dtype=np.float32)) == 1.0
    assert derive_pixel_scale(box(-np.inf, np.inf, (1, 84, 84, 12), dtype=np.float32)) == 1.0
    assert derive_pixel_scale(box(0, 1, (10, 10, 4), dtype=np.uint8)) == 1.0
    mixed = np.full((4, 4, 2), 255, dtype=np.float32)
    mixed[0, 0, 0] = 1
    assert derive_pixel_scale(box(np.zeros_like(mixed), mixed, dtype=np.float32)) == 1.0
    assert derive_pixel_scale(gym.spaces.MultiDiscrete([3, 3])) == 1.0


@pytest.mark.skipif(sys.platform != "linux", reason="EnvPool Atari is Linux-only.")
def test_envpool_atari_declares_8_bit_pixels() -> None:
    env = pp.make("envpool/Atari-v0", num_envs=1, num_threads=1)
    try:
        assert derive_pixel_scale(env.observation_space) == 255.0
    finally:
        cast(Any, env).close()


def test_a_configured_pixel_scale_wins_over_the_space_and_is_validated() -> None:
    pixels = gym.spaces.Box(0, 255, (1, 12, 12, 2), dtype=np.float32)
    raw = jax.random.randint(jax.random.key(3), (1, 12, 12, 2), 0, 256).astype(jnp.float32)
    learners: list[tuple[Any, Any]] = [(PPO, PPOConfig), (DQN, DQNConfig)]
    for learner, config in learners:
        derived = learner(action_size=3, config=config(torso="cnn"), observation_space=pixels)
        forced = learner(action_size=3, config=config(torso="cnn", pixel_scale=255.0))
        off = learner(
            action_size=3, config=config(torso="cnn", pixel_scale=1.0), observation_space=pixels
        )
        unset = learner(action_size=3, config=config(torso="cnn"))
        assert (derived.pixel_scale, forced.pixel_scale) == (255.0, 255.0)
        assert (off.pixel_scale, unset.pixel_scale) == (1.0, 1.0)
        state = unset.init(raw, jax.random.key(0))
        expected = jax.tree.leaves(unset.network.apply(state.params, raw / 255.0))
        for agent in (derived, forced):
            actual = jax.tree.leaves(agent.network.apply(state.params, raw))
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert derived.configuration["pixel_scale"] == 255.0
        assert "pixel_scale" not in learner(action_size=3).configuration
        for bad in (0, -1.0, float("inf"), float("nan"), True, "255"):
            with pytest.raises(ValueError, match="pixel_scale must be None or a finite positive"):
                config(pixel_scale=bad)


class Frames(gym.Env[np.ndarray, np.int64]):
    """Raw 8-bit frames whose largest value is 1 at reset and 200 afterwards."""

    observation_space = gym.spaces.Box(0, 255, (12, 12, 2), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.ones((12, 12, 2), dtype=np.uint8), {}

    def step(self, action: np.int64) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.full((12, 12, 2), 200, dtype=np.uint8), 1.0, False, False, {}


def test_a_runner_refuses_a_cnn_learner_whose_scale_is_not_the_environments() -> None:
    from perpetua.adapters import GymnasiumLifetime

    def config(**options: Any) -> PPOConfig:
        return PPOConfig(
            torso="cnn", rollout_steps=2, epochs=1, minibatches=1, hidden_size=8, **options
        )

    def start(build: Callable[[Any], PPO]) -> tuple[HostPPORunner[Any], Any, Any]:
        env = GymnasiumLifetime(Frames())
        try:
            runner = HostPPORunner(env, build(env))
            state = runner.init(seed=0)
            return runner, state, runner(state)
        finally:
            env.close()

    with pytest.raises(ValueError, match=r"divides observations by 1, but.*pixel scale of 255"):
        start(lambda env: PPO(action_size=2, config=config()))
    runner, state, trained = start(
        lambda env: PPO(action_size=2, config=config(), observation_space=env.observation_space)
    )
    assert cast(Any, runner.metadata["learner"])["pixel_scale"] == 255.0
    assert bool(trained.healthy)
    # The reset frame is all ones, which the old per-frame rule left unscaled.
    assert float(state.lifetime.obs.max()) == 1.0
    reference = PPO(action_size=2, config=config())
    for a, b in zip(
        runner.agent.evaluate(state.agent, state.lifetime.obs),
        reference.evaluate(state.agent, state.lifetime.obs / 255.0),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # A configured scale is the user's choice and is not checked.
    _, _, unscaled = start(lambda env: PPO(action_size=2, config=config(pixel_scale=1.0)))
    assert bool(unscaled.healthy)
    minatar = pp.make("gymnax/Breakout-MinAtar")
    pixels = gym.spaces.Box(0, 255, (1, 10, 10, 4), dtype=np.float32)
    dqn = DQN(action_size=3, config=DQNConfig(torso="cnn"), observation_space=pixels)
    with pytest.raises(ValueError, match=r"divides observations by 255, but.*pixel scale of 1 "):
        DQNRunner(minatar, dqn).init(seed=0)


def test_cnn_torso_scaling_does_not_depend_on_the_batch() -> None:
    torso = CNNTorso(hidden_size=12)
    key = jax.random.key(1)
    raw = jax.random.randint(key, (1, 84, 84, 4), 0, 256).astype(jnp.float32)
    binary = jax.random.bernoulli(key, 0.3, (1, 84, 84, 4)).astype(jnp.float32)
    params = torso.init(key, raw)
    together = _features(torso, params, jnp.concatenate([raw, binary]))
    # A GPU picks a different convolution algorithm per batch size (3.4e-4 seen on an H200).
    atol = 1e-5 if jax.default_backend() == "cpu" else 1e-3
    np.testing.assert_allclose(together[:1], _features(torso, params, raw), atol=atol)
    np.testing.assert_allclose(together[1:], _features(torso, params, binary), atol=atol)


def test_ppo_cnn_trains_on_minatar() -> None:
    env = pp.make("gymnax/Breakout-MinAtar")
    config = PPOConfig(torso="cnn", rollout_steps=4, epochs=1, minibatches=1, hidden_size=16)
    trainer = PPORunner(env, PPO(action_size=3, config=config))
    assert cast(Any, trainer.metadata["learner"])["pixel_scale"] == 1.0
    state = trainer.init(seed=0)
    assert state.lifetime.obs.shape == (1, 10, 10, 4)
    for _ in range(2):
        state = trainer(state)
    assert state.agent.gradient_steps.to_int() == 2
    assert bool(state.healthy)
    assert bool(jnp.all(jnp.isfinite(state.statistics.loss)))


def test_dqn_cnn_trains_on_minatar() -> None:
    env = pp.make("gymnax/Breakout-MinAtar")
    config = DQNConfig(
        torso="cnn",
        rollout_steps=4,
        buffer_capacity=32,
        batch_size=8,
        gradient_steps=1,
        hidden_size=16,
    )
    trainer = DQNRunner(env, DQN(action_size=3, config=config))
    state = trainer.init(seed=0)
    for _ in range(2):
        state = trainer(state)
    assert state.agent.gradient_steps.to_int() == 2
    assert state.replay.obs.shape == (32, 1, 10, 10, 4)
    assert bool(jnp.all(jnp.isfinite(state.statistics.loss)))


def test_cnn_checkpoint_round_trip(tmp_path: Path) -> None:
    env = pp.make("gymnax/Breakout-MinAtar")
    config = PPOConfig(torso="cnn", rollout_steps=4, epochs=1, minibatches=1, hidden_size=16)
    trainer = PPORunner(env, PPO(action_size=3, config=config))
    state = trainer(trainer.init(seed=0))
    path = tmp_path / "cnn"
    pp.checkpoint.save(path, state, metadata=trainer.metadata)
    restored = pp.checkpoint.restore(
        path, trainer.init(seed=99), expected_metadata=trainer.metadata
    )
    expected, actual = trainer(state), trainer(restored)
    for a, b in zip(
        jax.tree.leaves(expected.agent.params), jax.tree.leaves(actual.agent.params), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_mlp_is_still_the_default() -> None:
    assert PPOConfig().torso == "mlp"
    assert DQNConfig().torso == "mlp"
    agent = PPO(action_size=3, config=PPOConfig(hidden_size=8))
    state = agent.init(jnp.zeros((1, 4)), jax.random.key(0))
    assert state.params is not None


def test_the_pixel_scale_is_checked_when_a_runner_is_built_not_only_at_init() -> None:
    from perpetua.adapters import GymnasiumLifetime
    from perpetua.agents import ComputePPORunner, HostDQNRunner

    ppo = PPOConfig(torso="cnn", rollout_steps=2, epochs=1, minibatches=1, hidden_size=8)
    dqn = DQNConfig(torso="cnn", rollout_steps=2, buffer_capacity=4, batch_size=2, gradient_steps=1)
    refused = r"divides observations by 1, but.*pixel scale of 255"
    env = GymnasiumLifetime(Frames())
    try:
        for learner in (PPO, IPPO):
            with pytest.raises(ValueError, match=refused):
                HostPPORunner(env, learner(action_size=2, config=ppo))
            with pytest.raises(ValueError, match=refused):
                ComputePPORunner(env, learner(action_size=2, config=ppo))
        for learner in (DQN, IQL):
            with pytest.raises(ValueError, match=refused):
                HostDQNRunner(env, learner(action_size=2, config=dqn))
        # A carry initialised by a matching runner is no way around the check.
        matching = PPO(action_size=2, config=ppo, observation_space=env.observation_space)
        carry = HostPPORunner(env, matching).init(seed=0)
        assert carry.lifetime.obs.shape == (1, 12, 12, 2)
        with pytest.raises(ValueError, match=refused):
            HostPPORunner(env, PPO(action_size=2, config=ppo))
        HostDQNRunner(env, DQN(action_size=2, config=dqn, observation_space=env.observation_space))
    finally:
        env.close()
    minatar = pp.make("gymnax/Breakout-MinAtar")
    pixels = gym.spaces.Box(0, 255, (1, 10, 10, 4), dtype=np.float32)
    swapped = r"divides observations by 255, but.*pixel scale of 1 "
    for learner in (PPO, IPPO):
        with pytest.raises(ValueError, match=swapped):
            PPORunner(minatar, learner(action_size=3, config=ppo, observation_space=pixels))
    for learner in (DQN, IQL):
        with pytest.raises(ValueError, match=swapped):
            DQNRunner(minatar, learner(action_size=3, config=dqn, observation_space=pixels))
    DQNRunner(minatar, DQN(action_size=3, config=dqn, observation_space=minatar.observation_space))
