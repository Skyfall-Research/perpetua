"""Switching presets: MinAtar natively, Atari through EnvPool on Linux."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

import sys
from importlib.util import find_spec
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp


def test_switching_minatar_switches_tasks_and_masks() -> None:
    env = pp.presets.switching_minatar(period=4)
    state, obs, info = env.init(jax.random.key(0))
    assert obs.shape == (1, 10, 10, 4)
    events = []
    for tick in range(8):
        state, obs, _, info = env.step(state, jnp.zeros(1, dtype=jnp.int32), jax.random.key(tick))
        if bool(info["space_change"][0]):
            events.append(tick)
    assert events == [3, 7]
    assert int(state.drift["task"].value) == 0
    assert info["action_mask"].shape == (1, 5)


def test_switching_atari_requires_two_games() -> None:
    with pytest.raises(ValueError, match=r"switching_atari needs at least two games\."):
        pp.presets.switching_atari(["Breakout-v5"])
    with pytest.raises(ValueError, match=r"switching_minatar needs at least two games\."):
        pp.presets.switching_minatar(["Breakout-MinAtar"])


@pytest.mark.skipif(
    sys.platform != "linux" or find_spec("envpool") is None,
    reason="EnvPool Atari is Linux-only and optional.",
)
def test_switching_atari_switches_games_on_linux() -> None:
    env = pp.presets.switching_atari(
        ["Breakout-v5", "Asterix-v5"], num_envs=2, num_threads=1, period=4
    )
    try:
        state, obs, info = env.init(jax.random.key(0))
        assert obs.shape == (2, 84, 84, 4)
        assert info["action_mask"].shape == (2, 18)
        for tick in range(8):
            state, obs, reward, info = env.step(
                state, jnp.zeros(2, dtype=jnp.int32), jax.random.key(tick)
            )
            assert reward.shape == (2,)
        assert int(state.drift["task"].value) == 0
    finally:
        env.close()


def test_host_task_switch_with_gymnasium() -> None:
    import gymnasium as gym

    from perpetua.adapters import GymnasiumLifetime

    first = GymnasiumLifetime(gym.make("CartPole-v1"))
    second = GymnasiumLifetime(gym.make("CartPole-v1"))
    env = pp.tasks.switch(cast(Any, [first, second]), pp.drift.Cyclic([0.0, 1.0], 2))
    try:
        state, obs, info = env.init(jax.random.key(0))
        assert obs.shape == (1, 4)
        assert info["action_mask"].shape == (1, 2)
        for tick in range(6):
            state, obs, reward, info = env.step(
                state, jnp.zeros(1, dtype=jnp.int32), jax.random.key(tick)
            )
            assert reward.shape == (1,)
        # After six ticks with period 2 the active task is 1.
        assert int(state.drift["task"].value) == 1
    finally:
        env.close()


def test_host_task_switch_refuses_restart() -> None:
    import gymnasium as gym

    from perpetua.adapters import GymnasiumLifetime

    envs = [GymnasiumLifetime(gym.make("CartPole-v1")) for _ in range(2)]
    try:
        with pytest.raises(ValueError, match="resume' only"):
            pp.tasks.switch(cast(Any, envs), pp.drift.Cyclic([0.0, 1.0], 2), on_switch="restart")
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("bad", [{"period": 0}, {"on_switch": "restart"}])
def test_a_rejected_host_preset_closes_the_simulators_it_built(
    monkeypatch: pytest.MonkeyPatch, bad: dict[str, Any]
) -> None:
    import gymnasium as gym

    from perpetua.adapters import host

    built: list[Any] = []

    def fake_atari(game: str, **kwargs: Any) -> Any:
        del game, kwargs
        env = host.GymnasiumLifetime(gym.make("CartPole-v1"))
        built.append(env)
        return env

    monkeypatch.setattr(host, "EnvPoolAtari", fake_atari)
    with pytest.raises(ValueError, match="period|on_switch"):
        pp.presets.switching_atari(["Pong-v5", "Breakout-v5"], **bad)
    # A bad period is refused before anything is built; a bad on_switch after.
    assert len(built) == (0 if "period" in bad else 2)
    assert all(env.env._closed for env in built)


def test_a_numpy_period_is_stored_as_a_json_int() -> None:
    import json

    env = pp.presets.switching_minatar(period=cast(Any, np.int64(1000)))
    preset = env.metadata["configuration"]["preset"]
    assert type(preset["horizon_ticks"]) is int
    json.dumps(env.metadata["configuration"])
