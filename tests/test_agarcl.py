"""AgarCL screen integration through its pinned native runtime."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.adapters import agarcl
from perpetua.adapters.snapshots import ReplaySnapshot

ROOT = Path(__file__).resolve().parents[1]

needs_agarcl = pytest.mark.skipif(
    not agarcl.available(),
    reason="AgarCL runtime not installed (python -m perpetua.adapters.agarcl_setup).",
)


def test_missing_runtime_explains_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("PERPETUA_AGARCL_PYTHON", str(tmp_path / "missing" / "python"))
    with pytest.raises(RuntimeError, match="python -m perpetua.adapters.agarcl_setup"):
        pp.make("agarcl/agario-screen-v0")


def test_the_setup_command_is_a_module_of_the_installed_package() -> None:
    # The checkout's script is a wrapper around the module the error names.
    for command in (
        ["-m", "perpetua.adapters.agarcl_setup"],
        [str(ROOT / "scripts" / "setup_agarcl.py")],
    ):
        result = subprocess.run(
            [sys.executable, *command, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "--root" in result.stdout


def test_the_cmake_wrapper_adds_the_private_opengl_paths_except_when_building(
    tmp_path: Path,
) -> None:
    from perpetua.adapters import agarcl_setup

    venv, prefix = tmp_path / "venv", tmp_path / "prefix"
    (venv / "bin").mkdir(parents=True)
    # Stands in for CMake: prints its arguments, one per line.
    (venv / "bin" / "cmake").write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    (venv / "bin" / "cmake").chmod(0o755)
    agarcl_setup.cmake_wrapper(venv, prefix)
    agarcl_setup.cmake_wrapper(venv, prefix)  # a rerun keeps the real binary

    def cmake(*arguments: str) -> list[str]:
        command = [str(venv / "bin" / "cmake"), *arguments]
        return subprocess.run(
            command, capture_output=True, text=True, check=True
        ).stdout.splitlines()

    library = prefix / "usr" / "lib" / agarcl_setup.MULTIARCH
    configured = cmake("-S", "source dir")
    assert configured[-2:] == ["-S", "source dir"]
    assert f"-DOPENGL_egl_LIBRARY={library / 'libEGL.so'}" in configured
    assert f"-DOPENGL_INCLUDE_DIR={prefix / 'usr' / 'include'}" in configured
    assert cmake("--build", "dir") == ["--build", "dir"]


def test_a_failed_restore_does_not_keep_the_old_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWorker:
        timeout = 1.0
        fail_replay = False

        def close(self) -> None:
            pass

        def request_array(
            self, _data: bytes | None = None, /, **payload: Any
        ) -> tuple[dict[str, Any], np.ndarray]:
            if payload["command"] == "replay":
                if FakeWorker.fail_replay:
                    raise RuntimeError("replay failed")
                return {"ticks": len(_data or b"") // 12}, np.zeros(0, np.float32)
            return {"reward": 0.0, "respawn": False}, np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(agarcl, "_Worker", FakeWorker)
    backend = agarcl._AgarCLBackend(  # pyright: ignore[reportPrivateUsage]
        obs_type="screen",
        screen_len=4,
        agent_view=True,
        grid_size=4,
        num_bots=0,
        mode=0,
        respawn_after=None,
        arena_size=None,
        num_pellets=None,
        num_viruses=None,
        flatten=False,
        replay_capacity=8,
    )
    action = np.asarray([[0.0, 0.0, 0.0]], np.float32)
    backend.init(0)
    backend.step(action, 0)
    current = cast(ReplaySnapshot, backend.snapshot())
    branch = current._replace(actions=current.actions.copy(), length=np.asarray(0, np.int64))
    backend.step(action, 0)
    later = cast(ReplaySnapshot, backend.snapshot())

    FakeWorker.fail_replay = True
    with pytest.raises(RuntimeError, match="replay failed"):
        backend.restore(later._replace(actions=later.actions.copy()))
    with pytest.raises(RuntimeError, match="no valid simulator state"):
        backend.step(action, 0)
    # Restoring the state the backend claimed before the failure must replay, not shortcut.
    with pytest.raises(RuntimeError, match="replay failed"):
        backend.restore(later)

    FakeWorker.fail_replay = False
    backend.restore(branch)
    backend.step(action, 0)
    assert backend.length == 1


@needs_agarcl
def test_screen_and_hybrid_actions() -> None:
    env = pp.make("agarcl/agario-screen-v0")
    try:
        state, obs, info = env.init(jax.random.key(0))
        assert obs.shape == (1, 84, 84, 3)
        assert env.action_space.shape == (1, 3)
        for tick, discrete in enumerate([0.0, 1.0, 2.0]):
            action = jnp.asarray([[0.25, -0.5, discrete]], dtype=jnp.float32)
            state, obs, reward, info = env.step(state, action, jax.random.key(tick))
            assert obs.shape == (1, 84, 84, 3)
            assert reward.shape == (1,)
        assert info["t"].to_int() == 3
    finally:
        cast(Any, env).close()


@needs_agarcl
def test_seeded_lifetimes_reproduce() -> None:
    first, second = pp.make("agarcl/agario-screen-v0"), pp.make("agarcl/agario-screen-v0")
    try:
        a, _, _ = first.init(jax.random.key(4))
        b, _, _ = second.init(jax.random.key(4))
        for tick in range(12):
            action = jnp.asarray([[0.1, 0.2, tick % 3]], dtype=jnp.float32)
            a, obs_a, reward_a, _ = first.step(a, action, jax.random.key(tick))
            b, obs_b, reward_b, _ = second.step(b, action, jax.random.key(tick))
            np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))
            np.testing.assert_array_equal(np.asarray(reward_a), np.asarray(reward_b))
    finally:
        cast(Any, first).close()
        cast(Any, second).close()


@needs_agarcl
def test_flat_screen_trains_through_host_sac() -> None:
    from perpetua.agents import SAC, HostSACRunner, SACConfig

    env = pp.make("agarcl/agario-screen-v0", flatten=True)
    try:
        space = cast(Any, env.action_space)
        config = SACConfig(
            rollout_steps=4, buffer_capacity=16, batch_size=4, gradient_steps=1, hidden_size=8
        )
        agent = SAC(action_low=space.low[0], action_high=space.high[0], config=config)
        trainer = HostSACRunner(env, agent)
        state = trainer(trainer.init(seed=0))
        assert bool(state.healthy)
        assert state.agent.gradient_steps.to_int() == 1
    finally:
        cast(Any, env).close()


@needs_agarcl
def test_screen_shows_the_world_and_depends_on_seed() -> None:
    # A second simulator construction in one worker once rendered all-black
    # frames; shapes and reproducibility alone cannot catch that.
    first, second = pp.make("agarcl/agario-screen-v0"), pp.make("agarcl/agario-screen-v0")
    try:
        _, a, _ = first.init(jax.random.key(0))
        _, b, _ = second.init(jax.random.key(1))
        assert float(jnp.max(a)) > 0.0
        assert float(jnp.std(a)) > 0.0
        assert not np.array_equal(np.asarray(a), np.asarray(b))
    finally:
        cast(Any, first).close()
        cast(Any, second).close()


@needs_agarcl
def test_grid_observes_pellets_and_agent() -> None:
    env = pp.make("agarcl/agario-grid-v0", grid_size=32)
    try:
        state, obs, _ = env.init(jax.random.key(0))
        assert obs.shape == (1, 32, 32, 8)
        for tick in range(5):
            action = jnp.asarray([[0.3, -0.2, 0.0]], dtype=jnp.float32)
            state, obs, _, _ = env.step(state, action, jax.random.key(tick))
        channels = np.asarray(obs[0]).reshape(-1, 8).astype(bool).sum(axis=0)
        assert channels[1] > 0
        assert channels[5] > 0
    finally:
        cast(Any, env).close()


@needs_agarcl
def test_replay_checkpoints_branch_exactly() -> None:
    env = pp.make("agarcl/agario-grid-v0", grid_size=32, replay_capacity=32)
    rng = np.random.default_rng(0)
    actions = [
        jnp.asarray([[*rng.uniform(-1, 1, 2), tick % 3]], dtype=jnp.float32) for tick in range(12)
    ]
    try:
        state, _, _ = env.init(jax.random.key(0))
        states: list[Any] = [state]
        trace: list[tuple[Any, Any]] = []
        for tick, action in enumerate(actions):
            state, obs, reward, _ = env.step(state, action, jax.random.key(tick))
            states.append(state)
            trace.append((np.asarray(obs), np.asarray(reward)))
        # Branch from an older state (a replay in a fresh worker), then return to
        # a later state of the original line: both must reproduce it exactly.
        branch = states[5]
        for tick in range(5, 9):
            branch, obs, reward, _ = env.step(branch, actions[tick], jax.random.key(tick))
            np.testing.assert_array_equal(np.asarray(obs), trace[tick][0])
            np.testing.assert_array_equal(np.asarray(reward), trace[tick][1])
        _, obs, reward, _ = env.step(states[11], actions[11], jax.random.key(11))
        np.testing.assert_array_equal(np.asarray(obs), trace[11][0])
        np.testing.assert_array_equal(np.asarray(reward), trace[11][1])
    finally:
        cast(Any, env).close()


@needs_agarcl
def test_replay_restores_across_respawns_with_bots() -> None:
    env = pp.make(
        "agarcl/agario-grid-v0",
        grid_size=16,
        num_bots=8,
        respawn_after=5,
        replay_capacity=32,
    )
    try:
        state, _, _ = env.init(jax.random.key(0))
        actions = [
            jnp.asarray([[0.1 * tick, -0.05 * tick, tick % 3]], dtype=jnp.float32)
            for tick in range(12)
        ]
        states: list[Any] = [state]
        trace: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], bool]] = []
        for tick, action in enumerate(actions):
            state, obs, reward, info = env.step(state, action, jax.random.key(tick))
            states.append(state)
            trace.append((np.asarray(obs), np.asarray(reward), bool(info["respawn"][0])))
        assert [entry[2] for entry in trace] == [False] * 4 + [True] + [False] * 4 + [True] + [
            False
        ] * 2
        # Replaying the log from the start crosses both respawns in a fresh worker.
        branch = states[0]
        for tick, action in enumerate(actions):
            branch, obs, reward, info = env.step(branch, action, jax.random.key(tick))
            np.testing.assert_array_equal(np.asarray(obs), trace[tick][0])
            np.testing.assert_array_equal(np.asarray(reward), trace[tick][1])
            assert bool(info["respawn"][0]) == trace[tick][2]
    finally:
        cast(Any, env).close()


@needs_agarcl
def test_million_tick_capacity_allocates_12mb_and_replays_short_log(tmp_path: Any) -> None:
    env = pp.make("agarcl/agario-grid-v0", grid_size=16, replay_capacity=1_000_000)
    try:
        state, _, _ = env.init(jax.random.key(0))
        actions = [jnp.asarray([[0.1, -0.2, tick % 3]], dtype=jnp.float32) for tick in range(20)]
        for tick, action in enumerate(actions):
            state, *_ = env.step(state, action, jax.random.key(tick))
        assert state.env.snapshot.actions.shape == (1_000_000, 3)
        assert state.env.snapshot.actions.nbytes == 12_000_000
        path = tmp_path / "million"
        pp.checkpoint.save(path, state, metadata=env.metadata)
        _, obs, reward, _ = env.step(state, actions[0], jax.random.key(20))
        restored = pp.checkpoint.restore(path, state, expected_metadata=env.metadata)
        assert int(restored.env.snapshot.length) == 20
        restored, restored_obs, restored_reward, _ = env.step(
            restored, actions[0], jax.random.key(20)
        )
        np.testing.assert_array_equal(np.asarray(restored_obs), np.asarray(obs))
        np.testing.assert_array_equal(np.asarray(restored_reward), np.asarray(reward))
    finally:
        cast(Any, env).close()


@needs_agarcl
def test_replay_log_overflow_is_refused() -> None:
    env = pp.make("agarcl/agario-grid-v0", grid_size=16, replay_capacity=2)
    try:
        state, _, _ = env.init(jax.random.key(0))
        action = jnp.zeros((1, 3), dtype=jnp.float32)
        for tick in range(2):
            state, *_ = env.step(state, action, jax.random.key(tick))
        with pytest.raises(RuntimeError, match="replay log is full"):
            env.step(state, action, jax.random.key(2))
    finally:
        cast(Any, env).close()


@needs_agarcl
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="The worker soak reads /proc.")
def test_worker_soak_replays_exactly_and_leaks_no_descriptors_or_zombies() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "soak.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "worker",
            "--worker-ticks",
            "400",
            "--worker-sample-every",
            "50",
            "--churn",
            "10",
        ],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    summary = json.loads(result.stdout.splitlines()[-1])
    assert result.returncode == 0, summary
    assert all(summary["checks"].values())


@needs_agarcl
def test_grand_arena_uses_the_papers_full_game_world_and_accepts_overrides() -> None:
    env = pp.make("agarcl/grand-arena-v0")
    try:
        configuration = cast(dict[str, Any], env.metadata.get("configuration"))["backend"]
        assert env.metadata["env_id"] == "agarcl/grand-arena-v0"
        assert env.observation_space.shape == (1, 128, 128, 3)
        assert {key: configuration[key] for key in ("num_pellets", "num_viruses", "num_bots")} == {
            "num_pellets": 500,
            "num_viruses": 10,
            "num_bots": 8,
        }
        state, _, _ = env.init(jax.random.key(0))
        state, obs, reward, _ = env.step(state, jnp.zeros((1, 3), jnp.float32), jax.random.key(1))
        assert obs.shape == (1, 128, 128, 3)
        assert reward.shape == (1,)
    finally:
        cast(Any, env).close()
    small = pp.make("agarcl/grand-arena-v0", screen_len=32, num_bots=2)
    try:
        assert small.observation_space.shape == (1, 32, 32, 3)
        assert cast(dict[str, Any], small.metadata.get("configuration"))["backend"]["num_bots"] == 2
    finally:
        cast(Any, small).close()


def test_a_failed_reinit_does_not_close_the_old_simulator_twice(tmp_path: Any) -> None:
    from perpetua.adapters.worker import WorkerProcess

    (tmp_path / "gym_agario").mkdir()
    (tmp_path / "gym_agario" / "__init__.py").write_text("")
    (tmp_path / "gym_agario" / "AgarioEnv.py").write_text(
        """
class AgarioEnv:
    @staticmethod
    def _get_env_args(namespace, world):
        return (1, 1, *range(8))
"""
    )
    (tmp_path / "agarcl.py").write_text(
        """
import numpy as np

built = 0


class GridEnvironment:
    closed = False

    def __init__(self, *args):
        global built
        built += 1
        if built == 2:
            raise RuntimeError("build failed")

    def configure_observation(self, config):
        pass

    def observation_shape(self):
        return (2, 4, 4)

    def get_state(self):
        return [np.zeros((2, 4, 4), np.int32)]

    def seed(self, seed):
        pass

    def reset(self):
        pass

    def close(self):
        if self.closed:
            raise RuntimeError("closed twice")
        self.closed = True
"""
    )
    script = Path(agarcl.__file__).with_name("agarcl_worker.py")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    process = WorkerProcess("AgarCL", [sys.executable, "-P", str(script)], env=env, dtype=np.uint8)
    try:
        process.request_array(command="init", obs_type="grid", seed=0)
        with pytest.raises(RuntimeError, match="build failed"):
            process.request(command="init", obs_type="grid", seed=0)
        process.request_array(command="init", obs_type="grid", seed=0)
    finally:
        process.close()
    assert process.process.returncode == 0
