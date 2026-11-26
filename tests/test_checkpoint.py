"""A checkpoint must preserve the complete lifetime, not just its seed."""

# JAX transformation annotations are partial.
# pyright: reportUnknownMemberType=false

import importlib.metadata
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_same

import perpetua as pp
from perpetua.params import DriftEnvState


def test_native_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.RandomWalk(low=0, high=4, scale=0.2))
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=8)
    state = runner(pp.runtime.init_lifetime(env, seed=1, window_size=4))
    path = tmp_path / "lifetime"
    pp.checkpoint.save(path, state, metadata=runner.metadata)
    template = pp.runtime.init_lifetime(env, seed=999, window_size=4)
    restored = pp.checkpoint.restore(path, template, expected_metadata=runner.metadata)
    assert_same(state, restored)
    assert_same(runner(state), runner(restored))


def test_existing_checkpoint_is_not_overwritten(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0")
    state = pp.runtime.init_lifetime(env)
    path = tmp_path / "lifetime"
    pp.checkpoint.save(path, state, metadata=env.metadata)
    with pytest.raises(FileExistsError):
        pp.checkpoint.save(path, state, metadata=env.metadata)
    restored = pp.checkpoint.restore(path, state, expected_metadata=env.metadata)
    assert_same(state, restored)


def test_drift_configuration_mismatch_is_rejected(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(4, 4), (0, 0)], 2))
    other = pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(4, 4), (0, 0)], 3))
    state = pp.runtime.init_lifetime(env)
    path = tmp_path / "lifetime"
    pp.checkpoint.save(path, state, metadata=env.metadata)
    with pytest.raises(ValueError, match="configuration") as excinfo:
        pp.checkpoint.restore(path, state, expected_metadata=other.metadata)
    message = str(excinfo.value)
    assert "metadata.configuration.drift.period: 2 in the checkpoint, 3 expected" in message


def test_every_configuration_mismatch_names_the_key_and_both_values(tmp_path: Path) -> None:
    saved = {"environment": {"size": 4, "drifts": [{"period": 2}, {"period": 5}]}, "lr": 0.1}
    expected = {"environment": {"size": 5, "drifts": [{"period": 2}, {"period": 6}]}, "lr": 0.1}
    pp.checkpoint.save(tmp_path / "ck", {"x": jnp.arange(2)}, metadata=saved)
    with pytest.raises(ValueError, match="configuration does not match") as excinfo:
        pp.checkpoint.restore(tmp_path / "ck", {"x": jnp.arange(2)}, expected_metadata=expected)
    message = str(excinfo.value)
    assert "metadata.environment.size: 4 in the checkpoint, 5 expected" in message
    assert "metadata.environment.drifts[1].period: 5 in the checkpoint, 6" in message
    assert "lr" not in message


def test_nested_configuration_key_is_not_doubled_in_the_path(tmp_path: Path) -> None:
    saved = {"configuration": {"tasks": [{"configuration": {"size": 3}}]}}
    expected = {"configuration": {"tasks": [{"configuration": {"size": 4}}]}}
    pp.checkpoint.save(tmp_path / "ck", {"x": jnp.arange(2)}, metadata=saved)
    with pytest.raises(ValueError, match="configuration does not match") as excinfo:
        pp.checkpoint.restore(tmp_path / "ck", {"x": jnp.arange(2)}, expected_metadata=expected)
    message = str(excinfo.value)
    assert "metadata.configuration.tasks[0].configuration.size: 3 in the checkpoint" in message
    assert "configuration.configuration" not in message.replace("metadata.configuration", "")


def test_restore_rejects_incompatible_state_shapes(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0")
    state = pp.runtime.init_lifetime(env)
    path = tmp_path / "lifetime"
    pp.checkpoint.save(path, state, metadata=env.metadata)
    other = pp.make("ContinualGrid-v0", num_agents=2)
    template = pp.runtime.init_lifetime(other)
    with pytest.raises(ValueError, match=r"Checkpoint structure, shapes or dtypes do not") as e:
        pp.checkpoint.restore(path, template, expected_metadata=env.metadata)
    assert "leaf '.state.env.positions': int32[1, 2] in the checkpoint" in str(e.value)
    assert "int32[2, 2] in the template" in str(e.value)


def test_schema_mismatch_names_the_first_differing_leaf(tmp_path: Path) -> None:
    saved = {"a": jnp.zeros((2, 3)), "b": jnp.zeros(4, dtype=jnp.int32)}
    pp.checkpoint.save(tmp_path / "ck", saved, metadata={})
    shape = {"a": jnp.zeros((2, 4)), "b": jnp.zeros(4, dtype=jnp.int32)}
    with pytest.raises(ValueError, match=r"leaf \"\['a'\]\": float32\[2, 3\] in the checkpoint"):
        pp.checkpoint.restore(tmp_path / "ck", shape, expected_metadata={})
    dtype = {"a": jnp.zeros((2, 3)), "b": jnp.zeros(4)}
    with pytest.raises(ValueError, match=r"\['b'\]\": int32\[4\] in the checkpoint, float32"):
        pp.checkpoint.restore(tmp_path / "ck", dtype, expected_metadata={})
    renamed = {"a": jnp.zeros((2, 3)), "c": jnp.zeros(4, dtype=jnp.int32)}
    with pytest.raises(
        ValueError, match=r"leaf \"\['b'\]\" in the checkpoint, \"\['c'\]\" in the template"
    ):
        pp.checkpoint.restore(tmp_path / "ck", renamed, expected_metadata={})
    kind = ({"a": jnp.zeros((2, 3))}, {"b": jnp.zeros(4, dtype=jnp.int32)})
    with pytest.raises(ValueError, match="container"):
        pp.checkpoint.restore(tmp_path / "ck", kind, expected_metadata={})


def test_restoring_a_missing_path_says_it_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"No checkpoint at .*nowhere.*does not exist"):
        pp.checkpoint.restore(tmp_path / "nowhere", {"x": jnp.zeros(1)}, expected_metadata={})


def test_checkpoints_of_an_older_format_are_refused_with_that_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from perpetua import checkpoint as checkpoint_module

    state = {"x": jnp.arange(3)}
    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_module, "_VERSION", 1)
        pp.checkpoint.save(tmp_path / "old", state, metadata={})
    with pytest.raises(ValueError, match=r"format version 1 is older than the supported version"):
        pp.checkpoint.restore(tmp_path / "old", state, expected_metadata={})


_STATIC_NODE = """
import sys
import jax, jax.numpy as jnp
import perpetua as pp

@jax.tree_util.register_pytree_node_class
class Node:
    def __init__(self, weights, activation):
        self.weights, self.activation = weights, activation
    def tree_flatten(self):
        return (self.weights,), self.activation
    @classmethod
    def tree_unflatten(cls, activation, children):
        return cls(*children, activation)

def relu(x):
    return jnp.maximum(x, 0)

carry = {"node": Node(jnp.arange(4.0), relu), "count": jnp.int32(3)}
if sys.argv[1] == "save":
    pp.checkpoint.save(sys.argv[2], carry, metadata={"checkpointable": True})
else:
    template = {"node": Node(jnp.zeros(4), relu), "count": jnp.int32(0)}
    metadata = {"checkpointable": True}
    restored = pp.checkpoint.restore(sys.argv[2], template, expected_metadata=metadata)
    assert restored["node"].weights.tolist() == [0.0, 1.0, 2.0, 3.0], restored
    assert int(restored["count"]) == 3
    print("restored")
"""


def test_a_node_with_static_fields_restores_in_a_fresh_process(tmp_path: Path) -> None:
    import os

    process_env = {**os.environ, "JAX_PLATFORMS": "cpu"}

    def run(mode: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", _STATIC_NODE, mode, str(tmp_path / "ck")],
            env=process_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        return result.stdout

    run("save")
    assert run("restore").strip() == "restored"


def test_non_json_metadata_is_rejected_before_writing(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0")
    state = pp.runtime.init_lifetime(env)
    path = tmp_path / "lifetime"
    with pytest.raises(TypeError):
        pp.checkpoint.save(path, state, metadata={"invalid": object()})
    assert not path.exists()
    with pytest.raises(ValueError, match=r"Out of range float values are not JSON"):
        pp.checkpoint.save(path, state, metadata={"invalid": float("nan")})
    assert not path.exists()


def test_incomplete_checkpoint_is_not_restored(tmp_path: Path) -> None:
    path = tmp_path / "incomplete"
    path.mkdir()
    state = pp.runtime.init_lifetime(pp.make("ContinualGrid-v0"))
    with pytest.raises((ValueError, FileNotFoundError)):
        pp.checkpoint.restore(path, state, expected_metadata={})


def test_batched_lifetimes_and_typed_keys_roundtrip(tmp_path: Path) -> None:
    from perpetua.envs.continual_grid import GridState

    env = pp.make("ContinualGrid-v0", num_agents=2)

    def initialize(seed: jax.Array) -> pp.runtime.Lifetime[DriftEnvState[GridState]]:
        return pp.runtime.init_lifetime(env, seed=seed)

    state = jax.vmap(initialize)(jnp.asarray([1, 2]))
    path = tmp_path / "batch"
    pp.checkpoint.save(path, state, metadata=env.metadata)
    template = jax.vmap(initialize)(jnp.asarray([99, 100]))
    restored = pp.checkpoint.restore(path, template, expected_metadata=env.metadata)
    assert_same(state, restored)


def test_zero_size_state_leaves_round_trip(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0")
    metadata = pp.runtime.Runner(env, pp.policies.random, ticks=1).metadata
    state = {"values": jnp.arange(3.0), "empty": jnp.zeros((2, 0), dtype=jnp.int32)}
    pp.checkpoint.save(tmp_path / "empty", state, metadata=metadata)
    template = {"values": jnp.zeros(3), "empty": jnp.zeros((2, 0), dtype=jnp.int32)}
    restored = pp.checkpoint.restore(tmp_path / "empty", template, expected_metadata=metadata)
    assert_same(state, restored)
    assert restored["empty"].shape == (2, 0)


def test_repeated_save_restore_chain_matches_uninterrupted(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(4, 4), (0, 0)], 3))
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=3)
    initial = pp.runtime.init_lifetime(env, seed=7, window_size=4)
    complete = pp.runtime.Runner(env, pp.policies.random, ticks=30)(initial)
    chained = initial
    for index in range(10):
        chained = runner(chained)
        path = tmp_path / f"chain-{index}"
        pp.checkpoint.save(path, chained, metadata=runner.metadata)
        template = pp.runtime.init_lifetime(env, seed=100 + index, window_size=4)
        restored = pp.checkpoint.restore(path, template, expected_metadata=runner.metadata)
        assert_same(chained, restored)
        chained = restored
    assert_same(chained, complete)


def test_failed_save_leaves_no_checkpoint_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from perpetua import checkpoint as checkpoint_module

    env = pp.make("ContinualGrid-v0")
    state = pp.runtime.init_lifetime(env)
    path = tmp_path / "lifetime"

    def fail_after_partial_write(_self: Any, target: Path, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        Path(target).mkdir(parents=True)
        (Path(target) / "partial").write_text("not a checkpoint")
        raise RuntimeError("simulated interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_module.ocp.Checkpointer, "save", fail_after_partial_write)
        with pytest.raises(RuntimeError, match="simulated interrupted write"):
            pp.checkpoint.save(path, state, metadata=env.metadata)
    assert not path.exists()
    assert not list(tmp_path.glob(".lifetime.*.tmp"))
    pp.checkpoint.save(path, state, metadata=env.metadata)
    assert_same(state, pp.checkpoint.restore(path, state, expected_metadata=env.metadata))


def test_killed_save_leaves_no_restorable_checkpoint_and_keeps_the_previous_one(
    tmp_path: Path,
) -> None:
    env = pp.make("ContinualGrid-v0")
    state = pp.runtime.init_lifetime(env, seed=3)
    previous = tmp_path / "previous"
    pp.checkpoint.save(previous, state, metadata=env.metadata)
    interrupted = tmp_path / "interrupted"
    script = """
import sys, time
from pathlib import Path
import jax.numpy as jnp
import perpetua.checkpoint as cp

class SlowCheckpointer:
    def __init__(self, handler):
        self.handler = handler
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def save(self, path, args=None):
        Path(path).mkdir(parents=True)
        (Path(path) / "partial").write_text("partial")
        print("started", flush=True)
        time.sleep(60)

cp.ocp.Checkpointer = SlowCheckpointer
cp.save(sys.argv[1], {"x": jnp.arange(3)}, metadata={"checkpointable": True})
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(interrupted)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "started"
        child.kill()
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert child.returncode is not None
    assert child.returncode < 0
    assert not interrupted.exists()
    assert list(tmp_path.glob(".interrupted.*.tmp"))
    # The earlier checkpoint still restores, and the interrupted path can be reused.
    restored = pp.checkpoint.restore(previous, state, expected_metadata=env.metadata)
    assert_same(state, restored)
    pp.checkpoint.save(interrupted, state, metadata=env.metadata)
    assert_same(state, pp.checkpoint.restore(interrupted, state, expected_metadata=env.metadata))


def test_corrupted_checkpoint_bytes_are_rejected(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0")
    state = pp.runtime.init_lifetime(env)
    path = tmp_path / "corrupt"
    pp.checkpoint.save(path, state, metadata=env.metadata)
    target = next(
        item
        for item in (path / "state").rglob("*")
        if item.is_file() and item.parent.name == "d" and item.stat().st_size > 32
    )
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(data)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        pp.checkpoint.restore(path, state, expected_metadata=env.metadata)


def test_replay_repeats_a_lifetime_segment_from_a_checkpoint(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", drift=pp.drift.RandomWalk(low=0, high=3, scale=0.5))
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=5)
    carry = runner(pp.runtime.init_lifetime(env, seed=11, window_size=4))
    pp.checkpoint.save(tmp_path / "start", carry, metadata=runner.metadata)
    original: list[object] = []
    for _ in range(3):
        carry = runner(carry)
        original.append(carry)
    template = pp.runtime.init_lifetime(env, seed=99, window_size=4)
    replayed = pp.checkpoint.replay(tmp_path / "start", template, runner, chunks=3)
    assert len(replayed) == 3
    for left, right in zip(original, replayed, strict=True):
        assert_same(left, right)
    with pytest.raises(ValueError, match="positive"):
        pp.checkpoint.replay(tmp_path / "start", template, runner, chunks=0)


def test_replay_with_a_donating_runner_keeps_every_chunk(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=3)
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=8, donate=True)
    carry = runner(pp.runtime.init_lifetime(env, seed=0))
    pp.checkpoint.save(tmp_path / "ck", carry, metadata=runner.metadata)
    segment = pp.checkpoint.replay(
        tmp_path / "ck", pp.runtime.init_lifetime(env, seed=1), runner, chunks=3
    )
    assert [chunk.state.t.to_int() for chunk in segment] == [16, 24, 32]


@pytest.mark.parametrize(
    ("donate", "probed"), [(True, False), (True, True), (False, True)], ids=str
)
def test_replay_returns_readable_carries_for_bare_and_wrapped_runners(
    tmp_path: Path, donate: bool, probed: bool
) -> None:
    from perpetua.agents import PPO, PPOConfig, PPORunner

    def build() -> Any:
        agent = PPO(action_size=5, config=PPOConfig(rollout_steps=8, minibatches=1))
        runner = PPORunner(pp.make("ContinualGrid-v0", size=3), agent, donate=donate)
        return pp.probes.ProbedRunner(runner, memory=2) if probed else runner

    runner = build()
    assert runner.donates is donate
    carry = runner(runner.init(seed=0))
    pp.checkpoint.save(tmp_path / "ck", carry, metadata=runner.metadata)
    segment = pp.checkpoint.replay(tmp_path / "ck", runner.init(seed=1), runner, chunks=3)
    plain = build()
    expected = plain(plain.init(seed=0))
    for chunk in segment:
        expected = plain(expected)
        leaves = [leaf for leaf in jax.tree.leaves(chunk) if isinstance(leaf, jax.Array)]
        assert not any(leaf.is_deleted() for leaf in leaves)
        assert_same(expected, chunk)
        expected = jax.tree.map(jnp.copy, expected)


@pytest.mark.parametrize(
    ("metadata", "refused"),
    [
        ({"checkpointable": False}, True),
        ({"environment": {"checkpointable": False}}, True),
        ({"execution": "host"}, True),  # undeclared host simulator
        ({"environment": {"execution": "host"}}, True),
        ({"environment": {"execution": "host", "checkpointable": True}}, False),
        ({"environment": {"execution": "jax"}}, False),  # the JAX carry is the whole state
        ({}, False),
    ],
)
def test_checkpoint_support_follows_the_declared_or_default_capability(
    tmp_path: Path, metadata: dict[str, Any], refused: bool
) -> None:
    state = {"x": jnp.arange(3)}
    if refused:
        with pytest.raises(NotImplementedError, match="complete checkpointable state"):
            pp.checkpoint.save(tmp_path / "ckpt", state, metadata=metadata)
        with pytest.raises(NotImplementedError, match="complete checkpointable state"):
            pp.checkpoint.restore(tmp_path / "ckpt", state, expected_metadata=metadata)
        assert not (tmp_path / "ckpt").exists()
    else:
        pp.checkpoint.save(tmp_path / "ckpt", state, metadata=metadata)
        restored = pp.checkpoint.restore(tmp_path / "ckpt", state, expected_metadata=metadata)
        np.testing.assert_array_equal(np.asarray(restored["x"]), np.arange(3))


def test_replay_accepts_an_explicit_empty_expected_metadata(tmp_path: Path) -> None:
    state = {"x": jnp.arange(3)}
    pp.checkpoint.save(tmp_path / "ckpt", state, metadata={})
    replayed = pp.checkpoint.replay(
        tmp_path / "ckpt", state, lambda carry: carry, chunks=1, expected_metadata={}
    )
    np.testing.assert_array_equal(np.asarray(replayed[-1]["x"]), np.arange(3))


def test_restoring_across_jax_enable_x64_names_the_flag(tmp_path: Path) -> None:
    env = pp.make("ContinualGrid-v0", size=4, num_agents=2)
    runner = pp.runtime.Runner(env, pp.policies.random, ticks=3)
    state = runner(pp.runtime.init_lifetime(env, seed=0, window_size=4))
    pp.checkpoint.save(tmp_path / "saved", state, metadata=runner.metadata)
    with jax.enable_x64(True):
        other = pp.make("ContinualGrid-v0", size=4, num_agents=2)
        template = pp.runtime.init_lifetime(other, seed=0, window_size=4)
        with pytest.raises(ValueError, match=r"int32.*int64.*jax_enable_x64 must be set the same"):
            pp.checkpoint.restore(tmp_path / "saved", template, expected_metadata=runner.metadata)


def clean_revision(monkeypatch: pytest.MonkeyPatch, revision: str | None = "a" * 40) -> None:
    def known() -> tuple[str | None, bool | None]:
        return revision, None if revision is None else False

    monkeypatch.setattr(pp.reporting, "_source_revision", known)


def test_checkpoint_records_what_wrote_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A checkout with uncommitted changes warns on every restore; pin a clean one.
    clean_revision(monkeypatch)
    state = {"x": jnp.arange(3)}
    pp.checkpoint.save(tmp_path / "ckpt", state, metadata={})
    recorded = pp.checkpoint.provenance(tmp_path / "ckpt")
    assert {name: recorded[name] for name in pp.reporting.provenance()} == (
        pp.reporting.provenance()
    )
    assert recorded["orbax"] == importlib.metadata.version("orbax-checkpoint")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pp.checkpoint.restore(tmp_path / "ckpt", state, expected_metadata={})
    with pytest.raises(FileNotFoundError):
        pp.checkpoint.provenance(tmp_path / "absent")


def save_as_written_by(
    monkeypatch: pytest.MonkeyPatch, path: Path, state: object, software: dict[str, Any]
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(pp.checkpoint, "_software", lambda: software)
        pp.checkpoint.save(path, state, metadata={})


def test_restore_warns_once_when_the_software_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"x": jnp.arange(3)}
    written = {
        **pp.reporting.provenance(),
        "orbax": importlib.metadata.version("orbax-checkpoint"),
        "jax": "0.0.1",
        "revision": "0" * 40,
        "backend": "tpu",
    }
    save_as_written_by(monkeypatch, tmp_path / "ckpt", state, written)
    with pytest.warns(UserWarning, match="not known to come from the running software") as caught:
        restored = pp.checkpoint.restore(tmp_path / "ckpt", state, expected_metadata={})
    assert len(caught) == 1
    message = str(caught[0].message)
    assert f"jax 0.0.1 in the checkpoint, {jax.__version__} now" in message
    assert "backend tpu in the checkpoint" in message
    assert "revision 0000" in message
    assert "perpetua " not in message
    assert "orbax" not in message
    np.testing.assert_array_equal(np.asarray(restored["x"]), np.arange(3))
    assert pp.checkpoint.provenance(tmp_path / "ckpt")["jax"] == "0.0.1"


@pytest.mark.parametrize(
    ("written", "running", "expected"),
    [
        (("b" * 40, False), ("b" * 40, False), None),
        ((None, None), (None, None), None),
        (("b" * 40, None), ("b" * 40, None), None),
        (("b" * 40, False), ("c" * 40, False), f"revision {'b' * 40} in the checkpoint, "),
        ((None, None), ("b" * 40, False), f"revision unknown in the checkpoint, {'b' * 40} now"),
        (("b" * 40, False), (None, None), f"revision {'b' * 40} in the checkpoint, unknown now"),
        (("b" * 40, True), ("b" * 40, False), "the checkpoint was written from a revision with"),
        (("b" * 40, False), ("b" * 40, True), "the running revision has uncommitted changes"),
    ],
    ids=[
        "same",
        "both-unknown",
        "dirty-unknown",
        "differs",
        "unknown-then",
        "unknown-now",
        "dirty-then",
        "dirty-now",
    ],
)
def test_restore_warns_when_a_revision_does_not_identify_the_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    written: tuple[str | None, bool | None],
    running: tuple[str | None, bool | None],
    expected: str | None,
) -> None:
    state = {"x": jnp.arange(3)}
    monkeypatch.setattr(pp.reporting, "_source_revision", lambda: written)
    pp.checkpoint.save(tmp_path / "ckpt", state, metadata={})
    monkeypatch.setattr(pp.reporting, "_source_revision", lambda: running)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = pp.checkpoint.restore(tmp_path / "ckpt", state, expected_metadata={})
    np.testing.assert_array_equal(np.asarray(restored["x"]), np.arange(3))
    messages = [str(item.message) for item in caught if item.category is UserWarning]
    if expected is None:
        assert messages == []
    else:
        assert len(messages) == 1
        assert expected in messages[0]


def test_stored_arrays_of_another_checkpoint_are_refused(tmp_path: Path) -> None:
    template: dict[str, Any] = {
        "x": jnp.arange(3, dtype=jnp.float32),
        "k": jax.random.key(1),
        "n": 2,
    }
    metadata = {"run": 1}
    pp.checkpoint.save(tmp_path / "floats", template, metadata=metadata)
    for name, other in (
        ("ints", {**template, "x": jnp.arange(3, dtype=jnp.int32) + 10}),
        ("wide", {**template, "x": jnp.arange(4, dtype=jnp.float32)}),
        ("scalar", {**template, "n": 2.5}),
    ):
        pp.checkpoint.save(tmp_path / name, other, metadata=metadata)
        # The header of one checkpoint over the arrays of another.
        mixed = tmp_path / f"mixed-{name}"
        shutil.copytree(tmp_path / "floats", mixed)
        shutil.rmtree(mixed / "state")
        shutil.copytree(tmp_path / name / "state", mixed / "state")
        with pytest.raises(ValueError, match="do not match the template: leaf .* stored in the"):
            pp.checkpoint.restore(mixed, template, expected_metadata=metadata)
    with pytest.raises(ValueError, match=r"leaf \"\['x'\]\": int32\[3\] stored in the checkpoint"):
        pp.checkpoint.restore(tmp_path / "mixed-ints", template, expected_metadata=metadata)
    restored = pp.checkpoint.restore(tmp_path / "floats", template, expected_metadata=metadata)
    assert restored["x"].dtype == jnp.float32


@pytest.mark.parametrize("software", [None, {"jax": jax.__version__}])
def test_a_checkpoint_without_the_provenance_entries_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, software: dict[str, Any] | None
) -> None:
    # Checkpoints written before the revision was recorded hold jax, orbax and backend only.
    state = {"x": jnp.arange(3)}
    earlier = {
        "jax": jax.__version__,
        "orbax": importlib.metadata.version("orbax-checkpoint"),
        "backend": jax.default_backend(),
    }
    save_as_written_by(monkeypatch, tmp_path / "ckpt", state, software or earlier)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = pp.checkpoint.restore(tmp_path / "ckpt", state, expected_metadata={})
    np.testing.assert_array_equal(np.asarray(restored["x"]), np.arange(3))
    assert "revision" not in pp.checkpoint.provenance(tmp_path / "ckpt")
