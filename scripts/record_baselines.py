"""Record the seeded results that ``tests/test_baselines.py`` compares against.

Each run in ``RUNS`` is a few seconds of a seeded lifetime: a random policy on
one environment of each family that installs everywhere, or a built-in
learner for a few updates. A run is reduced to a small digest (lifetime mean
reward per slot, final tick, health, checksums of the final observation, for
random policies checksums of every action and observation, and for learners
the last losses and the sum and absolute sum of every parameter module). The
digests of this revision are written, with the platform they were computed
on, to the fixture of that platform,
``tests/fixtures/baselines-<sys.platform>-<machine>.json``:

    uv run --all-extras python scripts/record_baselines.py

Each platform has its own fixture because floating-point rounding differs
between them and a learner amplifies the difference. Re-record only when a
change to seeded results is intended, and list that change under "Changes
results" in ``CHANGELOG.md``. ``baselines-darwin-arm64.json`` is recorded on
an Apple Silicon Mac; ``baselines-linux-x86_64.json`` is recorded on a GitHub
runner by the ``Record baselines`` workflow (``.github/workflows/baselines.yml``),
which uploads it as an artifact to commit.

``--compare`` records nothing: it prints the largest relative difference
between a recording and the committed fixture (or ``--against`` another
file), which is how the spread between two machines of one platform is
measured. With ``--recording`` it reads a recorded file instead of running.
"""

import argparse
import json
import math
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
from gymnasium import spaces

import perpetua as pp

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def fixture_path(sys_platform: str | None = None, machine: str | None = None) -> Path:
    """The fixture of a platform (this one by default); CPU backend, 32-bit."""
    name = f"baselines-{sys_platform or sys.platform}-{machine or platform.machine()}.json"
    return FIXTURES / name


type Digest = dict[str, Any]


def current_platform() -> dict[str, str | bool]:
    """Where digests are computed; exact comparison needs all of it to match."""
    return {
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "backend": jax.default_backend(),
        "x64": bool(jax.config.jax_enable_x64),
    }


def _values(value: Any) -> float | list[float]:
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else [float(entry) for entry in array.reshape(-1)]


def checksum(array: Any) -> dict[str, float]:
    """Exactly rounded sum, absolute sum and position-weighted sum of an array.

    The weighted sum (entry ``i`` times ``i + 1``) moves when a value changes
    place, which the plain sums of a one-hot image do not.
    """
    values = np.asarray(array, dtype=np.float64).reshape(-1).tolist()
    return {
        "sum": math.fsum(values),
        "abs_sum": math.fsum(abs(value) for value in values),
        "weighted_sum": math.fsum(value * index for index, value in enumerate(values, 1)),
    }


def lifetime_digest(lifetime: Any) -> Digest:
    """Mean reward per slot, final tick, health and final observation of a lifetime carry."""
    return {
        "mean_reward": _values(lifetime.metrics.mean),
        "tick": lifetime.state.t.to_int(),
        "healthy": bool(lifetime.metrics.healthy),
        "observation": checksum(lifetime.obs),
    }


def trajectory_digest(env: Any, policy: Any, lifetime: Any, ticks: int) -> Digest:
    """Checksums of every action taken and observation seen over ``ticks`` ticks."""

    def advance(carry: Any, _: None) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        carry, action, _reward = pp.runtime.interact(env, carry, policy)
        return carry, (action, carry.obs)

    run = jax.jit(lambda carry: jax.lax.scan(advance, carry, None, length=ticks)[1])
    actions, observations = run(lifetime)
    return {"actions": checksum(actions), "observations": checksum(observations)}


def parameter_digest(params: Any) -> dict[str, dict[str, float]]:
    """Exactly rounded sum and absolute sum of each module's parameters.

    A module is the entry directly under a network's ``"params"``, prefixed by
    the network's name where a learner has several (SAC's actor and critics).
    """
    groups: dict[str, list[float]] = {}
    for path, leaf in jax.tree.leaves_with_path(params):
        labels = [str(getattr(step, "key", getattr(step, "name", step))) for step in path]
        module = labels.index("params") + 1
        name = "/".join(label for label in labels[: module + 1] if label != "params")
        groups.setdefault(name, []).extend(np.asarray(leaf, dtype=np.float64).reshape(-1).tolist())
    return {
        name: {"sum": math.fsum(values), "abs_sum": math.fsum(abs(value) for value in values)}
        for name, values in sorted(groups.items())
    }


def learner_digest(state: Any) -> Digest:
    """A lifetime digest plus the learner's parameters, step count and last losses."""
    statistics = state.statistics
    return {
        **lifetime_digest(state.lifetime),
        "healthy": bool(state.healthy),
        "gradient_steps": np.asarray(state.agent.gradient_steps.lo).tolist(),
        "losses": {name: _values(getattr(statistics, name)) for name in statistics._fields},
        "params": parameter_digest(state.agent.params),
    }


def _random(make: Callable[[], Any], *, ticks: int = 256, seed: int = 0) -> Digest:
    env = make()
    policy = pp.policies.uniform(env.action_space)
    runner = pp.runtime.Runner(env, policy, ticks=ticks)
    lifetime = pp.runtime.init_lifetime(env, seed=seed, window_size=16)
    return {
        **lifetime_digest(runner(lifetime)),
        "trajectory": trajectory_digest(env, policy, lifetime, ticks),
    }


def _train(runner: Any, *, updates: int = 3, seed: int = 0) -> Any:
    state = runner.init(seed=seed, window_size=16)
    for _ in range(updates):
        state = runner(state)
    return state


def _grid() -> Any:
    return pp.make(
        "ContinualGrid-v0", size=3, num_agents=4, drift=pp.drift.Cyclic([(2, 2), (0, 0)], 24)
    )


def _minatar() -> Any:
    return pp.make("gymnax/Breakout-MinAtar")


def _meal() -> Any:
    return pp.make("meal/overcooked-v0")


def _ppo(
    make: Callable[[], Any], *, ippo: bool = False, probed: bool = False, **config: Any
) -> Digest:
    from perpetua.agents import IPPO, PPO, PPOConfig, PPORunner

    env = make()
    settings = PPOConfig(
        **{"rollout_steps": 16, "epochs": 2, "minibatches": 2, "hidden_size": 16, **config}
    )
    action_size = int(cast(spaces.MultiDiscrete, env.action_space).nvec[0])
    agent = (IPPO if ippo else PPO)(
        action_size=action_size, config=settings, observation_space=env.observation_space
    )
    runner: Any = PPORunner(env, agent)
    if not probed:
        return learner_digest(_train(runner))
    tracked = pp.probes.ProbedRunner(runner, every=1, window=4, memory=2)
    carry = _train(tracked)
    return {
        **learner_digest(carry.state),
        "probes": {name: _values(value) for name, value in sorted(carry.trace.latest.items())},
    }


def _dqn(make: Callable[[], Any], *, iql: bool = False, **config: Any) -> Digest:
    from perpetua.agents import DQN, IQL, DQNConfig, DQNRunner

    env = make()
    settings = DQNConfig(
        **{
            "rollout_steps": 16,
            "buffer_capacity": 64,
            "batch_size": 16,
            "gradient_steps": 2,
            "hidden_size": 16,
            "target_update_interval": 3,
            **config,
        }
    )
    action_size = int(cast(spaces.MultiDiscrete, env.action_space).nvec[0])
    agent = (IQL if iql else DQN)(
        action_size=action_size, config=settings, observation_space=env.observation_space
    )
    return learner_digest(_train(DQNRunner(env, agent)))


def _sac(**config: Any) -> Digest:
    from perpetua.agents import SAC, SACConfig, SACRunner

    env = pp.make("gymnax/Pendulum-v1")
    box = cast(spaces.Box, env.action_space)
    settings = SACConfig(
        **{
            "rollout_steps": 16,
            "buffer_capacity": 64,
            "batch_size": 16,
            "gradient_steps": 2,
            "hidden_size": 16,
            **config,
        }
    )
    agent = SAC(action_low=box.low[0], action_high=box.high[0], config=settings)
    return learner_digest(_train(SACRunner(env, agent)))


def meal_pot_fill(seed: int = 0) -> tuple[jax.Array, jax.Array]:
    """MEAL Overcooked observations just before and after a pot gets its second onion.

    A random policy almost never fills a pot, so the state is arranged: cook 0
    stands at a free cell next to a pot that holds one onion, faces it and
    carries an onion. One real environment step with ``interact`` then puts
    the second onion in, and the pot's count layer goes from 1 to 2.
    """
    import jax.numpy as jnp
    from meal.env.overcooked.common import (  # pyright: ignore[reportMissingTypeStubs]
        DIR_TO_VEC,
        OBJECT_TO_INDEX,
    )

    env = cast(Any, _meal())
    state, _, _ = env.init(jax.random.key(seed))
    world = state.env.world
    maze = np.array(world.maze_map)
    directions = np.asarray(DIR_TO_VEC)
    pots = np.asarray(world.pot_pos)[np.asarray(world.pot_mask)]
    pot, cell, facing = next(
        (pot, pot - step, index)
        for pot in pots.astype(np.int64)
        for index, step in enumerate(directions)
        if maze[(pot - step)[1], (pot - step)[0], 0] == OBJECT_TO_INDEX["empty"]
    )
    old = np.asarray(world.agent_pos)[0]
    maze[old[1], old[0]] = (OBJECT_TO_INDEX["empty"], 0, 0)
    maze[cell[1], cell[0]] = (OBJECT_TO_INDEX["agent"], 0, facing)
    maze[pot[1], pot[0], 2] = int(world.pot_empty_status) - 1
    world = world.replace(
        maze_map=jnp.asarray(maze),
        agent_pos=world.agent_pos.at[0].set(jnp.asarray(cell, dtype=world.agent_pos.dtype)),
        agent_dir_idx=world.agent_dir_idx.at[0].set(facing),
        agent_dir=world.agent_dir.at[0].set(
            jnp.asarray(directions[facing], dtype=world.agent_dir.dtype)
        ),
        agent_inv=world.agent_inv.at[0].set(OBJECT_TO_INDEX["onion"]),
    )
    before = env.env._obs(env.env.backend.get_obs(world))  # pyright: ignore[reportPrivateUsage]
    state = state._replace(env=state.env._replace(world=world, obs=before))
    interact_and_stay = jnp.asarray([5, 4], dtype=jnp.int32)
    _, after, _, _ = env.step(state, interact_and_stay, jax.random.key(seed + 1))
    return before, after


def _meal_pot_fill_digest() -> Digest:
    from perpetua.agents import PPO, PPOConfig

    before, after = meal_pot_fill()
    agent = PPO(
        action_size=6,
        config=PPOConfig(torso="cnn", hidden_size=16),
        observation_space=_meal().observation_space,
    )
    learner = agent.init(before, jax.random.key(0))
    digest: Digest = {}
    for name, frame in (("before", before), ("after", after)):
        logits, value = agent.evaluate(learner, frame)
        digest[name] = {
            "largest": float(frame.max()),
            "logits": _values(logits),
            "value": _values(value),
        }
    return digest


def _forager() -> Any:
    return pp.make(
        "ForagerGrid-v0",
        num_agents=2,
        drift=pp.drift.Accumulate(gain=4.0, decay=0.9, low=1.0, high=8.0),
    )


# The environment and the number of ticks of each random-policy run.
RANDOM: dict[str, tuple[Callable[[], Any], int]] = {
    "random/continual-grid-cyclic": (_grid, 256),
    "random/forager-grid-accumulate": (_forager, 256),
    "random/gymnax-cartpole": (lambda: pp.make("gymnax/CartPole-v1"), 256),
    "random/jaxmarl-mpe-simple-tag": (lambda: pp.make("jaxmarl/MPE_simple_tag_v3"), 64),
    "random/mjx-pendulum": (lambda: pp.make("mjx/Pendulum-v0"), 64),
    "random/meal-overcooked": (_meal, 64),
}

# The optional dependency group a run needs beyond the default environment.
NEEDS: dict[str, str] = {
    "random/meal-overcooked": "meal",
    "ppo/cnn-meal": "meal",
    "ippo/cnn-meal": "meal",
    "ppo/cnn-meal-pot-fill": "meal",
}

RUNS: dict[str, Callable[[], Digest]] = {
    **{
        name: lambda make=make, ticks=ticks: _random(make, ticks=ticks)
        for name, (make, ticks) in RANDOM.items()
    },
    "ppo/mlp": lambda: _ppo(_grid),
    "ppo/cnn-minatar": lambda: _ppo(_minatar, torso="cnn"),
    "ppo/cnn-meal": lambda: _ppo(_meal, torso="cnn"),
    "ippo/cnn-meal": lambda: _ppo(_meal, ippo=True, torso="cnn"),
    "ppo/reward-centering": lambda: _ppo(_grid, reward_centering=True),
    "ppo/reset-interval": lambda: _ppo(_grid, reset_interval=2),
    "ppo/l2": lambda: _ppo(_grid, l2_coefficient=1e-3),
    "ppo/crelu": lambda: _ppo(_grid, activation="crelu"),
    "ppo/probed": lambda: _ppo(_grid, probed=True),
    "ppo/cnn-meal-pot-fill": _meal_pot_fill_digest,
    "dqn/mlp": lambda: _dqn(_grid),
    "dqn/cnn-minatar": lambda: _dqn(_minatar, torso="cnn"),
    "iql/mlp": lambda: _dqn(_grid, iql=True),
    "dqn/reward-centering": lambda: _dqn(_grid, reward_centering=True),
    "dqn/reset-interval": lambda: _dqn(_grid, reset_interval=3),
    "dqn/l2": lambda: _dqn(_grid, l2_coefficient=1e-3),
    "dqn/crelu": lambda: _dqn(_grid, activation="crelu"),
    "sac/mlp": lambda: _sac(),
    "sac/reward-centering": lambda: _sac(reward_centering=True),
    "sac/reset-interval": lambda: _sac(reset_interval=3),
    "sac/l2": lambda: _sac(l2_coefficient=1e-3),
}


def record() -> dict[str, Any]:
    """Compute every digest on this machine."""
    return {
        "platform": current_platform(),
        "jax": jax.__version__,
        "runs": {name: run() for name, run in RUNS.items()},
    }


def flatten(digest: Any, prefix: str = "") -> dict[str, bool | int | float]:
    """One entry per number, keyed by its path in the digest."""
    if isinstance(digest, dict):
        items: Any = cast(dict[str, Any], digest).items()
    elif isinstance(digest, list):
        items = enumerate(cast(list[Any], digest))
    else:
        return {prefix: digest}
    flat: dict[str, bool | int | float] = {}
    for key, value in items:
        flat.update(flatten(value, f"{prefix}/{key}" if prefix else str(key)))
    return flat


def compare(recording: dict[str, Any], fixture: dict[str, Any]) -> tuple[list[str], float, str]:
    """How far a recording is from a fixture.

    Returns the entries that are missing on one side or whose boolean or
    integer value differs, the largest relative difference between two
    recorded floats (``|new - old| / |old|``, infinite where ``old`` is 0 and
    ``new`` is not) and the path of that float.
    """
    now, then = flatten(recording["runs"]), flatten(fixture["runs"])
    broken = [f"only in one file: {path}" for path in sorted(now.keys() ^ then.keys())]
    largest, where = 0.0, ""
    for path in sorted(now.keys() & then.keys()):
        new, old = now[path], then[path]
        if type(new) is not type(old) or (not isinstance(old, float) and new != old):
            broken.append(f"{path}: {old!r} -> {new!r}")
        elif isinstance(old, float) and new != old:
            relative = abs(new - old) / abs(old) if old else math.inf
            if relative > largest:
                largest, where = relative, path
    return broken, largest, where


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=None, help="where to write the recording")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="write nothing; print the largest relative difference from the fixture",
    )
    parser.add_argument(
        "--recording", type=Path, default=None, help="with --compare: a recorded file to read"
    )
    parser.add_argument(
        "--against", type=Path, default=None, help="with --compare: the file to compare with"
    )
    args = parser.parse_args()
    if not args.compare:
        if args.recording is not None or args.against is not None:
            parser.error("--recording and --against need --compare.")
        output = fixture_path() if args.output is None else args.output
        recorded = record()
        output.write_text(json.dumps(recorded, indent=2) + "\n")
        print(f"wrote {len(recorded['runs'])} baselines to {output}")
        return
    if args.output is not None:
        parser.error("--compare writes nothing; drop --output.")
    against = fixture_path() if args.against is None else args.against
    if not against.exists():
        parser.exit(2, f"no fixture to compare with at {against}\n")
    fixture = json.loads(against.read_text())
    recording = record() if args.recording is None else json.loads(args.recording.read_text())
    broken, largest, where = compare(recording, fixture)
    print(f"recorded on {recording['platform']}, jax {recording['jax']}")
    print(f"compared with {against} ({fixture['platform']}, jax {fixture['jax']})")
    for line in broken:
        print(line)
    print(f"largest relative difference: {largest:.3e}" + (f" at {where}" if where else ""))
    if broken:
        parser.exit(1, f"{len(broken)} entries are missing or differ in a boolean or an integer\n")


if __name__ == "__main__":
    main()
