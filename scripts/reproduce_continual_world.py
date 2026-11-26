"""Reproduce Continual World's CW10 fine-tuning result (single head, one-hot task id).

Target: Wołczyk et al. (2021), Table 14, "Fine-tuning, Single-head + one-hot":
performance 0.12 [0.12, 0.13], forgetting 0.68 [0.66, 0.70], forward transfer
0.16 [0.12, 0.19] (20 seeds, 90% bootstrap intervals). The protocol follows
the paper and its released code: SAC with four 256-unit layers (LayerNorm and
tanh after the first, leaky ReLU after the others), learning rate 1e-3, batch
128, 50 gradient steps every 50 ticks, a 1M replay buffer, gamma 0.99,
polyak 0.995, automatic temperature from 1 (alpha = e) towards a target
standard deviation of 0.089, no gradient clipping; per task, the replay
buffer, optimizer and temperature restart, the first 10k ticks act uniformly
at random and learning starts after 1k ticks; 1M ticks per task; a new goal
configuration every 200-tick episode; time-limit respawns kept out of TD
targets. The learner sees the task id as a one-hot input through the Oracle
(recorded in the run's metadata) and learns of switches through the Oracle's
``boundary``.

Evaluation: every ``--eval-every`` ticks the current task is scored, and at
each task end every task, by the success rate of 10 stochastic 200-tick
episodes (an episode succeeds if Meta-World reports success at any tick) on
separate evaluation lifetimes. Forward transfer needs ``reference`` runs: one
task learned from scratch with the same learner.

Known differences from the paper: Meta-World v3 tasks and rewards (the paper
used v1), 39-dimensional observations (v1 had 12), and a single temperature
reset at each switch instead of one per task (equivalent for CW10, which
never revisits a task).

    uv run --extra metaworld --extra training python scripts/reproduce_continual_world.py \\
        cl --seed 0 --out results/cw10-seed0.jsonl
    uv run ... reference --task 3 --seed 0 --out results/reference-task3-seed0.jsonl
    uv run ... metrics --cl results/cw10-seed*.jsonl --reference results/reference-*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

import perpetua as pp
from perpetua.adapters.host import MetaWorldLifetime
from perpetua.agents import SAC, SACConfig, SACRunner
from perpetua.presets import CW10

# Reference AUC at or above this marks a task as near-trivial for forward transfer.
TRIVIAL = 0.95

TARGET = {
    "source": "Wołczyk et al. 2021, Table 14 (CW10, fine-tuning, single head + one-hot)",
    "performance": [0.12, 0.12, 0.13],
    "forgetting": [0.68, 0.66, 0.70],
    "forward_transfer": [0.16, 0.12, 0.19],
}


def paper_config(
    *,
    boundary: bool,
    random_steps: int = 10_000,
    update_after: int = 1_000,
    buffer_capacity: int = 1_000_000,
    kernel_init: Literal["orthogonal", "glorot_uniform"] = "orthogonal",
) -> SACConfig:
    """The paper's SAC; ``boundary`` enables the per-task restarts."""
    return SACConfig(
        rollout_steps=50,
        gradient_steps=50,
        buffer_capacity=buffer_capacity,
        batch_size=128,
        hidden_size=256,
        hidden_layers=4,
        layer_norm=True,
        activation="leaky_relu",
        learning_rate=1e-3,
        gamma=0.99,
        tau=0.005,
        alpha=math.e,
        autotune_alpha=True,
        target_entropy=4 * math.log(0.089 * math.sqrt(2 * math.pi * math.e)),
        max_grad_norm=1e12,
        task_boundary_reset=boundary,
        random_steps=random_steps,
        update_after=update_after,
        mask_respawns=True,
        kernel_init=kernel_init,
    )


def _emit(out: Any, record: dict[str, Any]) -> None:
    line = json.dumps(record)
    out.write(line + "\n")
    out.flush()
    print(line, flush=True)


class Evaluator:
    """Success rates on separate host lifetimes that persist between evaluations."""

    def __init__(
        self, agent: SAC, tasks: list[str], *, seed: int, episode_steps: int, size: int
    ) -> None:
        self.act = cast(Any, jax.jit(agent.act))
        self.episode_steps = episode_steps
        self.size = size
        self.envs = [
            pp.Oracle(
                MetaWorldLifetime(
                    env_name=name,
                    seed=100_000 + 100 * seed + index,
                    max_episode_steps=episode_steps,
                    randomize_goals=True,
                ),
                expose=["success"],
            )
            for index, name in enumerate(tasks)
        ]
        self.states: list[Any] = [None] * len(tasks)
        self.observations: list[Any] = [None] * len(tasks)
        self.infos: list[Any] = [None] * len(tasks)
        self.key = jax.random.key(10_000 + seed)

    def success(self, agent_state: Any, task: int, code: int, episodes: int) -> float:
        """Fraction of ``episodes`` with success at any tick; ``code`` is the one-hot index."""
        env = self.envs[task]
        if self.states[task] is None:
            self.key, init_key = jax.random.split(self.key)
            self.states[task], self.observations[task], self.infos[task] = env.init(init_key)
        one_hot = jax.nn.one_hot(jnp.asarray([code]), self.size, dtype=jnp.float32)
        act = self.act
        state, obs, info = self.states[task], self.observations[task], self.infos[task]
        wins = 0
        for _ in range(episodes):
            won = False
            for _ in range(self.episode_steps):
                self.key, act_key, step_key = jax.random.split(self.key, 3)
                action = act(agent_state, jnp.concatenate([obs, one_hot], axis=-1), info, act_key)
                state, obs, _, info = env.step(state, action.action, step_key)
                won = won or bool(np.asarray(cast(Any, info)["oracle"]["success"])[0] > 0)
            wins += int(won)
        self.states[task], self.observations[task], self.infos[task] = state, obs, info
        return wins / episodes

    def close(self) -> None:
        for env in self.envs:
            cast(Any, env).env.close()


def train(args: argparse.Namespace) -> None:
    tasks: list[str] = [str(name) for name in CW10][: args.tasks]
    single = args.mode == "reference"
    names: list[str] = [tasks[args.task]] if single else tasks
    if single:
        lifetime: Any = MetaWorldLifetime(
            env_name=names[0],
            seed=args.seed,
            max_episode_steps=args.episode_steps,
            compiled=True,
            randomize_goals=True,
        )
        env: Any = pp.OneHotObservation(
            pp.Oracle(lifetime, expose=["task_id"]), field="task_id", size=len(tasks)
        )
    else:
        lifetime = pp.presets.continual_world(
            tasks,
            seed=args.seed,
            period=args.steps_per_task,
            max_episode_steps=args.episode_steps,
            compiled=True,
            randomize_goals=True,
        )
        env = pp.OneHotObservation(
            pp.Oracle(lifetime, expose=["task_id", "boundary"]), field="task_id", size=len(tasks)
        )
    box = cast(Any, env.action_space)
    config = paper_config(
        boundary=not single,
        random_steps=args.random_steps,
        update_after=args.update_after,
        buffer_capacity=args.buffer,
        kernel_init=args.kernel_init,
    )
    agent = SAC(action_low=box.low[0], action_high=box.high[0], config=config)
    runner = SACRunner(env, agent, donate=True)
    evaluator = Evaluator(
        agent, names, seed=args.seed, episode_steps=args.episode_steps, size=len(tasks)
    )
    chunk = agent.config.rollout_steps
    total = args.steps_per_task * len(names)
    if args.steps_per_task % chunk or args.eval_every % chunk:
        raise ValueError(f"--steps-per-task and --eval-every must be multiples of {chunk}.")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with out_path.open("w") as out:
        _emit(
            out,
            {
                "kind": "run",
                "mode": args.mode,
                "seed": args.seed,
                "tasks": names,
                "reference_task_index": args.task if single else None,
                "steps_per_task": args.steps_per_task,
                "eval_every": args.eval_every,
                "eval_episodes": args.eval_episodes,
                "episode_steps": args.episode_steps,
                "learner": agent.configuration,
                "oracle_exposure": list(env.metadata["oracle_exposure"]),
                "observation_appends": cast(Any, env.metadata)["configuration"][
                    "observation_appends"
                ],
                "backend": jax.default_backend(),
                "provenance": pp.reporting.provenance(),
                "target": TARGET,
            },
        )
        state = runner.init(seed=args.seed, window_size=1000)
        tick = 0
        try:
            while tick < total:
                state = runner(state)
                tick += chunk
                current = (tick - 1) // args.steps_per_task
                if tick % args.eval_every == 0:
                    code = 0 if single else current
                    rate = evaluator.success(
                        state.agent, current if not single else 0, code, args.eval_episodes
                    )
                    _emit(
                        out,
                        {
                            "kind": "eval",
                            "tick": tick,
                            "training_task": current,
                            "task": current,
                            "success": rate,
                            "window_reward": float(
                                np.asarray(state.lifetime.metrics.window_mean)[0]
                            ),
                            "alpha": float(np.exp(np.asarray(state.agent.log_alpha))),
                            "healthy": bool(state.healthy),
                            "elapsed_s": round(time.perf_counter() - started, 1),
                        },
                    )
                if not single and tick % args.steps_per_task == 0:
                    for task in range(len(names)):
                        _emit(
                            out,
                            {
                                "kind": "boundary_eval",
                                "tick": tick,
                                "training_task": current,
                                "task": task,
                                "success": evaluator.success(
                                    state.agent, task, task, args.eval_episodes
                                ),
                            },
                        )
                if not bool(state.healthy):
                    _emit(out, {"kind": "unhealthy", "tick": tick})
                    break
        finally:
            evaluator.close()
            cast(Any, env).close()
        healthy = bool(state.healthy)
        _emit(
            out,
            {
                "kind": "done",
                "tick": tick,
                "complete": tick >= total and healthy,
                "wall_s": round(time.perf_counter() - started, 1),
            },
        )
    if tick < total or not healthy:
        sys.exit(1)


def _records(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _bootstrap(values: list[float], *, seed: int = 0, samples: int = 10_000) -> list[float]:
    """Mean and 90% bootstrap interval across seeds, as the paper reports."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 1:
        return [float(data[0])] * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(samples, data.size), replace=True).mean(axis=1)
    return [float(data.mean()), float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))]


def _auc(rates: list[float]) -> float:
    """Normalised area under a success curve sampled at evenly spaced ticks."""
    return float(np.mean(rates)) if rates else math.nan


def _expand(patterns: list[str], flag: str) -> list[str]:
    """The files the glob patterns match; a flag whose patterns match nothing is an error."""
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if patterns and not paths:
        sys.exit(f"error: {flag} matched no files: {' '.join(patterns)}")
    return paths


def _completed(records: list[dict[str, Any]]) -> bool:
    """Whether a run ended with a ``done`` record that reports it complete and healthy."""
    done = [r for r in records if r["kind"] == "done"]
    return bool(done) and bool(done[-1].get("complete"))


def metrics(args: argparse.Namespace) -> None:
    reference: dict[int, list[list[float]]] = {}
    aborted: list[str] = []
    for path in _expand(args.reference, "--reference"):
        records = _records(path)
        if not _completed(records):
            aborted.append(path)
            continue
        run = records[0]
        rates = [r["success"] for r in records if r["kind"] == "eval"]
        reference.setdefault(int(run["reference_task_index"]), []).append(rates)
    if aborted:
        sys.exit(
            "error: reference runs that did not finish healthy (no done record with "
            f"complete=true) would bias the baseline; rerun or remove: {' '.join(aborted)}"
        )
    cl_paths = _expand(args.cl, "--cl")
    per_seed: list[dict[str, Any]] = []
    for path in cl_paths:
        records = _records(path)
        if not _completed(records):
            print(f"skip {path}: aborted (no done record with complete=true)", file=sys.stderr)
            continue
        run = records[0]
        n, steps = len(run["tasks"]), int(run["steps_per_task"])
        boundary = [r for r in records if r["kind"] == "boundary_eval"]
        final_tick = n * steps
        final = {r["task"]: r["success"] for r in boundary if r["tick"] == final_tick}
        at_end = {
            r["task"]: r["success"]
            for r in boundary
            if r["tick"] == (r["task"] + 1) * steps and r["task"] == r["training_task"]
        }
        if len(final) < n:
            print(f"skip {path}: incomplete ({len(final)}/{n} final evaluations)", file=sys.stderr)
            continue
        performance = float(np.mean([final[i] for i in range(n)]))
        forgetting = float(np.mean([at_end[i] - final[i] for i in range(n)]))
        # None where forward transfer is undefined (reference AUC of 1).
        transfers: list[float | None] = []
        per_task: list[dict[str, Any]] = []
        for i in range(n):
            curve = [r["success"] for r in records if r["kind"] == "eval" and r["task"] == i]
            if i in reference:
                if any(len(rates) != len(curve) for rates in reference[i]):
                    sys.exit(
                        f"error: task {i} has {len(curve)} evaluations in {path} but its "
                        f"reference runs have {sorted({len(rates) for rates in reference[i]})}; "
                        "use the same --steps-per-task and --eval-every, so that the baseline "
                        "covers the same window"
                    )
                base = _auc(list(np.mean(np.asarray(reference[i]), axis=0)))
                ft = (_auc(curve) - base) / (1.0 - base) if base < 1.0 else None
                transfers.append(ft)
                per_task.append(
                    {"task": run["tasks"][i], "auc": _auc(curve), "reference_auc": base, "ft": ft}
                )
        per_seed.append(
            {
                "path": path,
                "seed": run["seed"],
                "performance": performance,
                "forgetting": forgetting,
                "forward_transfer": _mean_defined(transfers) if len(transfers) == n else None,
                "forward_transfer_undefined_tasks": sum(ft is None for ft in transfers),
                # The paper avoids near-trivial tasks (reference AUC close to 1), where
                # the 1 - AUC denominator makes FT ill-conditioned; report both.
                "forward_transfer_nontrivial": (
                    _mean_defined([row["ft"] for row in per_task if row["reference_auc"] < TRIVIAL])
                    if len(per_task) == n
                    else None
                ),
                "per_task": per_task,
                "success_at_task_end": [at_end[i] for i in range(n)],
                "final_success": [final[i] for i in range(n)],
            }
        )
    if not per_seed:
        sys.exit(f"error: no complete CL run among {len(cl_paths)} file(s) matched by --cl")
    summary: dict[str, Any] = {
        "seeds": len(per_seed),
        "target": TARGET,
        "provenance": pp.reporting.provenance(),
        "per_seed": per_seed,
    }
    for name in ("performance", "forgetting", "forward_transfer", "forward_transfer_nontrivial"):
        values = [s[name] for s in per_seed if s[name] is not None]
        summary[name] = _bootstrap(values) if values else None
    print(json.dumps(summary, indent=2, allow_nan=False))


def _mean_defined(values: list[float | None]) -> float | None:
    """The mean of the defined values; None when there are none."""
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("cl", "reference"):
        mode = modes.add_parser(name)
        mode.add_argument("--seed", type=int, default=0)
        mode.add_argument("--out", required=True)
        mode.add_argument("--tasks", type=int, default=10, help="first N tasks of CW10")
        mode.add_argument("--steps-per-task", type=int, default=1_000_000)
        mode.add_argument("--eval-every", type=int, default=20_000)
        mode.add_argument("--eval-episodes", type=int, default=10)
        mode.add_argument("--episode-steps", type=int, default=200)
        mode.add_argument("--random-steps", type=int, default=10_000)
        mode.add_argument("--update-after", type=int, default=1_000)
        mode.add_argument("--buffer", type=int, default=1_000_000)
        mode.add_argument(
            "--kernel-init",
            choices=("orthogonal", "glorot_uniform"),
            default="orthogonal",
            help="weight initialisation; the paper's code uses Keras's Glorot uniform default",
        )
        if name == "reference":
            mode.add_argument("--task", type=int, required=True, help="index into CW10")
    summary = modes.add_parser("metrics")
    summary.add_argument("--cl", nargs="+", required=True)
    summary.add_argument("--reference", nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == "metrics":
        metrics(args)
    else:
        if args.mode == "cl":
            args.task = 0
        train(args)


if __name__ == "__main__":
    main()
