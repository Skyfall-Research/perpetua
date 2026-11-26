"""Measure a standard AsyncVectorEnv + PyTorch PPO baseline (claim-1 comparison).

Reference stack for the throughput claim (docs/measurement.md): Gymnasium
``AsyncVectorEnv`` workers with a CleanRL-style PyTorch PPO learner on the
configured device. Reports environment steps/s over a fixed wall-clock
duration after warm-up, plus episode returns as evidence the learner is
actually training.

Gymnasium is a core dependency of Perpetua and PyTorch is its optional
``torch`` extra, so the project environment with that extra runs this::

    uv run --extra torch python scripts/measure_baseline.py --duration 60

For a GPU figure the installed torch has to be a build for that device.

Timings exclude warm-up and synchronize the device; the harness reports
throughput, not learning quality, and the same environment class can be
measured on both stacks (for example CartPole-v1).
"""

# torch comes from the optional torch extra.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import argparse
import json
import sys
import time
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

try:
    from perpetua.reporting import provenance
except ImportError:  # the baseline also runs where Perpetua is not installed
    provenance = None


def make_env(env_id: str, seed: int, index: int, max_steps: int):
    """Return an environment thunk for one vector worker."""

    def thunk():
        env = gym.make(env_id, max_episode_steps=max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + index)
        return env

    return thunk


class ActorCritic(nn.Module):
    """Standard two-layer tanh actor-critic, as in CleanRL's PPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def _get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """Sample (or score) actions and return log-prob, entropy and value."""
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        distribution = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value


def main() -> None:
    """Run the baseline and print a JSON throughput report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="run to this many environment steps instead of for --duration seconds",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print a progress line to stderr every this many updates (0: never)",
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the actor-critic; compile time falls in warm-up",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.num_envs, args.num_steps, args.warmup_updates, args.update_epochs) < 1:
        raise ValueError("num_envs, num_steps, warmup_updates and update_epochs must be positive.")
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // max(args.num_minibatches, 1)
    if args.num_minibatches < 1 or batch_size % args.num_minibatches or minibatch_size < 2:
        # The advantage's standard deviation is NaN on a minibatch of one sample.
        raise ValueError(
            f"num_minibatches ({args.num_minibatches}) must divide the rollout batch size "
            f"num_envs * num_steps ({batch_size}) into minibatches of at least 2 samples."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(args.env_id, args.seed, index, args.max_episode_steps)
            for index in range(args.num_envs)
        ]
    )
    obs_dim = int(np.prod(cast(tuple[int, ...], envs.single_observation_space.shape)))
    action_dim = int(cast(spaces.Discrete, envs.single_action_space).n)
    agent: Any = ActorCritic(obs_dim, action_dim).to(device)
    if args.compile:
        agent = torch.compile(agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    obs, _ = envs.reset(seed=args.seed)
    observations = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    log_probs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    episode_returns: list[float] = []

    def update(obs: np.ndarray) -> np.ndarray:
        for step in range(args.num_steps):
            observations[step] = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, _, value = agent._get_action_and_value(observations[step])
            actions[step] = action
            log_probs[step] = log_prob
            values[step] = value
            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            dones[step] = torch.as_tensor(done, dtype=torch.float32, device=device)
            if "episode" in info:
                finished = np.asarray(info["episode"]["r"])[np.asarray(done, dtype=bool)]
                episode_returns.extend(float(value) for value in finished)
            obs = next_obs
        with torch.no_grad():
            next_value = agent.critic(
                torch.as_tensor(obs, dtype=torch.float32, device=device)
            ).squeeze(-1)
            advantages = torch.zeros_like(rewards)
            last_advantage = torch.zeros(args.num_envs, device=device)
            for step in reversed(range(args.num_steps)):
                next_nonterminal = 1.0 - dones[step]
                next_values = next_value if step == args.num_steps - 1 else values[step + 1]
                delta = rewards[step] + 0.99 * next_values * next_nonterminal - values[step]
                last_advantage = delta + 0.99 * 0.95 * next_nonterminal * last_advantage
                advantages[step] = last_advantage
            returns = advantages + values
        flat_obs = observations.reshape(-1, obs_dim)
        flat_actions = actions.reshape(-1)
        flat_log_probs = log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        indices = torch.randperm(batch_size, device=device)
        for _ in range(args.update_epochs):
            for start in range(0, batch_size, minibatch_size):
                index = indices[start : start + minibatch_size]
                _, new_log_prob, entropy, new_value = agent._get_action_and_value(
                    flat_obs[index], flat_actions[index]
                )
                ratio = (new_log_prob - flat_log_probs[index]).exp()
                advantage = flat_advantages[index]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                policy_loss = torch.max(
                    -advantage * ratio,
                    -advantage * torch.clamp(ratio, 0.8, 1.2),
                ).mean()
                value_loss = 0.5 * ((new_value - flat_returns[index]) ** 2).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        return obs

    for _ in range(args.warmup_updates):
        obs = update(obs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    measured_steps = 0
    updates = 0
    started = time.perf_counter()
    while (
        measured_steps < args.total_steps
        if args.total_steps
        else time.perf_counter() - started < args.duration
    ):
        obs = update(obs)
        measured_steps += batch_size
        updates += 1
        if args.progress_every and updates % args.progress_every == 0:
            progress = {
                "steps": measured_steps,
                "seconds": time.perf_counter() - started,
                "mean_episode_return": (
                    float(np.mean(episode_returns[-100:])) if episode_returns else None
                ),
            }
            print(json.dumps(progress), file=sys.stderr, flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    envs.close()
    report = {
        "stack": "async-vector-env-pytorch-ppo",
        "env_id": args.env_id,
        "device": str(device),
        "torch_version": torch.__version__,
        "gymnasium_version": gym.__version__,
        "provenance": None if provenance is None else provenance(),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "batch_size": batch_size,
        "updates": updates,
        "steps_measured": measured_steps,
        "total_steps_target": args.total_steps or None,
        "seconds": seconds,
        "steps_per_second": measured_steps / seconds,
        "mean_episode_return": (
            float(np.mean(episode_returns[-100:])) if episode_returns else None
        ),
        "episodes_finished": len(episode_returns),
        "torch_compile": bool(args.compile),
        "note": (
            "CleanRL-style PPO with AsyncVectorEnv workers; timings exclude "
            "warm-up (including torch.compile compilation, which a tuned "
            "baseline amortizes) and synchronize CUDA. Throughput harness, "
            "not a tuned benchmark entry."
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
