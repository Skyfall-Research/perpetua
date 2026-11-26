"""Run PyTorch agents on any Perpetua lifetime.

A PyTorch policy cannot run inside a compiled ``lax.scan``, so this is a host
loop: each tick the observation and masks cross to PyTorch through DLPack (no
copy when both frameworks use the same device), the agent acts, and the action
crosses back to :func:`perpetua.runtime.transition`, the same per-tick body the
JAX runners use. Metrics, recovery tracking and the Oracle therefore behave
exactly as for the JAX learners; there is no reference policy, so no regret.

:class:`TorchRunner` collects ``rollout_steps`` ticks and calls the agent's
``update``. :class:`TorchPPO` is a reference continuing PPO for discrete
actions. Requires the optional ``torch`` extra.
"""

# torch ships partial type information.
# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportAttributeAccessIssue=false, reportUnknownLambdaType=false
# pyright: reportUntypedBaseClass=false

from __future__ import annotations

import importlib.metadata
import shutil
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from perpetua import checkpoint, metrics, runtime
from perpetua.core import Clock, Environment, Info, as_int, requires_extra

with requires_extra("torch", "perpetua.torch_bridge"):
    import torch

__all__ = [
    "to_torch",
    "from_torch",
    "TorchRollout",
    "TorchAgent",
    "TorchTrainingState",
    "TorchRunner",
    "TorchPPOConfig",
    "TorchPPO",
]


def to_torch(array: Any, device: torch.device | str | None = None) -> torch.Tensor:
    """A JAX array as a tensor: zero-copy through DLPack, then moved if asked."""
    tensor = torch.from_dlpack(jnp.asarray(array))
    return tensor if device is None else tensor.to(device)


def from_torch(tensor: torch.Tensor) -> jax.Array:
    """A tensor as a JAX array through DLPack (zero-copy when contiguous)."""
    return jax.dlpack.from_dlpack(tensor.detach().contiguous())


@dataclass
class TorchRollout:
    """One rollout on the agent's device; leading axes are (time, agent slots).

    ``extras`` holds whatever the agent returned from ``act`` per tick (for
    example log-probabilities and values), stacked over time. ``last_obs`` is
    the observation after the final tick, for bootstrapping.
    """

    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    action_mask: torch.Tensor
    active: torch.Tensor
    last_obs: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


class TorchAgent(Protocol):
    """What :class:`TorchRunner` needs from a PyTorch agent.

    ``act`` returns one action per agent slot and a dict of per-slot extras.
    Its ``info`` holds the public ``action_mask``, ``active_mask``, ``respawn``
    and ``space_change`` as tensors, the tick ``t`` as an int64 scalar and,
    under a :class:`perpetua.Oracle`, ``oracle``: a dict of the exposed fields.
    ``update`` learns from a rollout and returns float statistics. The state
    dict must include everything a resume needs (networks, optimizer, RNG).
    """

    device: torch.device

    @property
    def configuration(self) -> dict[str, object]: ...

    def act(
        self, obs: torch.Tensor, info: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    def update(self, rollout: TorchRollout) -> dict[str, float]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class TorchTrainingState[StateT](NamedTuple):
    """The lifetime carry plus the update count; the agent holds its own tensors.

    ``statistics`` is the dict the agent's last ``update`` returned (empty
    after ``init`` and ``restore``); it is not checkpointed.

    This carry is not the whole training state, so
    :func:`perpetua.checkpoint.save`, ``restore`` and ``replay`` refuse it.
    :meth:`TorchRunner.save` and :meth:`TorchRunner.restore` checkpoint the
    lifetime, the update count and the agent's state dict together.
    """

    lifetime: runtime.Lifetime[StateT]
    updates: Clock
    statistics: dict[str, float]


def _float_tensors(value: object) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            yield value
    elif isinstance(value, Mapping):
        for item in cast(Mapping[object, object], value).values():
            yield from _float_tensors(item)
    elif isinstance(value, list | tuple):
        for item in cast(Sequence[object], value):
            yield from _float_tensors(item)


class TorchRunner[StateT]:
    """Collect continuing rollouts with a PyTorch agent and update between them.

    Works with native JAX environments (each tick's transition is compiled) and
    host simulators. Checkpoints hold the lifetime through
    :mod:`perpetua.checkpoint` and the agent's state dict beside it, written
    together into one atomically renamed directory by :meth:`save`. The
    generic :func:`perpetua.checkpoint.save`, ``restore`` and ``replay``
    refuse this runner and its carry, which does not contain the agent. There
    is no ``reference`` policy, so the metrics carry no regret.
    """

    def __init__(
        self, env: Environment[StateT], agent: TorchAgent, *, rollout_steps: int = 128
    ) -> None:
        rollout_steps = as_int(
            rollout_steps, "rollout_steps must be a positive integer.", minimum=1
        )
        # Agents with minibatch settings refuse a rollout they cannot split now,
        # not after the first rollout has been collected.
        check = getattr(agent, "check_rollout", None)
        if check is not None:
            check(rollout_steps, env.num_agents)
        self.env = env
        self.agent = agent
        self.rollout_steps = rollout_steps
        self.metadata: dict[str, object] = {
            "environment": env.metadata.copy(),
            "learner": agent.configuration,
            "framework": f"torch-{importlib.metadata.version('torch')}",
            "rollout_steps": rollout_steps,
        }
        if env.metadata.get("execution") == "host":
            update = cast(metrics.Updater, jax.jit(metrics.update))

            def step(lifetime: Any, action: jax.Array) -> Any:
                return runtime.transition(env, lifetime, action, update=update)

            self._step = step
        else:
            self._step = jax.jit(lambda lifetime, action: runtime.transition(env, lifetime, action))
        discrete = hasattr(env.action_space, "nvec")
        self._action_dtype = jnp.int32 if discrete else jnp.float32

    def init(self, *, seed: int = 0, window_size: int = 128) -> TorchTrainingState[StateT]:
        """Start one lifetime; the agent's own seed governs its sampling."""
        lifetime = runtime.init_lifetime(self.env, seed=seed, window_size=window_size)
        return TorchTrainingState(lifetime, Clock.from_int(0), {})

    def _to_jax(self, action: torch.Tensor, like: jax.Array) -> jax.Array:
        """Hand the action back on the lifetime's device, copying only if needed."""
        try:
            array = from_torch(action)
        except (RuntimeError, TypeError, ValueError):
            array = jnp.asarray(action.detach().cpu().numpy())
        target = next(iter(like.devices()))
        if next(iter(array.devices())) != target:
            array = jax.device_put(array, target)
        return array.astype(self._action_dtype)

    def _public(self, info: Info) -> dict[str, Any]:
        device = self.agent.device
        fields = cast(dict[str, Any], info)
        public: dict[str, Any] = {
            name: to_torch(fields[name], device)
            for name in ("action_mask", "active_mask", "respawn", "space_change")
            if name in fields
        }
        clock = fields["t"]
        high = to_torch(clock.hi, device).to(torch.int64)
        public["t"] = (high << 32) | to_torch(clock.lo, device).to(torch.int64)
        if "oracle" in fields:
            public["oracle"] = jax.tree.map(lambda leaf: to_torch(leaf, device), fields["oracle"])
        return public

    @property
    def donates(self) -> bool:
        """Always False: a call leaves the carry passed in readable."""
        return False

    def __call__(self, state: TorchTrainingState[StateT]) -> TorchTrainingState[StateT]:
        """One rollout of ``rollout_steps`` ticks, then one agent update."""
        device = self.agent.device
        lifetime = state.lifetime
        observations, actions, rewards, masks, actives = [], [], [], [], []
        extras: dict[str, list[torch.Tensor]] = {}
        for _ in range(self.rollout_steps):
            obs = to_torch(lifetime.obs, device)
            public = self._public(lifetime.info)
            with torch.no_grad():
                action, extra = self.agent.act(obs, public)
            lifetime, reward = self._step(lifetime, self._to_jax(action, lifetime.obs))
            observations.append(obs)
            actions.append(action)
            rewards.append(to_torch(reward, device))
            masks.append(public["action_mask"])
            actives.append(public["active_mask"])
            for name, value in extra.items():
                extras.setdefault(name, []).append(value)
        rollout = TorchRollout(
            torch.stack(observations),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(masks),
            torch.stack(actives),
            to_torch(lifetime.obs, device),
            {name: torch.stack(values) for name, values in extras.items()},
        )
        statistics = self.agent.update(rollout)
        return TorchTrainingState(lifetime, state.updates.increment(), statistics)

    @property
    def healthy(self) -> bool:
        """Whether every floating tensor in ``agent.state_dict()`` is finite.

        Nested dicts, lists and tuples are walked, so networks, optimizer
        moments and any other tensors the agent saves are all checked.
        """
        return all(
            bool(torch.isfinite(tensor).all()) for tensor in _float_tensors(self.agent.state_dict())
        )

    def save(self, path: str | Path, state: TorchTrainingState[StateT]) -> None:
        """Save the lifetime and the agent's state together, atomically."""
        path = Path(path).expanduser().resolve()
        if path.exists():
            raise FileExistsError(f"Checkpoint already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            staging.mkdir()
            carry = (state.lifetime, state.updates)
            checkpoint.save(staging / "lifetime", carry, metadata=self.metadata)
            torch.save(self.agent.state_dict(), staging / "agent.pt")
            staging.rename(path)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def restore(
        self, path: str | Path, template: TorchTrainingState[StateT]
    ) -> TorchTrainingState[StateT]:
        """Restore the lifetime into ``template`` and the agent in place.

        The agent file is loaded with ``weights_only=True``: tensors and plain
        Python values only, never arbitrary pickled objects.
        """
        path = Path(path).expanduser().resolve()
        lifetime, updates = checkpoint.restore(
            path / "lifetime",
            (template.lifetime, template.updates),
            expected_metadata=self.metadata,
        )
        self.agent.load_state_dict(torch.load(path / "agent.pt", weights_only=True))
        return TorchTrainingState(lifetime, updates, {})


class _ActorCritic(torch.nn.Module):
    def __init__(self, obs_dim: int, actions: int, hidden: int) -> None:
        super().__init__()

        def mlp(out: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(obs_dim, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, out),
            )

        self.actor = mlp(actions)
        self.critic = mlp(1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)


@dataclass(frozen=True)
class TorchPPOConfig:
    """Static PPO settings; the same defaults as the JAX ``PPOConfig``."""

    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 4


class TorchPPO:
    """Continuing PPO in PyTorch for discrete per-slot actions.

    No episode boundaries: advantages bootstrap through respawns, as in the
    JAX PPO. The advantage estimate skips the ticks a slot sits out, as the
    JAX ``compute_gae(active=...)`` does: the tick before an inactive stretch
    bootstraps from the next tick the slot acts, discounted across the stretch
    and including its rewards, so the value of a placeholder observation is
    never used inside a rollout. Invalid actions are masked out of the policy,
    and inactive slots are weighted out of the policy, value and entropy
    losses; the per-minibatch advantage normalisation (mean and standard
    deviation) still includes them.
    Sampling and minibatch shuffling use the agent's own seeded generator,
    which is part of the state dict, so a restored agent continues the same
    streams. ``__init__`` also calls ``torch.manual_seed(seed)`` to draw the
    initial weights, which reseeds PyTorch's global RNG.
    """

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        action_space: spaces.Space[Any],
        *,
        config: TorchPPOConfig | None = None,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(action_space, spaces.MultiDiscrete) or len(action_space.shape) != 1:
            raise ValueError("TorchPPO needs one discrete action per agent slot.")
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 2:
            raise ValueError("TorchPPO needs flat (agents, features) observations.")
        self.config = config or TorchPPOConfig()
        self.device = torch.device(device)
        self.actions = int(np.max(action_space.nvec))
        self.obs_dim = int(observation_space.shape[1])
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.manual_seed(seed)  # initial weights
        self.network = _ActorCritic(self.obs_dim, self.actions, self.config.hidden_size).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate, eps=1e-5
        )
        self.seed = seed

    @property
    def configuration(self) -> dict[str, object]:
        """Algorithm name, action/observation sizes and every config field."""
        return {
            "algorithm": "torch-ppo",
            "actions": self.actions,
            "observation_features": self.obs_dim,
            **{name: getattr(self.config, name) for name in self.config.__dataclass_fields__},
        }

    def _distribution(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        logits, value = self.network(obs.float())
        logits = torch.where(mask, logits, torch.finfo(logits.dtype).min)
        return torch.distributions.Categorical(logits=logits), value

    def act(
        self, obs: torch.Tensor, info: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample a valid action per slot; extras are its log-probability and value."""
        distribution, value = self._distribution(obs, info["action_mask"][..., : self.actions])
        probabilities = distribution.probs
        action = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(-1)
        return action, {"log_prob": distribution.log_prob(action), "value": value}

    def check_rollout(self, rollout_steps: int, num_agents: int) -> None:
        """Refuse a rollout that does not split into minibatches of at least two samples."""
        samples, minibatches = rollout_steps * num_agents, self.config.minibatches
        detail = f"rollout_steps={rollout_steps} x {num_agents} agents"
        if samples % minibatches:
            raise ValueError(
                f"Rollout samples ({samples}) must divide evenly into {minibatches} minibatches "
                f"({detail}); change rollout_steps or TorchPPOConfig.minibatches."
            )
        if samples // minibatches < 2:
            raise ValueError(
                f"Rollout samples ({samples}) split into {minibatches} minibatches leave "
                f"{samples // minibatches} sample per minibatch ({detail}); advantage "
                "normalisation needs at least 2. Raise rollout_steps or lower "
                "TorchPPOConfig.minibatches."
            )

    def advantages(self, rollout: TorchRollout) -> tuple[torch.Tensor, torch.Tensor]:
        """Continuing GAE advantages and value targets, both (time, agent slots).

        An inactive tick's value is replaced by its own one-step bootstrap, so
        an active tick never bootstraps from a placeholder observation's
        value. Entries on inactive ticks are not meaningful. A slot that is
        inactive when the rollout ends still bootstraps from the critic's
        value of ``last_obs``.
        """
        config = self.config
        with torch.no_grad():
            _, last_value = self.network(rollout.last_obs.float())
            values = rollout.extras["value"]
            advantages = torch.zeros_like(rollout.reward)
            running = torch.zeros_like(last_value)
            following = last_value
            active = rollout.active.bool()
            for step in reversed(range(rollout.reward.shape[0])):
                bootstrap = rollout.reward[step] + config.gamma * following
                value = torch.where(active[step], values[step], bootstrap)
                running = bootstrap - value + config.gamma * config.gae_lambda * running
                advantages[step] = running
                following = value
            return advantages, advantages + values

    def update(self, rollout: TorchRollout) -> dict[str, float]:
        """Continuing GAE, then clipped PPO epochs; returns minibatch-averaged losses."""
        config = self.config
        advantages, returns = self.advantages(rollout)
        flat = {
            "obs": rollout.obs.reshape(-1, self.obs_dim),
            "action": rollout.action.reshape(-1),
            "log_prob": rollout.extras["log_prob"].reshape(-1),
            "advantage": advantages.reshape(-1),
            "return": returns.reshape(-1),
            "mask": rollout.action_mask[..., : self.actions].reshape(-1, self.actions),
            "active": rollout.active.reshape(-1).float(),
        }
        size = flat["action"].shape[0]
        # A one-sample minibatch has no advantage spread to normalise by, which
        # turns the update into NaN.
        self.check_rollout(rollout.reward.shape[0], rollout.reward.shape[1])
        batch = size // config.minibatches
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        steps = 0
        for _ in range(config.epochs):
            order = torch.randperm(size, generator=self.generator, device=self.device)
            for start in range(0, size, batch):
                index = order[start : start + batch]
                weight = flat["active"][index]
                count = weight.sum().clamp(min=1.0)
                distribution, value = self._distribution(flat["obs"][index], flat["mask"][index])
                log_prob = distribution.log_prob(flat["action"][index])
                ratio = (log_prob - flat["log_prob"][index]).exp()
                advantage = flat["advantage"][index]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                clipped = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                policy = (
                    torch.max(-advantage * ratio, -advantage * clipped) * weight
                ).sum() / count
                value_loss = (0.5 * (value - flat["return"][index]) ** 2 * weight).sum() / count
                entropy = (distribution.entropy() * weight).sum() / count
                loss = (
                    policy
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), config.max_grad_norm)
                self.optimizer.step()
                for name, item in (
                    ("loss", loss),
                    ("policy_loss", policy),
                    ("value_loss", value_loss),
                    ("entropy", entropy),
                ):
                    totals[name] += float(item.detach())
                steps += 1
        return {name: total / steps for name, total in totals.items()}

    def state_dict(self) -> dict[str, Any]:
        """Network, optimizer and sampling-generator state for an exact resume."""
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.network.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
