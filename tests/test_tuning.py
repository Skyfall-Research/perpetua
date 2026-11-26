"""k-percent tuning: budget, disjoint seeds, selection rule and honest accounting."""

# JAX typing surfaces are partially annotated.
# pyright: reportUnknownMemberType=false

from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner
from perpetua.agents.ppo import PPOHyperparameters


def test_protocol_spends_the_declared_budget_on_disjoint_seeds() -> None:
    calls: list[tuple[float, tuple[int, ...], int]] = []

    def score(candidate: float, seeds: Sequence[int], ticks: int) -> list[float]:
        calls.append((candidate, tuple(seeds), ticks))
        return [candidate + 0.01 * seed for seed in seeds]

    protocol = pp.tuning.TuningProtocol(0.1, tuning_seeds=2, evaluation_seeds=4, seed_offset=100)
    result = pp.tuning.tune(
        protocol, [0.5, 2.0, 1.0], score, lifetime_ticks=1_000, key=jax.random.key(0)
    )
    assert result.best == 2.0
    assert result.selected == 1
    assert [call[2] for call in calls] == [100, 100, 100, 1_000]
    assert calls[-1][:2] == (2.0, (100, 101, 102, 103))
    assert not set(calls[0][1]) & set(calls[-1][1])
    assert result.tuning_scores.shape == (3, 2)
    assert result.ticks_spent == {"tuning": 600, "evaluation": 4_000, "total": 4_600}
    assert result.protocol["protocol"] == "k-percent"
    assert result.protocol["fraction"] == 0.1
    low, high = float(result.intervals["iqm"].low), float(result.intervals["iqm"].high)
    assert low <= float(result.intervals["iqm"].point) <= high


def test_iqm_selection_ignores_one_lucky_seed() -> None:
    # Candidate 0 has one huge outlier; its mean wins, its IQM does not.
    table = {0: [0.0, 0.0, 0.0, 100.0], 1: [1.0, 1.0, 1.0, 1.0]}

    def score(candidate: int, seeds: Sequence[int], ticks: int) -> list[float]:
        del ticks
        return table[candidate][: len(seeds)] if len(seeds) == 4 else [0.0] * len(seeds)

    rules: tuple[tuple[Literal["mean", "iqm"], int], ...] = (("mean", 0), ("iqm", 1))
    for selection, winner in rules:
        protocol = pp.tuning.TuningProtocol(0.5, tuning_seeds=4, selection=selection)
        result = pp.tuning.tune(protocol, [0, 1], score, lifetime_ticks=10, key=jax.random.key(0))
        assert result.selected == winner


def test_iqm_selection_differs_from_the_mean_with_three_seeds() -> None:
    # The preset default (3 tuning seeds, "iqm") must not silently be the mean.
    table = {0: [0.0, 0.0, 9.0], 1: [2.0, 2.0, 2.0]}

    def score(candidate: int, seeds: Sequence[int], ticks: int) -> list[float]:
        del ticks
        return table[candidate] if len(seeds) == 3 else [0.0] * len(seeds)

    rules: tuple[tuple[Literal["mean", "iqm"], int], ...] = (("mean", 0), ("iqm", 1))
    for selection, winner in rules:
        protocol = pp.tuning.TuningProtocol(0.5, tuning_seeds=3, selection=selection)
        result = pp.tuning.tune(protocol, [0, 1], score, lifetime_ticks=10, key=jax.random.key(0))
        assert result.selected == winner


def test_protocol_validation_and_bad_scores() -> None:
    for kwargs in ({"fraction": 0.0}, {"fraction": 1.5}, {"fraction": 0.5, "tuning_seeds": 0}):
        with pytest.raises(ValueError, match=r"must be"):
            pp.tuning.TuningProtocol(**kwargs)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="disjoint"):
        pp.tuning.TuningProtocol(0.5, tuning_seeds=5, seed_offset=2)
    protocol = pp.tuning.TuningProtocol(0.5)
    with pytest.raises(ValueError, match="one finite value per seed"):
        pp.tuning.tune(
            protocol,
            [1],
            lambda c, s, t: [np.nan] * len(s),
            lifetime_ticks=4,
            key=jax.random.key(0),
        )
    with pytest.raises(ValueError, match="empty"):
        pp.tuning.tune(protocol, [], lambda c, s, t: [0.0], lifetime_ticks=4, key=jax.random.key(0))


def test_k_percent_tunes_a_ppo_learning_rate_on_a_lifetime() -> None:
    env = pp.make("ContinualGrid-v0", size=3, num_agents=4)
    config = PPOConfig()
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    rollout = config.rollout_steps

    def score(learning_rate: float, seeds: Sequence[int], ticks: int) -> list[float]:
        hyperparameters = PPOHyperparameters.from_config(config)._replace(
            learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32)
        )
        scores: list[float] = []
        for seed in seeds:
            state = trainer.init(seed=seed, hyperparameters=hyperparameters)
            for _ in range(ticks // rollout):
                state = trainer(state)
            scores.append(float(np.asarray(state.lifetime.metrics.mean).mean()))
        return scores

    protocol = pp.tuning.TuningProtocol(0.25, tuning_seeds=2, evaluation_seeds=2)
    result = pp.tuning.tune(
        protocol, [1e-7, 3e-3], score, lifetime_ticks=64 * rollout, key=jax.random.key(0)
    )
    # A near-zero learning rate barely moves its initial policy.
    assert result.best == 3e-3
    assert result.tuning_ticks == 16 * rollout
    assert float(result.intervals["mean"].point) > float(result.tuning_scores[0].mean())


def test_presets_declare_their_tuning_protocol() -> None:
    env = pp.presets.switching_minatar(period=10)
    protocol = pp.tuning.declared(env)
    assert protocol == pp.presets.DEFAULT_TUNING
    assert protocol.tuning_ticks(env.metadata["configuration"]["preset"]["horizon_ticks"]) == 2
    regimes = pp.presets.morpheus_regimes(period=5)
    assert pp.tuning.declared(regimes).selection == "iqm"
    assert regimes.metadata["configuration"]["preset"]["horizon_ticks"] == 30
    with pytest.raises(ValueError, match="no tuning protocol"):
        pp.tuning.declared(pp.make("ContinualGrid-v0"))


def test_tuning_ticks_do_not_overshoot_through_float_rounding() -> None:
    assert pp.tuning.TuningProtocol(0.07).tuning_ticks(100) == 7
    assert pp.tuning.TuningProtocol(0.07).tuning_ticks(10_000_000) == 700_000
    assert pp.tuning.TuningProtocol(0.14).tuning_ticks(50) == 7
    assert pp.tuning.TuningProtocol(0.1).tuning_ticks(15) == 2  # a true fraction still rounds up
