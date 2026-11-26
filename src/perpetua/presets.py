"""Ready-made continual-environment presets.

Presets are convenience constructors, not new environments: they compose the
existing adapters and the task-switch combinator with a drift schedule. Each
records ``metadata["configuration"]["preset"]``: its name, evaluation horizon
and a declared tuning protocol (k-percent, see :mod:`perpetua.tuning`), so the
protocol is part of the benchmark specification; read it back with
``pp.tuning.declared(env)``. ALE is
GPL-2.0, so Atari presets are available only when EnvPool is installed (an
optional dependency, never vendored here).
"""

# Optional benchmark packages ship incomplete type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from collections.abc import Iterable, Sequence
from contextlib import suppress
from typing import Any, cast

import perpetua as pp
from perpetua.core import as_int
from perpetua.drift import Cyclic
from perpetua.tasks import OnSwitch
from perpetua.tuning import TuningProtocol

__all__ = [
    "DEFAULT_TUNING",
    "switching_atari",
    "switching_minatar",
    "meal_sequence",
    "CW10",
    "CW20",
    "continual_world",
    "cw10",
    "cw20",
    "coom_sequence",
    "morpheus_regimes",
]

#: The tuning protocol every preset declares (no preset takes a tuning
#: argument): tune on the first 10% of the horizon with 3 seeds, select by IQM,
#: evaluate the winner on 10 fresh seeds.
DEFAULT_TUNING = TuningProtocol(0.1, tuning_seeds=3, evaluation_seeds=10, selection="iqm")


def _with_preset(env: Any, name: str, *, horizon_ticks: int, **preset: Any) -> Any:
    configuration = env.metadata.setdefault("configuration", {})
    configuration["preset"] = {
        **configuration.get("preset", {}),
        "name": name,
        **preset,
        "horizon_ticks": horizon_ticks,
        "tuning": DEFAULT_TUNING.configuration,
    }
    return env


def _period(period: int) -> int:
    return as_int(period, "period must be a positive integer.", minimum=1)


def _switched(envs: Iterable[Any], period: int, **kwargs: Any) -> Any:
    """Build the tasks and switch between them, closing what was built if either fails."""
    built: list[Any] = []
    try:
        built.extend(envs)
        schedule = Cyclic([float(index) for index in range(len(built))], period)
        return pp.tasks.switch(cast(Any, built), schedule, **kwargs)
    except BaseException:
        for env in built:
            with suppress(Exception):
                getattr(env, "close", lambda: None)()
        raise


def switching_atari(
    games: Sequence[str],
    *,
    num_envs: int = 8,
    num_threads: int = 4,
    period: int = 100_000,
    on_switch: OnSwitch = "resume",
) -> Any:
    """Cycle Atari games inside one host-controlled lifetime.

    Uses ``full_action_space=True`` for every game, so all share 18 actions and
    the compiled action shape never changes. EnvPool is Linux-only here.
    ``on_switch`` is passed to :func:`perpetua.tasks.switch`; these are host
    lifts, so only ``"resume"`` is accepted and ``"restart"`` raises
    ``ValueError``.
    """
    if len(games) < 2:
        raise ValueError("switching_atari needs at least two games.")
    period = _period(period)
    from perpetua.adapters.host import EnvPoolAtari

    envs = (EnvPoolAtari(game, num_envs=num_envs, num_threads=num_threads) for game in games)
    env = _switched(envs, period, on_switch=on_switch)
    return _with_preset(
        env, "switching_atari", tasks=list(games), horizon_ticks=len(games) * period
    )


def switching_minatar(
    games: Sequence[str] = ("Breakout-MinAtar", "Asterix-MinAtar"),
    *,
    period: int = 100_000,
    on_switch: OnSwitch = "resume",
) -> Any:
    """Cycle MinAtar games inside one compiled lifetime.

    ``on_switch`` is passed to :func:`perpetua.tasks.switch`: ``"resume"``
    continues a re-entered game where it left off, ``"restart"`` re-initialises
    it on the tick it becomes active.
    """
    if len(games) < 2:
        raise ValueError("switching_minatar needs at least two games.")
    period = _period(period)
    from perpetua.adapters.gymnax_params import GymnaxParameters

    env = _switched((GymnaxParameters(game) for game in games), period, on_switch=on_switch)
    return _with_preset(
        env, "switching_minatar", tasks=list(games), horizon_ticks=len(games) * period
    )


def meal_sequence(
    *,
    sequence_length: int = 8,
    difficulty: str = "medium",
    num_agents: int = 2,
    seed: int = 0,
    period: int = 100_000,
    on_switch: OnSwitch = "resume",
) -> Any:
    """Cycle a reproducible MEAL Overcooked task sequence in one lifetime.

    ``on_switch`` is passed to :func:`perpetua.tasks.switch`: ``"resume"``
    continues a re-entered task where it left off, ``"restart"`` re-initialises
    it on the tick it becomes active.
    """
    sequence_length = as_int(sequence_length, "meal_sequence needs at least two tasks.", minimum=2)
    period = _period(period)
    import meal

    from perpetua.adapters.meal import MealOvercooked

    backends = meal.make_sequence(
        sequence_length=sequence_length,
        num_agents=num_agents,
        difficulty=difficulty,
        seed=seed,
    )
    envs = (
        MealOvercooked(backend=backend, difficulty=difficulty, num_agents=num_agents, seed=seed)
        for backend in backends
    )
    env = _switched(envs, period, on_switch=on_switch)
    return _with_preset(
        env,
        "meal_sequence",
        tasks=sequence_length,
        difficulty=difficulty,
        horizon_ticks=sequence_length * period,
    )


CW10 = (
    "hammer-v3",
    "push-wall-v3",
    "faucet-close-v3",
    "push-back-v3",
    "stick-pull-v3",
    "handle-press-side-v3",
    "push-v3",
    "shelf-place-v3",
    "window-close-v3",
    "peg-unplug-side-v3",
)
CW20 = CW10 * 2


def continual_world(
    tasks: Sequence[str] = CW10,
    *,
    seed: int = 0,
    period: int = 1_000_000,
    max_episode_steps: int = 200,
    compiled: bool = False,
    randomize_goals: bool = False,
) -> Any:
    """Cycle the published CW10/CW20 order on Farama Meta-World v3 tasks.

    The paper's v1 names are mapped to Farama's current v3 implementations;
    this is the published order, not a claim of numerical equivalence to the
    original v1 code. Tasks resume on return because host lifts cannot restart.
    ``compiled=True`` steps each simulator through the compiled bridge, so the
    whole lifetime (and a learner such as ``SACRunner``) runs as one program.
    The switched environment offers no ``success`` Oracle field in either mode;
    Meta-World's success flag is available only on a single ``MetaWorldLifetime``
    (the evaluation scripts build one per task).
    ``randomize_goals=True`` draws a new goal configuration every episode, as
    the Continual World paper does.

    Each entry of ``tasks`` gets its own simulator seeded ``seed + index``, so
    ``continual_world(CW20)`` builds 20 simulators and its second pass runs
    fresh, differently seeded copies of the ten tasks, whereas :func:`cw20`
    revisits the same ten simulators.
    """
    if len(tasks) < 2:
        raise ValueError("continual_world needs at least two tasks.")
    period = _period(period)
    from perpetua.adapters.host import MetaWorldLifetime

    envs = (
        MetaWorldLifetime(
            env_name=name,
            seed=seed + index,
            max_episode_steps=max_episode_steps,
            compiled=compiled,
            randomize_goals=randomize_goals,
        )
        for index, name in enumerate(tasks)
    )
    env = _switched(envs, period)
    return _with_preset(
        env,
        "continual_world",
        tasks=list(tasks),
        seed=seed,
        max_episode_steps=max_episode_steps,
        compiled=compiled,
        randomize_goals=randomize_goals,
        horizon_ticks=len(tasks) * period,
    )


def cw10(*, period: int = 1_000_000, **kwargs: Any) -> Any:
    """Continual World CW10: the ten published tasks, ``period`` ticks each."""
    period = _period(period)
    env = continual_world(CW10, period=period, **kwargs)
    return _with_preset(env, "cw10", tasks=list(CW10), repetitions=1, horizon_ticks=10 * period)


def cw20(*, period: int = 1_000_000, **kwargs: Any) -> Any:
    """Continual World CW20: CW10 twice, i.e. ``MetaWorld ∘ Switch(CW10, 2, period)``.

    A lifetime has no end, so this is the CW10 cycle with a 20-task evaluation
    horizon (``20 * period`` ticks) recorded in the metadata. Each task keeps one
    simulator, so the second pass revisits the same environments.
    """
    period = _period(period)
    env = continual_world(CW10, period=period, **kwargs)
    return _with_preset(env, "cw20", tasks=list(CW10), repetitions=2, horizon_ticks=20 * period)


def coom_sequence(
    sequence: str = "CO8",
    *,
    period: int = 200_000,
) -> Any:
    """Cycle a COOM sequence (``CO8``, ``CD8``, ...) in COOM's own task order.

    Needs the separate COOM runtime from ``python -m perpetua.adapters.coom_setup``.
    Each task is its own worker process; tasks resume on return because host
    lifts cannot restart.
    """
    period = _period(period)
    from perpetua.adapters.coom import COOMLifetime, sequence_tasks

    tasks = sequence_tasks(sequence)
    envs = (COOMLifetime(scenario=scenario, task=task) for scenario, task in tasks)
    env = _switched(envs, period)
    return _with_preset(env, "coom_sequence", sequence=sequence, horizon_ticks=len(tasks) * period)


def morpheus_regimes(*, period: int = 500, **kwargs: Any) -> Any:
    """Run the MORPHEUS paper's six regimes in one persistent world.

    Baseline, EDI spike, infrastructure degradation, financial pressure, workforce constraint,
    then a baseline revisit, each for ``period`` decisions (the evaluation suite uses 500). The
    switches are declared drift, so recovery tracking and ``pp.protocol`` see them. After the
    horizon the world stays in the baseline revisit rather than wrapping into another
    baseline block. Keyword arguments go to ``MorpheusLifetime`` (``mock=False`` for a live
    deployment).
    """
    from perpetua.adapters.morpheus import PAPER_REGIMES, MorpheusLifetime

    period = _period(period)
    last = len(PAPER_REGIMES) - 1
    schedule = pp.drift.Sequence(
        [
            (pp.drift.Cyclic([float(index) for index in range(last)], period), last * period),
            (pp.drift.Stationary(float(last)), 1),
        ]
    )
    drift = cast(Any, {"regime": schedule})  # DriftEnv accepts a per-field mapping
    env = MorpheusLifetime(regimes=PAPER_REGIMES, drift=drift, **kwargs)
    return _with_preset(
        env,
        "morpheus_regimes",
        regimes=[regime.regime_id for regime in PAPER_REGIMES],
        horizon_ticks=len(PAPER_REGIMES) * period,
    )
