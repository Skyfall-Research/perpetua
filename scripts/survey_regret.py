"""Survey where same-state regret can see anything.

For every registered environment that runs as a JAX world on this machine,
print ``pp.testing.reward_action_sensitivity``: the fraction of ticks, along
uniformly random lifetimes, on which the tick's reward differed between
actions stepped from the same state. ``0`` means regret against any reference
is exactly zero on the states visited. Host simulators and environments whose
runtime is not installed are listed as skipped. Run with:

    uv run --all-extras python scripts/survey_regret.py

Not a test: the fractions depend on the installed environment versions.
"""

import argparse
from typing import Any

import perpetua as pp
from perpetua.testing import reward_action_sensitivity


def survey(env_id: str, *, ticks: int, actions: int, seeds: list[int]) -> float | str:
    """The fraction pooled over ``seeds``, or the reason ``env_id`` was skipped."""
    try:
        env: Any = pp.make(env_id)
    except Exception as error:  # noqa: BLE001 - a survey reports what it can
        return f"skipped: {type(error).__name__}"
    try:
        if env.metadata.get("execution") == "host":
            return "skipped: host simulator"
        fractions = [
            reward_action_sensitivity(env, seed=seed, ticks=ticks, actions=actions)
            for seed in seeds
        ]
        return sum(fractions) / len(fractions)
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ticks", type=int, default=1000, help="ticks per seed")
    parser.add_argument("--actions", type=int, default=24, help="actions compared at each tick")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument(
        "--env", action="append", help="environment id (repeatable); default: all registered"
    )
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    print(f"| Environment | Ticks with an action-dependent reward ({len(seeds)} x {args.ticks}) |")
    print("|---|---|")
    for env_id in args.env or pp.environments():
        result = survey(env_id, ticks=args.ticks, actions=args.actions, seeds=seeds)
        shown = result if isinstance(result, str) else f"{result:.4f}"
        print(f"| `{env_id}` | {shown} |", flush=True)


if __name__ == "__main__":
    main()
