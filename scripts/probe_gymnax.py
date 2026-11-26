# gymnax ships no type stubs, so its lazily-resolved generics stay loose.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportArgumentType=false

"""Probe every registered gymnax environment for the parameter adapter.

Prints one row per environment: observation shape, action cardinality, the
number of parameter fields, how many are float-valued (drift candidates), how
many are precomputed inputs (fields another field's default is computed from,
which the adapter refuses to drift because gymnax would leave the derived
field stale), whether the parameters carry a time limit, and whether the state
carries a clock. Run with:

    uv run --extra interop python scripts/probe_gymnax.py

Not a test: the table depends on the installed gymnax version.
"""

import argparse
import dataclasses
from typing import Any

import gymnax
import jax

from perpetua.adapters.gymnax_params import (
    _TIME_LIMIT,  # pyright: ignore[reportPrivateUsage]
    _derived_inputs,  # pyright: ignore[reportPrivateUsage]
)


def float_fields(params: Any) -> list[str]:
    """The fields ``GymnaxParameters`` treats as float-valued: floats other than the time limit."""
    return [
        field.name
        for field in dataclasses.fields(params)
        if isinstance(getattr(params, field.name), float) and field.name != _TIME_LIMIT
    ]


def main() -> None:
    print(
        f"{'env':26} {'obs':14} {'act':10} {'params':>6} {'float':>6} "
        f"{'inputs':>8} {'limit':>6} {'clock':>6}"
    )
    for name in gymnax.registered_envs:
        try:
            env, params = gymnax.make(name)
            fields = {f.name: getattr(params, f.name) for f in dataclasses.fields(params)}
            floats = float_fields(params)
            derived = sorted(_derived_inputs(type(params), list(fields)))
            # Derived field detection: does the value change when another float changes?
            obs_key = jax.random.key(0)
            _, state = env.reset(obs_key, params)
            try:
                action_space = env.action_space(params)
                act = getattr(action_space, "n", None) or getattr(action_space, "shape", "")
            except Exception:  # noqa: BLE001 - a probe reports what it can
                act = "?"
            obs_space = env.observation_space(params)
            for _ in range(4):
                obs, state, reward, done, info = env.step(jax.random.key(1), state, 0, params)
                if bool(done):
                    break
            print(
                f"{name:26} {str(obs_space.shape):14} {str(act):10} {len(fields):>6} "
                f"{len(floats):>6} {len(derived):>8} "
                f"{str('max_steps_in_episode' in fields):>6} "
                f"{str(hasattr(state, 'time')):>6}"
            )
        except Exception as error:  # noqa: BLE001 - a probe reports every failure
            print(f"{name:26} ERROR {type(error).__name__}: {str(error)[:40]}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Print one row per registered gymnax environment: observation shape, "
        "action cardinality, parameter counts, time limit and clock. Takes no options."
    ).parse_args()
    main()
