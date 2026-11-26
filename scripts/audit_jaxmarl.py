"""Audit which JaxMARL environments support dynamics drift by attribute injection.

JaxMARL environments keep their dynamics on the environment object (for
example MPE's ``dt``, ``damping``, ``mass``), not in a parameter argument.
Perpetua injects drifted values by binding traced arrays onto a shallow copy of
the environment inside a jitted step. That works only where an attribute is
used in traced arithmetic; if it drives Python control flow, tracing fails.

For every registered environment and every float attribute this script
measures two things on a short jitted rollout:

* **faithful** - injecting the attribute's own default value reproduces the
  uninjected trajectory exactly;
* **effective** - injecting a perturbed value changes the trajectory.

An attribute is *driftable* when both hold, *inert* when injection is faithful
but no effect was observed in this short constant-action rollout (it may still
matter elsewhere, for example contact forces only act on collisions), and
*broken* when injection fails to trace or changes the default trajectory.
Run with:

    uv run --extra jaxmarl python scripts/audit_jaxmarl.py

Not a test: results track the installed JaxMARL version.
"""

# JaxMARL ships no type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportUnknownParameterType=false

import argparse
import copy
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np

warnings.filterwarnings("ignore")

TICKS = 12


def float_attributes(env: Any) -> dict[str, Any]:
    """Float scalars and float arrays held on the environment object."""
    found: dict[str, Any] = {}
    for name, value in vars(env).items():
        if name.startswith("_"):
            continue
        if (
            isinstance(value, float)
            or hasattr(value, "dtype")
            and jnp.issubdtype(jnp.asarray(value).dtype, jnp.floating)
        ):
            found[name] = value
    return found


def rollout(env: Any, overrides: dict[str, jax.Array] | None) -> np.ndarray:
    """A jitted rollout with constant actions, optionally injecting attributes."""
    key = jax.random.key(0)
    obs, state = env.reset(key)
    agents = env.agents

    def actions_for(step: int) -> dict[str, Any]:
        result = {}
        for agent in agents:
            space = env.action_space(agent)
            if hasattr(space, "n"):
                result[agent] = jnp.asarray(step % int(space.n))
            else:
                result[agent] = jnp.zeros(space.shape, dtype=jnp.float32)
        return result

    def step(state: Any, values: dict[str, jax.Array], step_key: jax.Array, actions: Any) -> Any:
        target = copy.copy(env)
        for name, value in values.items():
            setattr(target, name, value)
        return target.step(step_key, state, actions)

    compiled = jax.jit(step)
    values = overrides or {}
    trace = []
    for tick in range(TICKS):
        obs, state, reward, done, _ = compiled(
            state, values, jax.random.key(tick + 1), actions_for(tick)
        )
        trace.append(np.concatenate([np.ravel(np.asarray(obs[a])) for a in agents]))
        trace.append(np.asarray([float(reward[a]) for a in agents]))
    return np.concatenate(trace)


def audit(name: str) -> tuple[str, dict[str, list[str]], str]:
    """Return a status, attributes by verdict, and a note on failure."""
    verdicts: dict[str, list[str]] = {"driftable": [], "inert": [], "broken": []}
    try:
        env = jaxmarl.make(name)
    except Exception as error:  # noqa: BLE001 - recorded, not raised
        return "build-failed", verdicts, type(error).__name__
    try:
        baseline = rollout(env, None)
    except Exception as error:  # noqa: BLE001
        return "rollout-failed", verdicts, type(error).__name__
    for attribute, value in float_attributes(env).items():
        default = jnp.asarray(value, dtype=jnp.float32)
        try:
            faithful = np.array_equal(rollout(env, {attribute: default}), baseline)
            perturbed = rollout(env, {attribute: default * 1.5 + 0.1})
        except Exception:  # noqa: BLE001 - control flow on the attribute
            verdicts["broken"].append(attribute)
            continue
        if not faithful:
            verdicts["broken"].append(attribute)
        elif np.allclose(perturbed, baseline):
            verdicts["inert"].append(attribute)
        else:
            verdicts["driftable"].append(attribute)
    return "ok", verdicts, ""


def main() -> None:
    print(f"{'environment':34} {'status':15} {'drift':>5} {'inert':>5} {'broke':>5}  driftable")
    for name in jaxmarl.registered_envs:
        status, verdicts, note = audit(name)
        if status != "ok":
            print(f"{name:34} {status:15} {note}")
            continue
        counts = " ".join(f"{len(verdicts[kind]):>5}" for kind in ("driftable", "inert", "broken"))
        print(f"{name:34} {status:15} {counts}  {', '.join(verdicts['driftable'][:5])}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Audit which JaxMARL environment attributes support dynamics drift "
        "(takes minutes). Takes no options.",
    ).parse_args()
    main()
