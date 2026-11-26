"""Generate the environment pages: one per registered id, one per family, an overview.

Pages combine live metadata (spaces, drift fields, Oracle fields, constructor
arguments) with the prose in ``scripts/env_content.py``. Run with:

    uv run python scripts/generate_env_docs.py

Environments that cannot be constructed on this host (EnvPool off Linux, COOM or
AgarCL without their runtimes) keep their committed page.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
from collections.abc import Sized
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np

import perpetua as pp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "environments"
FAMILY_OUT = OUT / "families"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_content import (  # noqa: E402
    FAMILIES,
    content_for,
    family_of,
    install_of,
    title_of,
    upstream_of,
    version_of,
)

GENERIC_ORACLE = ("phase", "drift_parameters", "drift_changed", "task_id", "boundary")


FIGURES = ROOT / "docs" / "figures" / "environments"
# Captions for the frames drawn by scripts/plot_docs_figures.py.
_TICKS = "One frame after a few random ticks: "
FRAMES = {
    "ContinualGrid-v0": _TICKS + "the agent in blue and the goal in orange.",
    "DoorGrid-v0": _TICKS + "the agent in blue and the goal in orange. Arrows show the four "
    "moves: green is available, red is masked by a closed door or a wall.",
    "ForagerGrid-v0": _TICKS + "the agent in blue and the goal in orange.",
    "CooperativeGrid-v0": _TICKS + "the two agents in blue and each agent's target in orange.",
    "PartnerGrid-v0": _TICKS + "the learner in blue, the scripted partner in grey and each "
    "agent's target in orange.",
    "OpponentGrid-v0": _TICKS + "the learner in blue, the scripted opponent in grey and the "
    "goal in orange.",
    **dict.fromkeys(
        (
            "gymnax/Asterix-MinAtar",
            "gymnax/Breakout-MinAtar",
            "gymnax/Freeway-MinAtar",
            "gymnax/SpaceInvaders-MinAtar",
            "gymnax/Pong-misc",
        ),
        "The observation after 40 random ticks, with one colour per channel.",
    ),
    **dict.fromkeys(
        ("gymnax/Catch-bsuite", "gymnax/DeepSea-bsuite", "gymnax/MNISTBandit-bsuite"),
        "The observation after 40 random ticks.",
    ),
    **dict.fromkeys(
        ("gymnax/FourRooms-misc", "gymnax/MetaMaze-misc"),
        "The map after a few random ticks: walls in grey, the agent in blue and the goal in "
        "orange.",
    ),
    **dict.fromkeys(
        (
            "gymnax/CartPole-v1",
            "gymnax/Pendulum-v1",
            "gymnax/Acrobot-v1",
            "gymnax/MountainCar-v0",
            "gymnax/MountainCarContinuous-v0",
            "mjx/Pendulum-v0",
            "mjx/CartPole-v0",
        ),
        "A drawing of the observation after 12 random ticks (positions and angles; not a "
        "simulator render).",
    ),
    "gymnax/Reacher-misc": "A drawing of the state after 12 random ticks: the two links in "
    "blue and the goal in orange.",
    "gymnax/Swimmer-misc": "A drawing of the state after 12 random ticks: the swimmer in blue, "
    "the urchins in grey and the goal in orange.",
    "jaxmarl/": "A drawing of the state after 12 random ticks, with each entity at its own "
    "radius: agents in blue, adversaries in orange and landmarks in grey.",
    "metaworld/MT1-v0": "A MuJoCo render of the default task, `hammer-v3`, after 12 random "
    "actions. It is rendered from the upstream environment; the adapter runs without a "
    "renderer.",
    "meal/overcooked-v0": "The kitchen after 6 random ticks: the two cooks in blue, walls "
    "and counters in grey, pots in black, the onion pile in yellow, the plate pile in purple "
    "and the delivery square in green. The layout is generated, so yours may differ.",
    "BeanWorld-v0": "A 33-by-33 part of the world around the agent (black) after a few random "
    "ticks. Jelly beans are blue, bananas yellow and onions red; the box is the agent's 7-by-7 "
    "view.",
}


def frame(env_id: str) -> list[str]:
    """The frame and its caption, for environments that have one drawn."""
    names = [f"{slug(env_id)}{suffix}" for suffix in (".svg", ".png")]
    name = next((name for name in names if (FIGURES / name).exists()), None)
    caption = FRAMES.get(env_id) or FRAMES.get(env_id.split("/")[0] + "/")
    if caption is None or name is None:
        return []
    return [
        "",
        f"![A frame of {title_of(env_id)}](../figures/environments/{name})",
        "",
        f"*{caption}*",
    ]


def slug(env_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", env_id.lower()).strip("-")


def driftable_fields(env: Any, configuration: dict[str, Any]) -> list[str]:
    """The parameter names ``drift=`` accepts, not merely the active schedule."""
    world = getattr(env, "env", env)
    allowed = getattr(world, "drift_fields", None)
    if allowed is not None:  # the world restricts drift to these parameters
        return sorted(allowed)
    names = getattr(env, "_names", None)
    if names:
        return sorted(names)
    fields = set(configuration.get("dynamics_fields", ()))
    primary = configuration.get("drift_parameter")
    if isinstance(primary, str):
        fields.add(primary)
    for name in ("default_drift_field", "drift_fields", "dynamics_fields"):
        value = getattr(world, name, ())
        if isinstance(value, str):
            fields.add(value)
        elif value is not None:
            fields.update(value)
    return sorted(fields)


def bare_drift_field(env: Any, fields: list[str]) -> str | None:
    """The parameter a bare ``drift=`` process drives, as the environment resolves it."""
    if not fields:
        return None
    resolve = getattr(env, "_bare_field", None)
    if resolve is None:
        return getattr(getattr(env, "env", env), "default_drift_field", None)
    try:
        return resolve()
    except ValueError:  # several driftable parameters and no declared default
        return None


def backend_version(env_id: str, configuration: dict[str, Any]) -> str:
    if env_id.startswith("gymnax/"):
        return f"gymnax-{version('gymnax')}"
    backend = configuration.get("backend", "native JAX")
    if not isinstance(backend, dict):
        return str(backend)
    name = str(backend.get("backend", "host"))
    if "coom_commit" in backend:
        return f"coom-{str(backend['coom_commit'])[:12]}"
    if "metaworld" in backend:
        name = "metaworld"
    package = "metaworld" if "metaworld" in backend else "envpool" if "envpool" in name else ""
    if package:
        try:
            return f"{name}-{version(package)}"
        except PackageNotFoundError:
            pass
    return name


def _number(value: float) -> str:
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:g}"


def values(value: Any) -> str:
    """PettingZoo-style value range: ``[0, 1]`` for boxes, ``Discrete(5)`` for slots."""
    if hasattr(value, "nvec"):
        counts = sorted({int(n) for n in np.asarray(value.nvec).reshape(-1)})
        return " to ".join(f"`Discrete({n})`" for n in counts) + " per agent"
    low, high = float(np.min(value.low)), float(np.max(value.high))
    opening = "(" if np.isinf(low) else "["
    closing = ")" if np.isinf(high) else "]"
    return f"`{opening}{_number(low)}, {_number(high)}{closing}`"


GENERIC_MEANING = {
    "phase": (
        "the drift phase: an integer that changes at every declared switch "
        "(for `Cyclic`, the index of the current value)"
    ),
    "drift_parameters": "the current value of the default drifting parameter",
    "drift_changed": "deprecated alias of `boundary`",
    "task_id": "the active task: the task index under a task switch, the regime "
    "under MORPHEUS, otherwise the phase of the default drifting parameter",
    "boundary": "whether a declared switch happened on this tick (the name "
    "task-sequence protocols use, for example SAC's `task_boundary_reset`)",
}
NO_DRIFT_MEANING = {
    "phase": "always 0: nothing in this environment drifts",
    "drift_parameters": "the value of the first parameter field in alphabetical order; "
    "nothing drifts here, so it carries no information",
    "task_id": "the task index under a task switch, otherwise always 0",
}
OWN_MEANING = {
    "success": "Meta-World's task-success flag for the last step",
    "regime_index": "the index of the MORPHEUS regime declared on this tick (the world "
    "takes it before the next transition)",
    "chaos_active": "whether the declared MORPHEUS regime differs from the first regime "
    "(false for the first regime and for a revisit of it)",
    "invalid_action": "whether the live service rejected the last action's call (the "
    "seeded mock never rejects one)",
}


def oracle_summary(fields: tuple[str, ...]) -> str:
    """The short header-table form: the names most protocols ask for."""
    own = [
        name
        for name in fields
        if name not in GENERIC_ORACLE and not name.endswith(("_parameters", "_phase", "_changed"))
    ]
    shown = ["`task_id`", "`boundary`", "drift values", *(f"`{name}`" for name in own)]
    return ", ".join(shown) + " ([all](#privileged-fields))"


def privileged_section(
    fields: tuple[str, ...], driftable: list[str], keeps_limit: bool = False
) -> list[str]:
    """Every Oracle field with its meaning, grouped as the generic, per-parameter and own.

    Parameters that ``drift=`` rejects are reported as constants in one row, not as drifting
    parameters; ``max_steps_in_episode`` is the backend's time limit.
    """
    lines = [
        "",
        "## Privileged fields",
        "",
        "Hidden from the agent; expose any of them with "
        '`pp.Oracle(env, expose=["task_id", ...])`, which records the choice in '
        '`metadata["oracle_exposure"]`.',
        "",
        "| Field | Meaning |",
        "|---|---|",
    ]
    for name in GENERIC_ORACLE:
        if name in fields:
            meaning = GENERIC_MEANING[name] if driftable else NO_DRIFT_MEANING.get(name)
            lines.append(f"| `{name}` | {meaning or GENERIC_MEANING[name]} |")
    parameters = sorted(
        {name.removesuffix("_parameters") for name in fields if name.endswith("_parameters")}
        - {"drift"}
    )
    constants = [name for name in parameters if name not in driftable]
    for parameter in parameters:
        if parameter in constants:
            continue
        lines.append(
            f"| `{parameter}_parameters`, `{parameter}_phase`, `{parameter}_changed` | "
            f"the current value, phase and switch flag of `{parameter}` |"
        )
    limit = "max_steps_in_episode"
    others = [name for name in constants if name != limit]
    if others:
        names = ", ".join(f"`{name}`" for name in others)
        lines.append(
            f"| `<name>_parameters`, `<name>_phase`, `<name>_changed` for {names} | "
            "constants: the backend's value of each field, phase 0, never changed; "
            "`drift=` rejects these fields |"
        )
    if limit in constants:
        meaning = (
            "the backend's time limit, which this environment keeps (it ends the "
            "episode, as a respawn)"
            if keeps_limit
            else "the backend's original time limit, as a constant; the adapter removes "
            "the limit, so it is not enforced"
        )
        lines.append(f"| `{limit}_parameters`, `{limit}_phase`, `{limit}_changed` | {meaning} |")
    for name in fields:
        if name in GENERIC_ORACLE or name.endswith(("_parameters", "_phase", "_changed")):
            continue
        lines.append(f"| `{name}` | {OWN_MEANING.get(name, 'environment-specific diagnostic')} |")
    return lines


def _named(reference: str) -> Any:
    module, name = reference.split(":")
    return getattr(importlib.import_module(module), name)


_SCRIPTED_ARGS = ("num_scripted", "scripted_slots", "scripted_behavior")


def arguments(env_id: str, solo: bool = False) -> list[tuple[str, str]]:
    """(name, default) for the keyword arguments ``pp.make`` forwards.

    Defaults are the constructor's, overlaid with the id's registered preset.
    ``solo`` leaves out the scripted-slot arguments, which a task with one
    controllable agent rejects.
    """
    content = content_for(env_id, solo)
    target = _named(content.constructor)
    preset: dict[str, Any] = dict(_named(content.preset)) if content.preset else {}
    hidden = ("self", "env_id", "obs_type", *content.extra_args, *(_SCRIPTED_ARGS if solo else ()))
    rows: list[tuple[str, str]] = []
    for parameter in inspect.signature(target.__init__).parameters.values():
        value = preset.pop(parameter.name, parameter.default)
        if parameter.name in hidden:
            continue
        if parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
            continue
        default = "required" if value is parameter.empty else repr(value)
        if len(default) > _LONG_DEFAULT:
            default = f"{len(cast('Sized', value))} built-in entries (see the source)"
        rows.append((parameter.name, default))
    if preset:
        raise ValueError(f"{env_id}: {content.preset} names unknown arguments {sorted(preset)}.")
    if not any(row[0] == "drift" for row in rows):
        rows.append(("drift", "None"))
    if not solo and any(row[0] == "num_scripted" for row in rows):
        rows.append(("scripted_drift", "None"))
    rows += list(content.make_args)
    return rows


_LONG_DEFAULT = 200


def _literal(value: np.ndarray) -> str:
    """A drift phase written the way ``Cyclic`` expects for this parameter's shape."""
    if value.ndim == 0:
        return f"{float(value):g}"
    items = ", ".join(f"{float(item):g}" for item in value.reshape(-1))
    return f"({items},)" if value.size == 1 else f"({items})"


def drift_example(env: Any, env_id: str) -> list[str]:
    """Usage lines that drift the default parameter between two valid values."""
    world = getattr(env, "env", None)
    field = getattr(world, "default_drift_field", None)
    names = getattr(env, "_names", None)
    if world is None or field is None or not names or field not in names:
        return []
    drift = cast("dict[str, Any]", env.metadata.get("configuration", {})).get("drift", {})
    if isinstance(drift, dict) and drift.get("type", "stationary") != "stationary":
        return []  # the default drift is already non-stationary (ForagerGrid's mass)
    value = np.asarray(getattr(world.default_params, field), dtype=np.float32)
    if value.ndim > 1:
        return []
    if env_id == "ForagerGrid-v0":
        comment = (
            f"# Make it non-stationary: each collect adds 4 to `{field}`, which then decays "
            "by 10% per tick."
        )
        process = "pp.drift.Accumulate(gain=4.0, decay=0.9, low=1.0, high=8.0)"
        return _drifting(env, env_id, field, comment, process)
    if field in {"goal"}:
        other = np.zeros_like(value)
    elif field == "doors":
        other = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    elif field == "item_rewards":
        other = -value
    elif field == "regime":
        other = value + 1
    else:
        other = np.where(value != 0, value * 0.5, 0.1).astype(np.float32)
    comment = (
        f"# Make it non-stationary: `{field}` alternates between two values every 1,000 ticks."
    )
    process = f"pp.drift.Cyclic([{_literal(value)}, {_literal(other)}], period=1000)"
    return _drifting(env, env_id, field, comment, process)


def _drifting(env: Any, env_id: str, field: str, comment: str, process: str) -> list[str]:
    # The trailing comma keeps `ruff format` from re-wrapping the committed page.
    lines = [
        "",
        comment,
        "drifting = pp.make(",
        f'    "{env_id}",',
        f'    drift={{"{field}": {process}}},',
        ")",
        "state, obs, info = drifting.init(jax.random.key(0))",
    ]
    if env.metadata.get("execution") == "host":
        lines.append("drifting.close()")
    return lines


def usage(env: Any, env_id: str) -> list[str]:
    discrete = hasattr(env.action_space, "nvec")
    action = (
        "pp.policies.random(obs, info, jax.random.key(100 + tick))"
        if discrete
        else "jnp.zeros(env.action_space.shape, dtype=jnp.float32)"
    )
    host = env.metadata.get("execution") == "host"
    body = [
        "import jax",
        *([] if discrete else ["import jax.numpy as jnp"]),
        "import perpetua as pp",
        "",
        f'env = pp.make("{env_id}")',
        "state, obs, info = env.init(jax.random.key(0))",
        "for tick in range(4):",
        f"    action = {action}",
        "    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))",
        'print(obs.shape, reward, info["t"].to_int())',
        *drift_example(env, env_id),
    ]
    if host:
        body.append("env.close()  # host simulators hold an external process or library")
    return ["```python", *body, "```"]


def agents(env: Any, execution: str) -> str:
    """Slot count, saying how many are scripted when some are not learner-controlled."""
    if execution == "host":
        return f"`{env.num_agents}`"
    _, _, info = env.init(jax.random.key(0))
    learners = int(np.asarray(info["active_mask"]).sum())
    scripted = env.num_agents - learners
    if scripted == 0:
        return f"`{env.num_agents}`"
    return f"`{env.num_agents}` ({learners} learner-controlled, {scripted} scripted)"


def single_controllable(env: Any) -> bool:
    """Whether an MPE task has one controllable agent, so no slot can be scripted."""
    reserved = getattr(getattr(env, "env", env), "backend_scripted_slots", None)
    return reserved is not None and env.num_agents - len(reserved) < 2


def describe(env: Any, env_id: str) -> str:
    """One environment page, laid out like PettingZoo's (metadata, description,
    definition, arguments, usage, version history, API)."""
    solo = single_controllable(env)
    content = content_for(env_id, solo)
    family = FAMILIES[family_of(env_id)]
    metadata = cast("dict[str, Any]", env.metadata)
    configuration = cast("dict[str, Any]", metadata.get("configuration", {}))
    drift = configuration.get("drift", {})
    drift_type = drift.get("type", "stationary") if isinstance(drift, dict) else "?"
    fields = driftable_fields(env, configuration)
    execution = metadata.get("execution", "jax")
    checkpointable = metadata.get("checkpointable", execution != "host")
    upstream, defines = upstream_of(env_id)
    discrete = hasattr(env.action_space, "nvec")
    rows = [
        ("Creation", f'`pp.make("{env_id}")`'),
        ("Install", f"`{install_of(env_id)}`"),
        ("Actions", "Discrete" if discrete else "Continuous"),
        ("Agents", agents(env, execution)),
        ("Action Shape", f"`{tuple(env.action_space.shape)}`"),
        ("Action Values", values(env.action_space)),
        ("Observation Shape", f"`{tuple(env.observation_space.shape)}`"),
        ("Observation Values", values(env.observation_space)),
        ("Execution", f"`{execution}`"),
        ("Checkpointable", "yes" if checkpointable else "no"),
        ("Backend", f"`{backend_version(env_id, configuration)}`"),
        ("Default drift", f"`{drift_type}`"),
        ("Driftable parameters", ", ".join(f"`{name}`" for name in fields) or "none"),
    ]
    if fields:
        bare = bare_drift_field(env, fields)
        rows.append(
            ("Default drift field", f"`{bare}`" if bare else "none: `drift=` needs a mapping")
        )
    rows.append(("Privileged (Oracle)", oracle_summary(tuple(env.oracle_fields))))
    if upstream is not None:
        rows.append(("Upstream", f"[{upstream[0]}]({upstream[1]})"))
    lines = [
        f"# {title_of(env_id)}",
        "",
        f"This environment is part of the [{family.title}](families/{family.slug}.md) "
        "family; its page lists the install command and the family's other environments.",
        "",
        "| | |",
        "|---|---|",
        *(f"| {name} | {value} |" for name, value in rows),
        "",
        content.description,
        *frame(env_id),
    ]
    sections: tuple[tuple[str, str], ...] = (
        ("Observation Space", content.observations),
        ("Action Space", content.actions),
        ("Rewards", content.rewards),
        ("Initial State", content.initial),
        ("Respawns", content.respawns),
        ("Drift", content.no_drift if content.no_drift and not fields else content.drift),
    )
    if upstream is not None and defines:
        assert upstream is not None
        lines += [
            "",
            f"The observation, action and reward definitions are upstream's: see "
            f"[{upstream[0]}]({upstream[1]}). This page lists only what changes in Perpetua.",
            "",
            "## What changes in Perpetua",
        ]
    else:
        lines += ["", "## Definition"]
    for title, text in sections:
        if text:
            lines += ["", f"### {title}", "", text]
    lines += ["", "## Arguments", "", "| Argument | Default |", "|---|---|"]
    lines += [f"| `{name}` | `{default}` |" for name, default in arguments(env_id, solo)]
    lines += ["", "## Usage", "", *usage(env, env_id)]
    if content.limits:
        lines += ["", "## Limits", "", content.limits]
    if content.guide:
        guide_title = (ROOT / "docs" / content.guide).read_text().splitlines()[0].lstrip("# ")
        lines += ["", "## See also", "", f"The [{guide_title}](../{content.guide}) guide."]
    keeps_limit = str(configuration.get("episodes", "")).startswith("time limit kept")
    lines += privileged_section(tuple(env.oracle_fields), fields, keeps_limit)
    history = content.history or (f"`{version_of(env_id)}`: initial release.",)
    lines += ["", "## Version History", "", *(f"- {entry}" for entry in history)]
    module, name = content.constructor.split(":")
    lines += [
        "",
        "## API",
        "",
        "```{eval-rst}",
        f".. autoclass:: {module}.{name}",
        "   :no-index:",
        "```",
    ]
    return "\n".join(lines).rstrip() + "\n"


def summary_row(env_id: str, page: Path) -> str:
    """One overview-table row, read from the page header so unbuilt pages still list."""
    text = page.read_text() if page.exists() else ""

    def field(name: str) -> str:
        match = re.search(rf"^\| {name} \| (.*) \|$", text, re.MULTILINE)
        return match.group(1) if match else "—"

    return (
        f"| [{title_of(env_id)}]({slug(env_id)}.md) | `{env_id}` | {field('Actions')} | "
        f"{field('Agents')} | {field('Observation Shape')} | {field('Upstream')} |"
    )


HEADER = "| Environment | Creation | Actions | Agents | Observation shape | Upstream |"
RULE = "|---|---|---|---|---|---|"


def family_page(key: str, ids: list[str]) -> str:
    family = FAMILIES[key]
    lines = [
        f"# {family.title}",
        "",
        family.intro,
        "",
        *(
            ["```bash", family.install, "```"]
            if family.install.startswith("uv ")
            else [family.install]
        ),
        "",
    ]
    if family.shared:
        lines += [family.shared, ""]
    lines += [HEADER, RULE]
    lines += [
        summary_row(env_id, OUT / f"{slug(env_id)}.md").replace("](", "](../", 1) for env_id in ids
    ]
    lines += ["", "```{toctree}", ":hidden:", "", *(f"../{slug(env_id)}" for env_id in ids), "```"]
    return "\n".join(lines) + "\n"


def index_page(groups: dict[str, list[str]]) -> str:
    lines = [
        "# Environments",
        "",
        f"{sum(len(ids) for ids in groups.values())} registered environments in "
        f"{len(groups)} families. Every one is a lifetime: `init` once, `step` forever, "
        "with respawns as events and drift owned by the framework.",
        "",
        "New here? Start with [ContinualGrid-v0](continualgrid-v0.md): it needs no extras "
        "and runs in seconds on a CPU.",
        "",
        "| Family | Environments | Install |",
        "|---|---:|---|",
    ]
    for key, ids in groups.items():
        commands = list(dict.fromkeys(install_of(env_id) for env_id in ids))
        install = (
            "<br>".join(f"`{command}`" for command in commands)
            if len(commands) <= 2
            else "differs per environment; see the family page"
        )
        lines.append(f"| [{FAMILIES[key].title}](families/{key}.md) | {len(ids)} | {install} |")
    lines += ["", "## All environments", ""]
    lines += [HEADER, RULE]
    for ids in groups.values():
        lines += [summary_row(env_id, OUT / f"{slug(env_id)}.md") for env_id in ids]
    # The families are listed in the root toctree, as PettingZoo's sidebar does.
    return "\n".join(lines) + "\n"


def groups() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {key: [] for key in FAMILIES}
    for env_id in pp.environments(builtin=True):
        result[family_of(env_id)].append(env_id)
    return {key: ids for key, ids in result.items() if ids}


def main(check: bool = False) -> int:
    """Write every page, or with ``check`` list the pages that would change."""
    stale: list[str] = []
    unbuilt: list[str] = []

    def write(page: Path, text: str) -> None:
        if not check:
            page.write_text(text)
        elif not page.exists() or page.read_text() != text:
            stale.append(str(page.relative_to(ROOT)))

    if not check:
        OUT.mkdir(parents=True, exist_ok=True)
        FAMILY_OUT.mkdir(parents=True, exist_ok=True)
    expected = {"index.md"}
    for env_id in pp.environments(builtin=True):
        page = OUT / f"{slug(env_id)}.md"
        expected.add(page.name)
        try:
            env = pp.make(env_id)
        except Exception as error:  # noqa: BLE001 - optional backend unavailable on this host
            if not page.exists():
                write(page, f"# `{env_id}`\n\nNot built on this host: `{error}`\n")
            print(f"kept {page.name}: {type(error).__name__}")
            unbuilt.append(page.name)
            continue
        try:
            write(page, describe(env, env_id))
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
    for page in OUT.glob("*.md"):
        if page.name not in expected:
            if check:
                stale.append(str(page.relative_to(ROOT)))
            else:
                page.unlink()
    families = groups()
    for key, ids in families.items():
        write(FAMILY_OUT / f"{key}.md", family_page(key, ids))
    write(OUT / "index.md", index_page(families))
    if check:
        current = "all pages are current"
        if unbuilt:
            current = (
                f"all checked pages are current; not checked on this host: {', '.join(unbuilt)}"
            )
        print("\n".join(f"stale: {name}" for name in stale) or current)
        return 1 if stale else 0
    print(f"wrote {len(expected)} environment pages and {len(families)} family pages")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate docs/environments/*.md and docs/environments/families/*.md "
        "from the registered environments and scripts/env_content.py."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; list stale pages and exit 1 if any page would change",
    )
    sys.exit(main(parser.parse_args().check))
