"""Generated environment pages must cover, match and run against the live registry."""

# pyright: reportPrivateUsage=false
# JAX 0.11 leaves jit partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from helpers import unavailable

import perpetua as pp
from perpetua.envs.opponent_grid import OpponentGrid

path = Path(__file__).resolve().parents[1] / "scripts" / "generate_env_docs.py"
spec = importlib.util.spec_from_file_location("generate_env_docs", path)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
OUT, FAMILY_OUT, describe, slug = module.OUT, module.FAMILY_OUT, module.describe, module.slug
groups, family_page, index_page = module.groups, module.family_page, module.index_page
BUILTIN = pp.environments(builtin=True)
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def test_environment_pages_cover_exactly_the_registered_ids() -> None:
    expected = {f"{slug(env_id)}.md" for env_id in BUILTIN}
    actual = {page.name for page in OUT.glob("*.md")} - {"index.md"}
    assert actual == expected
    index = (OUT / "index.md").read_text()
    for env_id in BUILTIN:
        assert f"]({slug(env_id)}.md) | `{env_id}` |" in index
    assert "gymnax/SimpleBandit-bsuite" not in BUILTIN
    assert "gymnax/PointRobot-misc" not in BUILTIN


def test_every_builtin_id_has_content_and_an_unknown_id_is_refused() -> None:
    for env_id in BUILTIN:
        content = module.content_for(env_id)
        assert content.description, env_id
        assert content.family in module.FAMILIES, env_id
    # An id outside the two generic families used to get the particle-task text.
    with pytest.raises(KeyError, match="no ENVS entry for 'newsim/Thing-v0'"):
        module.content_for("newsim/Thing-v0")


def test_install_rows_come_from_the_registry() -> None:
    for env_id in BUILTIN:
        record = pp.spec(env_id)
        command = module.install_of(env_id)
        assert f"| Install | `{command}` |" in page_text(env_id), env_id
        for extra in (record.extra, record.optional_extra):
            assert extra is None or f"--extra {extra}" in command, env_id
        assert record.setup is None or command == f"uv run python -m {record.setup}", env_id
        assert (record.platform == "linux") == command.endswith("# Linux"), env_id
    assert module.install_of("ContinualGrid-v0") == "uv sync --locked"


def test_family_pages_and_index_match_the_generator() -> None:
    families = groups()
    assert {page.stem for page in FAMILY_OUT.glob("*.md")} == set(families)
    assert sum(len(ids) for ids in families.values()) == len(BUILTIN)
    for key, ids in families.items():
        assert (FAMILY_OUT / f"{key}.md").read_text() == family_page(key, ids), key
    assert (OUT / "index.md").read_text() == index_page(families)


@pytest.mark.parametrize("env_id", BUILTIN)
def test_page_matches_live_environment_metadata(env_id: str) -> None:
    reason = unavailable(env_id)
    if reason is not None:
        pytest.skip(reason)
    env = pp.make(env_id)
    try:
        expected = describe(env, env_id)
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    assert (OUT / f"{slug(env_id)}.md").read_text() == expected


@pytest.mark.parametrize("env_id", BUILTIN)
def test_page_usage_blocks_execute(env_id: str) -> None:
    reason = unavailable(env_id)
    if reason is not None:
        pytest.skip(reason)
    blocks = BLOCK.findall((OUT / f"{slug(env_id)}.md").read_text())
    assert blocks, env_id
    for code in blocks:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False
        )
        assert result.returncode == 0, f"{env_id}:\n{result.stderr[-2000:]}"


def page_text(env_id: str) -> str:
    return (OUT / f"{slug(env_id)}.md").read_text()


def respawn_ticks(env_id: str, ticks: int) -> list[int]:
    """The ticks (counted from 1) on which ``info["respawn"]`` is true, under a fixed policy."""
    env: Any = pp.make(env_id)
    state, obs, info = env.init(jax.random.key(0))
    step: Any = jax.jit(env.step)
    found: list[int] = []
    for tick in range(ticks):
        if hasattr(env.action_space, "nvec"):
            action = pp.policies.random(obs, info, jax.random.key(100 + tick))
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        state, obs, _, info = step(state, action, jax.random.key(1000 + tick))
        if bool(np.asarray(info["respawn"]).any()):
            found.append(tick + 1)
    return found


def test_forager_example_reaches_a_mass_the_old_constants_could_not() -> None:
    match = re.search(
        r'drift=\{"mass": (pp\.drift\.Accumulate\(.*?\))\}', page_text("ForagerGrid-v0")
    )
    assert match is not None
    process = eval(match.group(1), {"pp": pp})  # noqa: S307 - the page's own example
    env = pp.Oracle(pp.make("ForagerGrid-v0", drift={"mass": process}), expose=["mass_parameters"])
    state, obs, info = env.init(jax.random.key(0))
    masses: list[float] = []
    for tick in range(400):
        action = pp.policies.greedy(obs, info, jax.random.key(100 + tick))
        state, obs, _, info = env.step(state, action, jax.random.key(1000 + tick))
        masses.append(float(np.max(env.privileged(state)["mass_parameters"])))
    # decay 0.5 with gain 1 can never pass 2; the documented example must be able to.
    assert max(masses) > 2.0


def test_opponent_grid_page_states_that_spawns_can_be_unsafe() -> None:
    text = page_text("OpponentGrid-v0")
    assert "spawn apart" not in text
    assert "Manhattan distance 1 or less" in text
    world = OpponentGrid(size=5)
    near = 0
    for seed in range(100):
        state, _, _ = world.init(jax.random.key(seed), world.default_params)
        learner, opponent = np.asarray(state.world.positions)
        near += int(np.abs(learner - opponent).sum() <= 1)
    assert near > 0


def test_bean_world_layout_sentence_matches_the_observation() -> None:
    assert "reshape them to `(view, view, 3)`" in page_text("BeanWorld-v0")
    env = pp.make("BeanWorld-v0")
    moves = {"up": (1, (2, 3)), "down": (2, (4, 3)), "left": (3, (3, 2)), "right": (4, (3, 4))}
    checked = 0
    for action, (row, column) in moves.values():
        for seed in range(60):
            state, obs, _ = env.init(jax.random.key(seed))
            window = np.asarray(obs)[0, :147].reshape(7, 7, 3)
            if window[3, 3].sum() == 0 and window[row, column].tolist() == [1, 0, 0]:
                _, _, reward, _ = env.step(state, np.array([action]), jax.random.key(1))
                assert float(reward[0]) == 1.0  # the jelly bean is where (row, column, item) says
                checked += 1
                break
    assert checked == len(moves)


def test_gymnax_pages_without_float_parameters_say_drift_is_rejected() -> None:
    nothing = [
        env_id
        for env_id in BUILTIN
        if env_id.startswith("gymnax/") and "| Driftable parameters | none |" in page_text(env_id)
    ]
    assert len(nothing) == 9
    for env_id in nothing:
        text = page_text(env_id)
        assert "Any float parameter" not in text, env_id
        assert "`drift=` is rejected" in text, env_id
    with pytest.raises(ValueError, match="no driftable parameters"):
        pp.make("gymnax/Pong-misc", drift={"ball_x_speed": pp.drift.Stationary()})


def test_respawn_cadence_matches_the_pages() -> None:
    assert respawn_ticks("gymnax/MNISTBandit-bsuite", 5) == [1, 2, 3, 4, 5]
    assert respawn_ticks("gymnax/Reacher-misc", 30) == []
    assert respawn_ticks("jaxmarl/MPE_simple_spread-v3", 60) == [26, 52]
    assert "ticks 26, 52, 78" in page_text("jaxmarl/MPE_simple_spread-v3")
    assert "true on every tick" in page_text("gymnax/MNISTBandit-bsuite")
    assert "there is never a respawn" in page_text("gymnax/Reacher-misc")


def test_simple_crypto_page_states_the_255_scale() -> None:
    env = pp.make("jaxmarl/MPE_simple_crypto_v3")
    state, obs, info = env.init(jax.random.key(0))
    state, obs, reward, info = env.step(
        state, pp.policies.random(obs, info, jax.random.key(1)), jax.random.key(2)
    )
    assert float(np.max(np.asarray(obs))) == 255.0
    assert float(np.min(np.asarray(reward))) < -1e4
    text = page_text("jaxmarl/MPE_simple_crypto_v3")
    assert "255" in text
    assert "The observation, action and reward definitions are upstream's" not in text


def test_gymnax_pages_state_the_field_a_bare_drift_drives() -> None:
    row = re.compile(r"^\| Default drift field \| `(.*)` \|$", re.MULTILINE)
    alphabetical_differs: list[str] = []
    for env_id in BUILTIN:
        if not env_id.startswith("gymnax/"):
            continue
        env: Any = pp.make(env_id)
        field = env.env.default_drift_field
        text = page_text(env_id)
        stated = row.search(text)
        assert (stated.group(1) if stated else None) == field, env_id
        if field is None:
            continue
        assert env._bare_field() == field, env_id
        listed = re.search(r"^\| Driftable parameters \| `(.*?)`", text, re.MULTILINE)
        assert listed is not None, env_id
        if listed.group(1) != field:
            alphabetical_differs.append(env_id.removeprefix("gymnax/"))
    # The pages list driftable parameters alphabetically; these start with another field.
    assert sorted(alphabetical_differs) == [
        "BernoulliBandit-misc",
        "MetaMaze-misc",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Reacher-misc",
    ]
    drifting: Any = pp.make("gymnax/MountainCar-v0", drift=pp.drift.Cyclic([-1.2, -1.0], 10))
    assert list(drifting.drifts) == ["min_position"]


def test_agarcl_grand_arena_arguments_show_the_registered_preset() -> None:
    from perpetua.adapters.agarcl import GRAND_ARENA

    rows = dict(module.arguments("agarcl/grand-arena-v0"))
    text = page_text("agarcl/grand-arena-v0")
    overlaid = {name: value for name, value in GRAND_ARENA.items() if name != "obs_type"}
    assert len(overlaid) == 5
    for name, value in overlaid.items():
        assert rows[name] == repr(value), name
        assert f"| `{name}` | `{value!r}` |" in text, name
    plain = dict(module.arguments("agarcl/agario-screen-v0"))
    assert (plain["screen_len"], plain["num_bots"], plain["arena_size"]) == ("84", "0", "None")


def test_single_agent_mpe_page_does_not_offer_scripted_slots() -> None:
    assert "scripted" not in page_text("jaxmarl/MPE_simple_v3").replace(
        "it takes no scripted slots", ""
    )
    with pytest.raises(ValueError, match="at least one must be a learner"):
        pp.make("jaxmarl/MPE_simple_v3", num_scripted=1)
    for env_id in BUILTIN:
        if env_id.startswith("jaxmarl/") and env_id != "jaxmarl/MPE_simple_v3":
            text = page_text(env_id)
            assert "| `num_scripted` | `0` |" in text, env_id
            assert "| `scripted_drift` | `None` |" in text, env_id
