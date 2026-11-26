"""Draw the figures used in the documentation.

    uv run --extra training --with matplotlib python scripts/plot_docs_figures.py lifetime
    uv run --extra training --with matplotlib python scripts/plot_docs_figures.py quickstart
    uv run --all-extras --with matplotlib python scripts/plot_docs_figures.py frames

``lifetime`` is a schematic. ``quickstart`` runs the quickstart experiment and
plots its reward. ``frames`` draws one frame of each native world and of the
gymnax and MJX environments it knows how to draw, after a few random ticks: from
the state, from an image observation, or as a drawing of a control task's
positions and angles. It also draws the JaxMARL MPE tasks and MEAL Overcooked
from their state, and renders Meta-World with MuJoCo. Output goes to
``docs/figures``.
"""

# matplotlib's stubs leave many members partially typed.
# pyright: reportUnknownMemberType=false

import argparse
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Rectangle

import perpetua as pp

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, CRITICAL, GREEN = "#2a78d6", "#eb6834", "#d03b3b", "#2f8f5b"
PHASES = ("#dbe9fa", "#fde3d6", "#dcefe3")

# Fixed ids and no timestamp, so a re-run writes the same file.
mpl.rcParams.update(
    {"svg.hashsalt": "perpetua", "svg.fonttype": "none", "font.family": "sans-serif"}
)


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    print(f"wrote {path.relative_to(OUT.parents[1])}")


def lifetime() -> None:
    """Episodes against one lifetime, with drift phases and respawns marked."""
    figure, axis = plt.subplots(figsize=(8.0, 3.1), facecolor=SURFACE)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 10)
    axis.axis("off")

    axis.text(0, 9.3, "Episodes (Gymnasium)", color=INK, fontsize=11, weight="bold")
    starts = (0, 21, 34, 62, 80)
    ends = (19, 32, 60, 78, 100)
    for start, end in zip(starts, ends, strict=True):
        axis.add_patch(Rectangle((start, 7.2), end - start, 1.3, color=GRID, ec=AXIS, lw=0.8))
    for start in starts:
        axis.text(start + 0.4, 6.5, "reset", color=SECONDARY, fontsize=8)
    axis.text(100, 6.5, "done", color=SECONDARY, fontsize=8, ha="right")

    axis.text(0, 4.9, "One lifetime (Perpetua)", color=INK, fontsize=11, weight="bold")
    bounds = (0, 36, 70, 100)
    labels = ("drift phase 1", "phase 2", "phase 3")
    for index, label in enumerate(labels):
        left, right = bounds[index], bounds[index + 1]
        axis.add_patch(Rectangle((left, 2.8), right - left, 1.3, color=PHASES[index], lw=0))
        axis.text(
            (left + right) / 2, 3.45, label, color=SECONDARY, fontsize=9, ha="center", va="center"
        )
    axis.add_patch(Rectangle((0, 2.8), 100, 1.3, fill=False, ec=AXIS, lw=0.8))
    for switch in bounds[1:-1]:
        axis.plot([switch, switch], [2.5, 4.4], color=CRITICAL, lw=1.4)
        axis.text(
            switch,
            1.9,
            "switch\n(not shown to the agent)",
            color=CRITICAL,
            fontsize=8,
            ha="center",
            va="top",
        )
    for respawn in (9, 17, 28, 44, 51, 63, 77, 88, 95):
        axis.plot(respawn, 4.35, marker="v", color=BLUE, ms=5)
    axis.plot(9, 0.55, marker="v", color=BLUE, ms=5)
    axis.text(
        10.5,
        0.55,
        "respawn: an event inside the lifetime",
        color=SECONDARY,
        fontsize=8,
        va="center",
    )
    axis.text(0, 1.9, "init, once", color=SECONDARY, fontsize=8, va="top")
    axis.add_patch(
        FancyArrow(
            0,
            2.45,
            99.2,
            0,
            width=0.02,
            head_width=0.35,
            head_length=0.8,
            color=MUTED,
            length_includes_head=True,
        )
    )
    axis.text(100, 1.9, "tick t, never ends", color=SECONDARY, fontsize=8, ha="right", va="top")
    _save(figure, OUT / "lifetime.svg")


def quickstart() -> None:
    """The quickstart's reward over the lifetime, with switches and recovery spans."""
    from perpetua.agents import PPO, PPOConfig, PPORunner

    goals = [(3.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
    period, rollout = 4096, 64
    env = pp.make("ContinualGrid-v0", size=4, num_agents=4, drift=pp.drift.Cyclic(goals, period))
    config = PPOConfig(rollout_steps=rollout, entropy_coefficient=0.1)
    trainer = PPORunner(env, PPO(action_size=5, config=config))
    train_state = trainer.init(seed=0, window_size=rollout)
    chunks: list[np.ndarray] = []
    for _ in range(3 * period // rollout):
        train_state = trainer(train_state)
        metrics = train_state.lifetime.metrics
        chunks.append(np.asarray(pp.evaluation.recent_rewards(metrics, steps=rollout)))
    rewards = np.concatenate(chunks).mean(axis=1)
    switches = [period, 2 * period]
    recovery = np.asarray(pp.evaluation.recovery_ticks(rewards, jnp.asarray(switches), window=256))
    print("ticks to recover after each switch:", recovery.tolist())

    window = 256
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ticks = np.arange(window - 1, rewards.size)
    figure, axis = plt.subplots(figsize=(8.0, 3.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    top = float(smooth.max()) * 1.18
    for index, goal in enumerate(goals):
        axis.axvspan(index * period, (index + 1) * period, color=PHASES[index], lw=0)
        label = f"goal ({goal[0]:.0f}, {goal[1]:.0f})"
        axis.text(
            (index + 0.5) * period,
            top * 0.95,
            label,
            color=SECONDARY,
            fontsize=9,
            ha="center",
            va="top",
        )
    for switch, span in zip(switches, recovery.tolist(), strict=True):
        axis.axvline(switch, color=CRITICAL, lw=1.2, ls="--")
        axis.annotate(
            "",
            xy=(switch + span, top * 0.74),
            xytext=(switch, top * 0.74),
            arrowprops={"arrowstyle": "<->", "color": CRITICAL, "lw": 1.0},
        )
        axis.text(
            switch + span / 2,
            top * 0.77,
            f"recovery\n{span:,} ticks",
            color=CRITICAL,
            fontsize=8,
            ha="center",
            va="bottom",
        )
    axis.plot(ticks[::8], smooth[::8], color=BLUE, lw=1.6)
    axis.set_xlim(0, 3 * period)
    axis.set_ylim(0, top)
    axis.set_xticks([0, period, 2 * period, 3 * period])
    axis.set_xlabel("tick", color=SECONDARY)
    axis.set_ylabel(f"reward per tick\n(mean of the last {window})", color=SECONDARY)
    axis.tick_params(colors=SECONDARY, length=3)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    _save(figure, OUT / "quickstart.svg")


def _advance(env: Any, ticks: int, seed: int = 0) -> tuple[Any, Any, Any]:
    key = jax.random.key(seed)
    state, obs, info = env.init(key)
    for tick in range(ticks):
        act_key, step_key = jax.random.split(jax.random.fold_in(key, tick))
        if hasattr(env.action_space, "nvec"):
            action = pp.policies.random(obs, info, act_key)
        else:
            box = env.action_space
            action = jax.random.uniform(
                act_key, box.shape, minval=jnp.asarray(box.low), maxval=jnp.asarray(box.high)
            )
        state, obs, _, info = env.step(state, action, step_key)
    return state, obs, info


def _board(size: int) -> tuple[Any, Any]:
    figure, axis = plt.subplots(figsize=(2.6, 2.6), facecolor=SURFACE)
    axis.set_xlim(0, size)
    axis.set_ylim(size, 0)
    axis.set_aspect("equal")
    axis.axis("off")
    for row in range(size):
        for column in range(size):
            axis.add_patch(Rectangle((column, row), 1, 1, fc="white", ec=AXIS, lw=0.8))
    return figure, axis


def _goal(axis: Any, cell: Any, color: str = ORANGE) -> None:
    row, column = (int(value) for value in cell)
    axis.add_patch(
        Rectangle((column + 0.12, row + 0.12), 0.76, 0.76, fc=color, ec="none", alpha=0.85)
    )


def _agent(axis: Any, cell: Any, color: str = BLUE) -> None:
    row, column = (int(value) for value in cell)
    axis.add_patch(Circle((column + 0.5, row + 0.5), 0.27, fc=color, ec="white", lw=1.2, zorder=3))


def _moves(axis: Any, cell: Any, mask: Any) -> None:
    """Arrows for up/down/left/right: green where the move is allowed, red where masked."""
    row, column = (int(value) for value in cell)
    steps = {1: (0.0, -1.0), 2: (0.0, 1.0), 3: (-1.0, 0.0), 4: (1.0, 0.0)}
    for action, (dx, dy) in steps.items():
        color = GREEN if bool(mask[action]) else CRITICAL
        axis.add_patch(
            FancyArrow(
                column + 0.5 + 0.33 * dx,
                row + 0.5 + 0.33 * dy,
                0.3 * dx,
                0.3 * dy,
                width=0.05,
                head_width=0.2,
                head_length=0.14,
                color=color,
                length_includes_head=True,
                zorder=4,
            )
        )


def _grid_frame(env_id: str, ticks: int, **kwargs: Any) -> None:
    env: Any = pp.make(env_id, **kwargs)
    state, _, info = _advance(env, ticks)
    size = int(env.metadata["configuration"]["size"])
    world = state.env.world if hasattr(state.env, "world") else state.env
    positions = np.asarray(world.positions)
    active = np.asarray(info["active_mask"])
    privileged = env.privileged(state)
    figure, axis = _board(size)
    if "goal_parameters" in privileged:
        anchor = np.asarray(privileged["goal_parameters"]).astype(int)
        if env_id in ("CooperativeGrid-v0", "PartnerGrid-v0"):
            # One target per slot: consecutive cells in row-major order from the anchor.
            start = int(anchor[0]) * size + int(anchor[1])
            for slot in range(positions.shape[0]):
                cell = (start + slot) % (size * size)
                _goal(axis, (cell // size, cell % size))
        else:
            _goal(axis, anchor)
    else:
        _goal(axis, (size - 1, size - 1))
    for slot, cell in enumerate(positions):
        _agent(axis, cell, BLUE if bool(active[slot]) else MUTED)
    if env_id == "DoorGrid-v0":
        _moves(axis, positions[0], np.asarray(info["action_mask"])[0])
    _save(figure, OUT / "environments" / f"{env_id.lower()}.svg")
    close = getattr(env, "close", None)
    if close is not None:
        close()


def _bean_frame() -> None:
    env: Any = pp.make("BeanWorld-v0")
    state, _, _ = _advance(env, 8)
    items = np.asarray(state.env.items)
    row, column = (int(value) for value in np.asarray(state.env.positions)[0])
    half, view = 16, int(env.metadata["configuration"]["view"])
    rows = np.arange(row - half, row + half + 1) % items.shape[0]
    columns = np.arange(column - half, column + half + 1) % items.shape[1]
    crop = items[np.ix_(rows, columns)]
    colors = np.asarray(
        [[1.0, 1.0, 1.0], [0.16, 0.47, 0.84], [0.93, 0.74, 0.20], [0.82, 0.23, 0.23]]
    )
    figure, axis = plt.subplots(figsize=(2.6, 2.6), facecolor=SURFACE)
    axis.imshow(colors[crop], interpolation="nearest")
    axis.add_patch(
        Rectangle((half - view / 2, half - view / 2), view, view, fill=False, ec=INK, lw=1.0)
    )
    axis.add_patch(Circle((half, half), 0.6, fc=INK, ec="white", lw=0.8))
    axis.set_xticks([])
    axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color(AXIS)
    _save(figure, OUT / "environments" / "beanworld-v0.svg")


def _slug(env_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", env_id.lower()).strip("-")


def _plain(size: tuple[float, float] = (2.6, 2.6)) -> tuple[Any, Any]:
    figure, axis = plt.subplots(figsize=size, facecolor=SURFACE)
    axis.set_xticks([])
    axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color(AXIS)
    return figure, axis


CHANNELS = np.asarray(
    [
        [1.0, 1.0, 1.0],
        [0.16, 0.47, 0.84],
        [0.92, 0.41, 0.20],
        [0.18, 0.56, 0.36],
        [0.93, 0.74, 0.20],
        [0.54, 0.34, 0.74],
        [0.82, 0.23, 0.23],
        [0.32, 0.32, 0.31],
    ]
)


def _image_frame(env_id: str, ticks: int) -> None:
    """The observation itself: one colour per channel, or grey levels for one channel."""
    env: Any = pp.make(env_id)
    _, obs, _ = _advance(env, ticks)
    image = np.asarray(obs)[0]
    figure, axis = _plain()
    if image.ndim == 3:
        index = np.where(image.max(axis=2) > 0, image.argmax(axis=2) + 1, 0)
        axis.imshow(CHANNELS[index], interpolation="nearest")
    else:
        axis.imshow(image, cmap="Greys", interpolation="nearest")
    _save(figure, OUT / "environments" / f"{_slug(env_id)}.svg")


def _maze_frame(env_id: str, ticks: int) -> None:
    """Walls from the gymnax map, with the agent and the goal from its state."""
    env: Any = pp.make(env_id)
    state, _, _ = _advance(env, ticks)
    walls = 1.0 - np.asarray(env.env.env.env_map, dtype=np.float32)
    inner = state.env.env
    figure, axis = _plain()
    axis.imshow(walls, cmap="Greys", vmin=0.0, vmax=1.6, interpolation="nearest")
    goal, position = np.asarray(inner.goal), np.asarray(inner.pos)
    axis.add_patch(Rectangle((goal[1] - 0.4, goal[0] - 0.4), 0.8, 0.8, fc=ORANGE, ec="none"))
    axis.add_patch(Circle((position[1], position[0]), 0.33, fc=BLUE, ec="white", lw=0.8))
    _save(figure, OUT / "environments" / f"{_slug(env_id)}.svg")


def _stage(low: float, high: float, bottom: float, top: float) -> tuple[Any, Any]:
    figure, axis = _plain((2.9, 2.2))
    axis.set_facecolor("white")
    axis.set_xlim(low, high)
    axis.set_ylim(bottom, top)
    axis.set_aspect("equal")
    return figure, axis


def _rod(axis: Any, start: tuple[float, float], angle: float, length: float) -> tuple[float, float]:
    """A rod from ``start`` at ``angle`` from upright (clockwise positive); returns its tip."""
    tip = (start[0] + length * np.sin(angle), start[1] + length * np.cos(angle))
    axis.plot([start[0], tip[0]], [start[1], tip[1]], color=ORANGE, lw=5, solid_capstyle="round")
    axis.add_patch(Circle(start, 0.035 * length + 0.02, fc=INK, ec="none", zorder=3))
    return tip


def _control_frame(env_id: str, ticks: int) -> None:
    """A drawing from the observation: positions and angles, not a simulator render."""
    env: Any = pp.make(env_id)
    _, obs, _ = _advance(env, ticks)
    o = np.asarray(obs)[0]
    name = env_id.split("/")[1]
    if env_id in ("gymnax/CartPole-v1", "mjx/CartPole-v0"):
        if env_id.startswith("gymnax"):
            x, angle = float(o[0]), float(o[2])
        else:
            x, angle = float(o[0]), float(np.arctan2(o[3], o[2]))
        low = -0.6 if env_id.startswith("gymnax") else -1.3  # the MJX pole starts hanging
        figure, axis = _stage(-2.4, 2.4, low, 1.5)
        axis.plot([-2.4, 2.4], [0, 0], color=AXIS, lw=1.2)
        axis.add_patch(Rectangle((x - 0.3, -0.12), 0.6, 0.24, fc=BLUE, ec="none"))
        _rod(axis, (x, 0.0), angle, 1.0)
    elif name.startswith("Pendulum"):
        figure, axis = _stage(-1.3, 1.3, -1.3, 1.3)
        _rod(axis, (0.0, 0.0), float(np.arctan2(o[1], o[0])), 1.0)
    elif name.startswith("Acrobot"):
        # Angles are from hanging straight down; the second is relative to the first.
        first = float(np.arctan2(o[1], o[0]))
        second = first + float(np.arctan2(o[3], o[2]))
        figure, axis = _stage(-2.3, 2.3, -2.3, 2.3)
        axis.plot([-2.3, 2.3], [1.0, 1.0], color=AXIS, lw=1.0, ls="--")
        joint = _rod(axis, (0.0, 0.0), np.pi - first, 1.0)
        _rod(axis, joint, np.pi - second, 1.0)
    else:  # MountainCar
        x = float(o[0])
        track = np.linspace(-1.2, 0.6, 200)
        figure, axis = _stage(-1.25, 0.65, 0.0, 1.15)
        axis.plot(track, np.sin(3 * track) * 0.45 + 0.55, color=SECONDARY, lw=1.5)
        goal = 0.45 if "Continuous" in name else 0.5
        height = float(np.sin(3 * goal) * 0.45 + 0.55)
        axis.plot([goal, goal], [height, height + 0.16], color=ORANGE, lw=2)
        axis.add_patch(Circle((x, float(np.sin(3 * x) * 0.45 + 0.55) + 0.04), 0.05, fc=BLUE))
    _save(figure, OUT / "environments" / f"{_slug(env_id)}.svg")


def _point_frame(env_id: str, ticks: int) -> None:
    """Reacher's two unit links, or Swimmer's point among its urchins, from the state."""
    env: Any = pp.make(env_id)
    state, _, _ = _advance(env, ticks)
    inner = state.env.env
    goal: Any = np.asarray(inner.goal_xy)
    if "Reacher" in env_id:
        figure, axis = _stage(-2.3, 2.3, -2.3, 2.3)
        joint = (0.0, 0.0)
        for angle in np.asarray(inner.angles):
            tip = (joint[0] + float(np.cos(angle)), joint[1] + float(np.sin(angle)))
            axis.plot(*zip(joint, tip, strict=True), color=BLUE, lw=5, solid_capstyle="round")
            axis.add_patch(Circle(joint, 0.07, fc=INK, ec="none", zorder=3))
            joint = tip
    else:
        figure, axis = _stage(-1.3, 1.3, -1.3, 1.3)
        urchins: Any = np.asarray(inner.urchin_xys)
        for urchin in urchins:
            axis.add_patch(Circle(urchin, 0.06, fc=SECONDARY, ec="none"))
        axis.add_patch(
            Circle(
                tuple(np.asarray(inner.xy).tolist()), 0.07, fc=BLUE, ec="white", lw=0.8, zorder=3
            )
        )
    axis.add_patch(Rectangle(goal - 0.09, 0.18, 0.18, fc=ORANGE, ec="none", zorder=2))
    _save(figure, OUT / "environments" / f"{_slug(env_id)}.svg")


def _mpe_frame(env_id: str, ticks: int) -> None:
    """Entity positions and radii from the JaxMARL state: agents first, then landmarks."""
    env: Any = pp.make(env_id)
    state, _, _ = _advance(env, ticks, seed=3)  # seed 0 starts everything on one side
    backend = env.env.backend
    positions = np.asarray(state.env.world.p_pos)
    radii = np.asarray(backend.rad)
    reach = max(1.3, float(np.abs(positions).max()) + 0.35)
    figure, axis = _stage(-reach, reach, -reach, reach)
    names = list(backend.agents)
    for index, (position, radius) in enumerate(zip(positions, radii, strict=True)):
        if index >= len(names):
            color, order = AXIS, 1
        elif names[index].startswith(("adversary", "leadadversary", "eve")):
            color, order = ORANGE, 3
        else:
            color, order = BLUE, 3
        axis.add_patch(Circle(position, float(radius), fc=color, ec="white", lw=0.6, zorder=order))
    _save(figure, OUT / "environments" / f"{_slug(env_id)}.svg")


OVERCOOKED = {
    "wall": "#b9b8ae",
    "onion_pile": "#edbd33",
    "plate_pile": "#8a57bd",
    "goal": GREEN,
    "pot": INK,
}


def _overcooked_frame(ticks: int) -> None:
    """The MEAL kitchen from the state's object map, with the cooks from their positions."""
    from meal.env.overcooked import OBJECT_TO_INDEX

    env_id = "meal/overcooked-v0"
    env: Any = pp.make(env_id)
    state, _, _ = _advance(env, ticks)
    world = state.env.world
    kinds = np.asarray(world.maze_map)[..., 0]
    figure, axis = _plain()
    axis.set_facecolor("white")
    axis.set_xlim(0, kinds.shape[1])
    axis.set_ylim(kinds.shape[0], 0)
    axis.set_aspect("equal")
    for name, color in OVERCOOKED.items():
        for row, column in np.argwhere(kinds == OBJECT_TO_INDEX[name]):
            inset = 0.0 if name == "wall" else 0.12
            axis.add_patch(
                Rectangle(
                    (column + inset, row + inset),
                    1 - 2 * inset,
                    1 - 2 * inset,
                    fc=color,
                    ec=color,
                    lw=0.4,
                )
            )
    for x, y in np.asarray(world.agent_pos):
        axis.add_patch(Circle((x + 0.5, y + 0.5), 0.33, fc=BLUE, ec="white", lw=0.8, zorder=3))
    _save(figure, OUT / "environments" / f"{_slug(env_id)}.svg")


def _metaworld_frame(ticks: int) -> None:
    """A MuJoCo render of the upstream task; the adapter itself runs without a renderer."""
    import gymnasium as gym
    import metaworld  # noqa: F401  (registers the Meta-World ids)

    source: Any = pp.make("metaworld/MT1-v0")
    task = source.metadata["configuration"]["backend"]["task"]
    env: Any = gym.make(
        "Meta-World/MT1",
        env_name=task,
        render_mode="rgb_array",
        camera_name="corner2",
        width=480,
        height=360,
    )
    env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(ticks):
        env.step(env.action_space.sample())
    image = np.asarray(env.render())[::-1, ::-1]  # this camera renders upside down
    env.close()
    path = OUT / "environments" / "metaworld-mt1-v0.png"
    plt.imsave(path, image)
    print(f"wrote {path.relative_to(OUT.parents[1])}")


MPE_TASKS = (
    "MPE_simple_v3",
    "MPE_simple_spread-v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    "MPE_simple_crypto_v3",
)


def frames() -> None:
    for name in MPE_TASKS:
        _mpe_frame(f"jaxmarl/{name}", 12)
    for env_id in ("gymnax/Reacher-misc", "gymnax/Swimmer-misc"):
        _point_frame(env_id, 12)
    _overcooked_frame(6)
    _metaworld_frame(12)
    for env_id in (
        "gymnax/Asterix-MinAtar",
        "gymnax/Breakout-MinAtar",
        "gymnax/Freeway-MinAtar",
        "gymnax/SpaceInvaders-MinAtar",
        "gymnax/Pong-misc",
        "gymnax/Catch-bsuite",
        "gymnax/DeepSea-bsuite",
        "gymnax/MNISTBandit-bsuite",
    ):
        _image_frame(env_id, 40)
    for env_id in ("gymnax/FourRooms-misc", "gymnax/MetaMaze-misc"):
        _maze_frame(env_id, 8)
    for env_id in (
        "gymnax/CartPole-v1",
        "gymnax/Pendulum-v1",
        "gymnax/Acrobot-v1",
        "gymnax/MountainCar-v0",
        "gymnax/MountainCarContinuous-v0",
        "mjx/Pendulum-v0",
        "mjx/CartPole-v0",
    ):
        _control_frame(env_id, 12)
    _grid_frame("ContinualGrid-v0", 6)
    _grid_frame(
        "DoorGrid-v0", 6, drift=pp.drift.Cyclic([(1.0, 1.0, 1.0, 1.0), (0.0, 1.0, 1.0, 0.0)], 4)
    )
    _grid_frame("ForagerGrid-v0", 6)
    _grid_frame("CooperativeGrid-v0", 6)
    _grid_frame("PartnerGrid-v0", 6)
    _grid_frame("OpponentGrid-v0", 6)
    _bean_frame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the figures used in the documentation.")
    parser.add_argument("figure", choices=("lifetime", "quickstart", "frames", "all"))
    args = parser.parse_args()
    for name, draw in (("lifetime", lifetime), ("quickstart", quickstart), ("frames", frames)):
        if args.figure in (name, "all"):
            draw()


if __name__ == "__main__":
    main()
