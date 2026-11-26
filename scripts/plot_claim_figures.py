"""Draw the figures for the reports written by ``scripts/claim_figures.py``.

    uv run --with matplotlib python scripts/plot_claim_figures.py switching \
        grid.json mjx.json --out switching.png
    uv run --with matplotlib python scripts/plot_claim_figures.py reset reset.json --out reset.png
    uv run --with matplotlib python scripts/plot_claim_figures.py matrix matrix.json --out m.png

``throughput`` reads reports of ``examples/measure_training_throughput.py`` and
``scripts/measure_baseline.py`` (one baseline report per seed), in row order:

    uv run --with matplotlib python scripts/plot_claim_figures.py throughput \
        baseline-seed*.json native-128.json native-512.json --out throughput.png

``tuning`` reads reports of ``scripts/measure_tuning.py --mode sweep``, one per
sweep size and training length:

    uv run --with matplotlib python scripts/plot_claim_figures.py tuning \
        sweep-*.json --out tuning.png

``lifetime`` reads the JSON-lines logs of ``scripts/reproduce_continual_world.py cl``
(one per seed) instead and prints ``pp.evaluation.forgetting`` per seed: the
best score of the policies saved from each task onward minus the final policy's
score, averaged over the first n-1 tasks. This is not the Continual World
paper's forgetting (the score right after training on a task minus the final
score, averaged over all n tasks), which ``reproduce_continual_world.py
metrics`` reports, so the two numbers differ on the same log:

    uv run --with matplotlib python scripts/plot_claim_figures.py lifetime \
        cl-seed0.jsonl cl-seed1.jsonl --out lifetime.png

``reward-lifetime`` does the same for logs that score reward, not success
(``scripts/coom_lifetime.py``), rescaling each task between the untrained
policy and the policy that has just trained on it:

    uv run --with matplotlib python scripts/plot_claim_figures.py reward-lifetime \
        co8-ppo-seed*.jsonl --out lifetime-coom.png
"""

# matplotlib's stubs leave many members partially typed.
# pyright: reportUnknownMemberType=false

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import perpetua as pp

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, CRITICAL = "#2a78d6", "#eb6834", "#d03b3b"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def plot_switching(paths: list[str], out: Path) -> None:
    """Cumulative wall time per phase: one panel per environment."""
    reports = [_load(path) for path in paths]
    figure, axes = plt.subplots(1, len(reports), figsize=(5.2 * len(reports), 3.6), squeeze=False)
    for axis, report in zip(axes[0], reports, strict=True):
        phases = np.arange(report["phases"] + 1)
        top = 0.0
        for name, label, color in (
            ("naive", "Naive task switching", ORANGE),
            ("lifetime", "Drift as data", BLUE),
        ):
            total = np.concatenate(([0.0], np.cumsum(report[name]["seconds"])))
            compiles = report[name]["compile_events"]
            later = sum(compiles[1:])
            axis.step(phases, total, where="post", color=color, linewidth=2)
            axis.annotate(
                f"{label}\n{total[-1]:.3g} s, {later} recompiles after phase 1",
                (phases[-1], total[-1]),
                xytext=(-6, 8 if name == "lifetime" else -6),
                textcoords="offset points",
                ha="right",
                va="bottom" if name == "lifetime" else "top",
                color=SECONDARY,
                fontsize=9,
            )
            axis.plot(phases[-1], total[-1], "o", color=color, markersize=6)
            top = max(top, float(total[-1]))
        axis.set_ylim(0, top * 1.08)
        axis.set_xlim(0, report["phases"] * 1.02)
        axis.set_title(f"{report['env']} ({report['backend']})")
        axis.set_xlabel(
            f"Phase ({report['ticks_per_phase']} ticks each, a new task value per phase)"
        )
        axis.set_ylabel("Cumulative wall time (s)")
    figure.savefig(out)


def plot_reset(path: str, out: Path) -> None:
    """Auto-reset cost per environment: static FLOPs and measured slowdown."""
    report = _load(path)
    rows = [row for row in report["rows"] if "error" not in row]
    batches = sorted({row["batch"] for row in rows})
    styles = ((BLUE, "o"), (ORANGE, "s"))
    if len(batches) > len(styles):
        raise ValueError(
            f"The reset figure draws at most {len(styles)} batch sizes; {path} has {batches}."
        )
    by_env: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_env.setdefault(row["env"], {})[row["batch"]] = row

    def slowdown(row: dict[str, Any]) -> float:
        return row["lifetime_steps_per_second"] / row["autoreset_steps_per_second"]

    names = sorted(by_env, key=lambda name: slowdown(by_env[name][batches[0]]))
    labels = [name.removeprefix("gymnax/") for name in names]
    y = np.arange(len(names))
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, 0.3 * len(names) + 1.6), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    single = [by_env[name][1] for name in names]
    extra = [row["autoreset_flops"] / row["step_flops"] for row in single]
    left.barh(y, extra, height=0.6, color=BLUE)
    left.axvline(1, color=AXIS, linewidth=1)
    left.set_title("Static cost of one step")
    left.set_xlabel("Auto-reset step FLOPs / lifetime step FLOPs")
    left.set_yticks(y, labels)
    left.grid(axis="y", visible=False)
    for batch, (color, marker) in zip(batches, styles, strict=False):
        values = [slowdown(by_env[name][batch]) for name in names]
        right.plot(
            values,
            y,
            marker,
            color=color,
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            label="One lifetime" if batch == 1 else f"vmap over {batch} lifetimes",
        )
    right.axvline(1, color=AXIS, linewidth=1)
    right.set_xscale("log")
    right.set_title(f"Measured speed ({report['backend']})")
    right.set_xlabel("Lifetime steps/s / auto-reset steps/s (log)")
    right.grid(axis="y", visible=False)
    right.legend(frameon=False, loc="lower right", labelcolor=SECONDARY)
    figure.savefig(out)


def plot_matrix(path: str, out: Path) -> None:
    """Learner by benchmark grid: ran, not applicable, or failed."""
    cells = _load(path)["cells"]
    benchmarks = list(dict.fromkeys(cell["benchmark"] for cell in cells))
    learners = list(dict.fromkeys(cell["learner"] for cell in cells))
    styles = {
        "ran": (BLUE, "#ffffff", "ran"),
        "not applicable": ("#f0efec", MUTED, "n/a"),
        "failed": (CRITICAL, "#ffffff", "✕ failed"),
    }
    figure, axis = plt.subplots(figsize=(1.25 * len(learners) + 2.2, 0.55 * len(benchmarks) + 1.0))
    for cell in cells:
        x, y = learners.index(cell["learner"]), benchmarks.index(cell["benchmark"])
        fill, ink, text = styles[cell["status"]]
        axis.add_patch(Rectangle((x + 0.04, y + 0.06), 0.92, 0.88, color=fill, linewidth=0))
        axis.text(x + 0.5, y + 0.5, text, ha="center", va="center", color=ink, fontsize=9)
    axis.set_xlim(0, len(learners))
    axis.set_ylim(len(benchmarks), 0)
    axis.set_xticks(np.arange(len(learners)) + 0.5, learners)
    axis.set_yticks(np.arange(len(benchmarks)) + 0.5, benchmarks)
    axis.xaxis.tick_top()
    axis.tick_params(length=0)
    axis.grid(visible=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.savefig(out)


def plot_lifetime(paths: list[str], out: Path) -> None:
    """One training lifetime per seed: online traces and the revisit matrix."""
    runs = [[json.loads(line) for line in Path(path).read_text().splitlines()] for path in paths]
    header = runs[0][0]
    tasks = [name.removesuffix("-v3") for name in header["tasks"]]
    count, steps = len(tasks), int(header["steps_per_task"])
    matrices = []
    for records in runs:
        matrix = np.full((count, count), np.nan)
        for record in records:
            if record["kind"] == "boundary_eval":
                matrix[record["tick"] // steps - 1, record["task"]] = record["success"]
        matrices.append(matrix)
    forgetting = [float(pp.evaluation.forgetting(matrix)) for matrix in matrices]
    print(json.dumps({"forgetting_per_seed": forgetting, "mean": float(np.mean(forgetting))}))

    figure = plt.figure(figsize=(13, 5.6))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.55, 1], wspace=0.34, hspace=0.16)
    traces = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[1, 0])]
    for axis, field, label in zip(
        traces,
        ("success", "window_reward"),
        ("Success on the current task", "Reward per tick (last 1,000 ticks)"),
        strict=True,
    ):
        curves = []
        ticks = np.empty(0)
        for records in runs:
            evals = [record for record in records if record["kind"] == "eval"]
            ticks = np.asarray([record["tick"] for record in evals]) / 1e6
            curves.append([record[field] for record in evals])
            axis.plot(ticks, curves[-1], color=BLUE, linewidth=0.8, alpha=0.35)
        axis.plot(ticks, np.mean(curves, axis=0), color=BLUE, linewidth=2)
        for switch in range(1, count):
            axis.axvline(switch * steps / 1e6, color=AXIS, linewidth=0.8)
        axis.set_xlim(0, count * steps / 1e6)
        axis.set_ylabel(label)
        axis.grid(axis="x", visible=False)
    traces[0].set_title(f"Online, during training (mean of {len(runs)} seeds, seeds faint)")
    traces[0].set_xticks((np.arange(count) + 0.5) * steps / 1e6, tasks, rotation=30, ha="left")
    traces[0].xaxis.tick_top()
    traces[0].tick_params(axis="x", length=0, labelsize=8)
    traces[1].set_xlabel("Lifetime tick (millions); a task switch at each line")

    _revisit(
        figure.add_subplot(grid[:, 1]),
        np.mean(matrices, axis=0),
        tasks,
        f"Revisit: success of each saved policy (pp forgetting {np.mean(forgetting):.2f})",
    )
    figure.savefig(out)


def _revisit(heat: Any, mean: Any, tasks: list[str], title: str) -> None:
    count = len(tasks)
    heat.imshow(mean, cmap=BLUES, vmin=0, vmax=1)
    for row in range(count):
        for column in range(count):
            value = mean[row, column]
            heat.text(
                column,
                row,
                f"{value:.1f}".replace("0.", "."),
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if value > 0.55 else SECONDARY,
            )
    heat.set_title(title)
    heat.set_xticks(np.arange(count), tasks, rotation=45, ha="right", fontsize=8)
    heat.set_yticks(np.arange(count), [f"after {name}" for name in tasks], fontsize=8)
    heat.set_xlabel("Evaluated on task")
    heat.grid(visible=False)
    heat.tick_params(length=0)
    for spine in heat.spines.values():
        spine.set_visible(False)


def plot_reward_lifetime(paths: list[str], out: Path) -> None:
    """A reward-scored lifetime per seed: online reward and the rescaled revisit matrix."""
    runs = [[json.loads(line) for line in Path(path).read_text().splitlines()] for path in paths]
    header = runs[0][0]
    tasks = [name.split("/")[0].replace("_", " ") for name in header["tasks"]]
    count, steps = len(tasks), int(header["steps_per_task"])
    raw, scaled, curves = [], [], []
    ticks = np.empty(0)
    for records in runs:
        matrix = np.full((count + 1, count), np.nan)
        for record in records:
            if record["kind"] == "boundary_eval":
                matrix[record["tick"] // steps, record["task"]] = record["reward"]
        # 0 is the untrained policy, 1 the policy that has just trained on the task.
        raw.append(matrix)
        scaled.append((matrix[1:] - matrix[0]) / (np.diag(matrix[1:]) - matrix[0]))
        evals = [record for record in records if record["kind"] == "eval"]
        ticks = np.asarray([record["tick"] for record in evals], dtype=float)
        # Differencing the lifetime mean gives the mean reward of each interval.
        total = ticks * np.asarray([record["lifetime_reward"] for record in evals])
        curves.append(np.diff(total, prepend=0.0) / np.diff(ticks, prepend=0.0))
    forgetting = [float(pp.evaluation.forgetting(matrix)) for matrix in scaled]
    mean = np.mean(raw, axis=0)
    print(
        json.dumps(
            {
                "forgetting_per_seed": forgetting,
                "mean": float(np.mean(forgetting)),
                "untrained_reward": mean[0].tolist(),
                "just_trained_reward": np.diag(mean[1:]).tolist(),
                "final_reward": mean[-1].tolist(),
            }
        )
    )

    figure = plt.figure(figsize=(13, 5.2))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.36)
    trace = figure.add_subplot(grid[0, 0])
    interval = int(header["eval_every"])
    middle = (ticks - interval / 2) / 1e6
    for curve in curves:
        trace.plot(middle, curve, color=BLUE, linewidth=0.8, alpha=0.35)
    trace.plot(middle, np.mean(curves, axis=0), color=BLUE, linewidth=2)
    for switch in range(1, count):
        trace.axvline(switch * steps / 1e6, color=AXIS, linewidth=0.8)
    trace.set_xlim(0, count * steps / 1e6)
    trace.set_ylabel(f"Reward per tick ({interval:,}-tick intervals)")
    trace.set_xlabel("Lifetime tick (millions); a task switch at each line")
    trace.grid(axis="x", visible=False)
    trace.set_title(f"Online, during training (mean of {len(runs)} seeds, seeds faint)")
    names = trace.secondary_xaxis("top")
    names.set_xticks((np.arange(count) + 0.5) * steps / 1e6, tasks, rotation=30, ha="left")
    names.tick_params(length=0, labelsize=8)
    names.spines["top"].set_visible(False)
    heat = figure.add_subplot(grid[0, 1])
    title = f"Revisit, rescaled per task (forgetting {np.mean(forgetting):.2f})"
    _revisit(heat, np.mean(scaled, axis=0), tasks, title)
    heat.set_xlabel("Evaluated on task (0 untrained, 1 just trained)")
    figure.savefig(out)


def plot_throughput(paths: list[str], out: Path) -> None:
    """Training steps/s per seed: one row per report, lifetime or baseline."""
    rows: dict[str, tuple[str, list[float]]] = {}
    for path in paths:
        report = _load(path)
        if "measurement" in report:
            measurement = report["measurement"]
            label = f"Lifetime PPO, {measurement['batch']} lifetimes"
            rows[label] = (BLUE, measurement["steps_per_second_per_seed"])
        else:
            label = f"AsyncVectorEnv + PyTorch PPO, {report['num_envs']} workers"
            rows.setdefault(label, (ORANGE, []))[1].append(report["steps_per_second"])
    figure, axis = plt.subplots(figsize=(8.4, 0.62 * len(rows) + 1.3))
    every = [value for _, values in rows.values() for value in values]
    middle = float(np.sqrt(min(every) * max(every)))
    for y, (label, (color, values)) in enumerate(rows.items()):
        right = float(np.mean(values)) > middle
        axis.plot(
            values,
            [y] * len(values),
            "o",
            color=color,
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            alpha=0.85,
        )
        axis.annotate(
            f"{label}: {_si(float(np.mean(values)))} steps/s",
            (max(values) if right else min(values), y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="right" if right else "left",
            color=SECONDARY,
            fontsize=9,
        )
    axis.set_xscale("log")
    axis.set_ylim(len(rows) - 0.35, -0.75)
    axis.set_yticks([])
    axis.grid(axis="y", visible=False)
    axis.spines["left"].set_visible(False)
    axis.set_title("PPO training throughput on CartPole (one dot per seed)")
    axis.set_xlabel("Environment steps per second (log)")
    figure.savefig(out)


def plot_tuning(paths: list[str], out: Path) -> None:
    """GPU-hour ratio of a job array to one vmapped job, by sweep size and length."""
    reports = [_load(path) for path in paths]
    sizes = sorted({report["sweep"]["configs"] for report in reports})
    lengths = sorted({report["sweep"]["updates"] for report in reports})
    ratio = np.full((len(lengths), len(sizes)), np.nan)
    figure, axis = plt.subplots(figsize=(2.3 * len(sizes) + 1.6, 1.0 * len(lengths) + 1.5))
    for report in reports:
        row = lengths.index(report["sweep"]["updates"])
        column = sizes.index(report["sweep"]["configs"])
        ratio[row, column] = report["gpu_hours"]["ratio"]
    axis.imshow(np.log10(ratio), cmap=BLUES, vmin=0, vmax=3.2, aspect="auto")
    for report in reports:
        row = lengths.index(report["sweep"]["updates"])
        column = sizes.index(report["sweep"]["configs"])
        ink = "#ffffff" if ratio[row, column] > 100 else INK
        jobs, vmapped = report["job_array"]["total_seconds"], report["vmapped"]["seconds"]
        axis.text(
            column,
            row - 0.12,
            f"{ratio[row, column]:.0f}×",
            ha="center",
            va="center",
            color=ink,
            fontsize=14,
            fontweight="bold",
        )
        axis.text(
            column,
            row + 0.24,
            f"{_duration(jobs)} vs {vmapped:.0f} s",
            ha="center",
            va="center",
            color=ink,
            fontsize=8.5,
        )
    axis.set_xticks(np.arange(len(sizes)), [f"{size:,} configurations" for size in sizes])
    axis.set_yticks(np.arange(len(lengths)), [f"{length:,} updates" for length in lengths])
    axis.set_title("GPU time: per-config job array vs one vmapped job")
    axis.grid(visible=False)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.savefig(out)


def _duration(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def _si(value: float) -> str:
    return f"{value / 1e6:.2f}M" if value >= 1e6 else f"{value / 1e3:.1f}k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figure",
        choices=[
            "switching",
            "reset",
            "matrix",
            "lifetime",
            "reward-lifetime",
            "throughput",
            "tuning",
        ],
    )
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.figure == "switching":
        plot_switching(args.reports, args.out)
    elif args.figure == "tuning":
        plot_tuning(args.reports, args.out)
    elif args.figure == "throughput":
        plot_throughput(args.reports, args.out)
    elif args.figure == "lifetime":
        plot_lifetime(args.reports, args.out)
    elif args.figure == "reward-lifetime":
        plot_reward_lifetime(args.reports, args.out)
    elif args.figure == "reset":
        plot_reset(args.reports[0], args.out)
    else:
        plot_matrix(args.reports[0], args.out)


if __name__ == "__main__":
    main()
