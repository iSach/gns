"""Build the replication figures from evaluated runs.

    gns-figures table1     --results <dir>  --out figures/
    gns-figures error-time --results <dir>  --out figures/
    gns-figures ablations  --results <dir>  --out figures/

Each subcommand reads the ``result_*.json`` files written by ``gns-evaluate``
and plots them against the published values in :mod:`gns.paper`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gns import paper  # noqa: E402

PAPER_COLOR = "0.55"
OURS_COLOR = "#c1272d"


def load_results(directory: Path, pattern: str = "**/result_test_best.json") -> dict:
    """Collect evaluation results keyed by the name of their run directory."""
    results = {}
    for path in sorted(directory.glob(pattern)):
        results[path.parent.name] = json.loads(path.read_text())
    return results


def _bar_panel(ax, labels, ours, reference, ylabel, title, baseline=None):
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, reference, width, color=PAPER_COLOR, label="paper")
    ax.bar(x + width / 2, ours, width, color=OURS_COLOR, label="this reimplementation")
    if baseline is not None:
        for position, value in zip(x, baseline):
            if value is None:
                continue
            ax.plot(
                [position - 0.5, position + 0.5], [value, value],
                color="black", linestyle="--", linewidth=1.0,
                label="no-model baseline" if position == 0 else None,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def figure_table1(results: dict, out: Path, convention: str) -> Path:
    """Table 1: one-step and rollout MSE beside the published values."""
    key = "free_particles" if convention == "free" else "all_particles"
    rows = [
        (r["dataset"], r) for r in results.values() if r["dataset"] in paper.TABLE_1
    ]
    rows.sort(key=lambda item: item[0])
    labels = [name for name, _ in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    _bar_panel(
        axes[0],
        labels,
        [r[f"one_step_mse_{key}"] for _, r in rows],
        [paper.one_step_mse(name) for name in labels],
        "one-step MSE",
        "One-step position error",
        baseline=[
            (r.get("constant_velocity_one_step_mse") or {}).get(key)
            for _, r in rows
        ],
    )
    axes[0].legend(fontsize=7, frameon=False)
    _bar_panel(
        axes[1],
        labels,
        [r[f"rollout_mse_{key}"] for _, r in rows],
        [paper.rollout_mse(name) for name in labels],
        "rollout MSE",
        "Full-rollout position error",
    )
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"Table 1 replication ({convention} particles, full test split)", fontsize=10
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_error_vs_time(results: dict, out: Path) -> Path:
    """Figure C.3: rollout error as a function of rollout step."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.8, max(len(results), 1)))
    for color, result in zip(colors, sorted(results.values(), key=lambda r: r["dataset"])):
        curve = np.asarray(result["per_step_rollout_mse"])
        ax.plot(np.arange(1, len(curve) + 1), curve, color=color,
                label=result["dataset"], linewidth=1.4)
        ax.axhline(
            paper.rollout_mse(result["dataset"]),
            color=color, linestyle=":", linewidth=1.0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("rollout step", fontsize=9)
    ax.set_ylabel("MSE against ground truth", fontsize=9)
    ax.set_title(
        "Rollout error over time (dotted: paper's trajectory average)", fontsize=9
    )
    ax.grid(which="both", linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _ablation_label(axis: str, value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def figure_ablations(results: dict, out: Path, convention: str) -> Path:
    """Figure 4(a-j): each ablation axis against the default architecture."""
    key = "free_particles" if convention == "free" else "all_particles"
    grouped: dict[str, dict] = {axis: {} for axis in paper.ABLATION_AXES}
    default = None
    for name, result in results.items():
        if not name.startswith("ablation_"):
            continue
        axis = result.get("ablation_axis")
        if axis == "default":
            default = result
        elif axis in grouped:
            grouped[axis][result["ablation_value"]] = result
    # The default architecture is the default point of several axes; it is
    # trained once and reused rather than retrained per axis.
    if default is not None:
        for axis, spec in paper.ABLATION_AXES.items():
            grouped[axis].setdefault(spec["default"], default)

    axes_with_data = [a for a in paper.ABLATION_AXES if grouped[a]]
    if not axes_with_data:
        raise SystemExit("No ablation results found; run experiments/ablations.sh")

    fig, panels = plt.subplots(
        2, len(axes_with_data), figsize=(2.3 * len(axes_with_data), 4.6),
        squeeze=False, sharey="row",
    )
    for column, axis in enumerate(axes_with_data):
        spec = paper.ABLATION_AXES[axis]
        values = [v for v in spec["values"] if v in grouped[axis]]
        labels = [_ablation_label(axis, v) for v in values]
        colors = [
            OURS_COLOR if v == spec["default"] else PAPER_COLOR for v in values
        ]
        for row, metric in enumerate(
            [f"one_step_mse_{key}", f"rollout_mse_{key}"]
        ):
            ax = panels[row][column]
            ax.bar(
                np.arange(len(values)),
                [grouped[axis][v][metric] for v in values],
                color=colors,
            )
            ax.set_xticks(np.arange(len(values)))
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
            ax.set_yscale("log")
            ax.grid(axis="y", which="both", linewidth=0.3, alpha=0.4)
            ax.set_axisbelow(True)
            if column == 0:
                ax.set_ylabel(
                    "one-step MSE" if row == 0 else "rollout MSE", fontsize=9
                )
        panels[0][column].set_title(spec["label"], fontsize=8)
    fig.suptitle(
        "Figure 4 replication on Goop (red: default architecture)", fontsize=10
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def write_table(results: dict, out: Path) -> Path:
    """A markdown table of both particle conventions beside the paper's values."""
    header = (
        "| dataset | step | one-step (free) | one-step (all) | paper one-step "
        "| rollout (free) | rollout (all) | paper rollout |"
    )
    lines = [header, "| --- " * 8 + "|"]
    for result in sorted(results.values(), key=lambda r: r["dataset"]):
        name = result["dataset"]
        known = name in paper.TABLE_1
        lines.append(
            f"| {name} | {result['step']:,} "
            f"| {result['one_step_mse_free_particles']:.3e} "
            f"| {result['one_step_mse_all_particles']:.3e} "
            f"| {paper.one_step_mse(name):.3e} " if known else
            f"| {name} | {result['step']:,} "
            f"| {result['one_step_mse_free_particles']:.3e} "
            f"| {result['one_step_mse_all_particles']:.3e} | - "
        )
        lines[-1] += (
            f"| {result['rollout_mse_free_particles']:.3e} "
            f"| {result['rollout_mse_all_particles']:.3e} "
            + (f"| {paper.rollout_mse(name):.3e} |" if known else "| - |")
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figure", choices=["table1", "error-time", "ablations", "all"]
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument(
        "--convention",
        choices=["free", "all"],
        default="free",
        help="Average the MSE over non-obstacle particles only, or over all.",
    )
    parser.add_argument("--pattern", default="**/result_test_best.json")
    args = parser.parse_args()

    results = load_results(args.results, args.pattern)
    if not results:
        raise SystemExit(f"No results matching {args.pattern} under {args.results}")

    wanted = (
        ["table1", "error-time", "ablations"] if args.figure == "all" else [args.figure]
    )
    for figure in wanted:
        try:
            if figure == "table1":
                path = figure_table1(results, args.out / "table1.png", args.convention)
                print(f"wrote {path}")
                print(f"wrote {write_table(results, args.out / 'table1.md')}")
            elif figure == "error-time":
                print(f"wrote {figure_error_vs_time(results, args.out / 'error_vs_time.png')}")
            else:
                path = figure_ablations(
                    results, args.out / "ablations.png", args.convention
                )
                print(f"wrote {path}")
        except SystemExit as error:
            if args.figure == "all":
                print(f"skipped {figure}: {error}")
            else:
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
