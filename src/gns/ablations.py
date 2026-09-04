"""The Figure 4 ablation grid, in one place.

The local launcher and the Slurm array both read this, so there is one list of
runs rather than two that can drift apart.

    python -m gns.ablations --list        # one run per line, tab separated
    python -m gns.ablations --count

The default architecture is the default point of four separate axes, so it is
trained once as ``ablation_default`` and the figure builder reuses it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AblationRun:
    """One training run of the sweep."""

    axis: str
    label: str
    json_value: str  # what lands in ablation.json, parsed as JSON
    flags: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        if self.axis == "default":
            return "ablation_default"
        return f"ablation_{self.axis}_{self.label.replace('.', 'p')}"


def _noise_last_step(per_step: float) -> float:
    """The paper's Figure 4 axis is per step; the trainer wants the last step."""
    return per_step * 5**0.5


def runs() -> list[AblationRun]:
    sweep = [AblationRun("default", "default", "null")]
    for steps in (1, 2, 5, 15):
        sweep.append(
            AblationRun(
                "message_passing_steps", str(steps), str(steps),
                ["--message-passing-steps", str(steps)],
            )
        )
    sweep.append(
        AblationRun("shared_processor", "true", "true", ["--shared-processor"])
    )
    for radius in (0.003, 0.007, 0.011, 0.02, 0.03):
        sweep.append(
            AblationRun(
                "connectivity_radius", f"{radius:g}", f"{radius:g}",
                ["--connectivity-radius", f"{radius:g}"],
            )
        )
    for sigma in (0.0, 3e-5, 1e-4, 1e-3, 3e-3):
        sweep.append(
            AblationRun(
                "noise_std_per_step", f"{sigma:g}", f"{sigma:g}",
                ["--noise-std", repr(_noise_last_step(sigma))],
            )
        )
    sweep.append(
        AblationRun(
            "use_relative_positions", "false", "false", ["--absolute-encoder"]
        )
    )
    return sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument(
        "--index", type=int, default=None, help="Print just this run's line."
    )
    args = parser.parse_args()

    sweep = runs()
    if args.count:
        print(len(sweep))
        return 0
    selected = sweep if args.index is None else [sweep[args.index]]
    for run in selected:
        print("\t".join([run.name, run.axis, run.json_value, *run.flags]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
