"""The Slurm job scripts, run the way Slurm runs them.

Slurm copies a batch script into a per-job spool directory and executes it
there, so ``$0`` does not point into the repository and anything resolved
relative to it breaks. These tests reproduce that by copying each script to a
temporary directory before running it, with ``GNS_DRY_RUN`` so the trainer is
printed rather than started.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SLURM = REPO / "experiments" / "slurm"
SCRIPTS = ["table1.sbatch", "ablations.sbatch"]


def run_like_slurm(tmp_path: Path, script: str, task_id: int, runs: Path):
    """Copy the script out of the repository and run it, as Slurm would."""
    spool = tmp_path / "spool"
    spool.mkdir(exist_ok=True)
    copied = spool / "slurm_script"
    shutil.copy(SLURM / script, copied)

    environment = {
        **os.environ,
        "GNS_DRY_RUN": "1",
        "GNS_RUNS": str(runs),
        "SLURM_SUBMIT_DIR": str(REPO),
        "SLURM_ARRAY_TASK_ID": str(task_id),
    }
    environment.pop("GNS_REPO", None)
    return subprocess.run(
        ["bash", str(copied)],
        cwd=spool,
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs_from_a_spool_copy(tmp_path, script):
    result = run_like_slurm(tmp_path, script, 0, tmp_path / "runs")
    assert result.returncode == 0, result.stderr
    assert "No such file or directory" not in result.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_bash_syntax(script):
    result = subprocess.run(
        ["bash", "-n", str(SLURM / script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_table1_array_covers_every_dataset(tmp_path):
    seen = set()
    for task in range(3):
        result = run_like_slurm(tmp_path, "table1.sbatch", task, tmp_path / "runs")
        assert result.returncode == 0, result.stderr
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("DRY RUN")
        )
        seen.add(line.split("--dataset ")[1].split()[0])
    assert seen == {"Goop", "WaterRamps", "SandRamps"}


def test_ablation_array_covers_every_configuration_exactly_once(tmp_path):
    from gns.ablations import runs as ablation_runs

    expected = {run.name for run in ablation_runs()}
    seen: list[str] = []
    for task in range(6):
        result = run_like_slurm(tmp_path, "ablations.sbatch", task, tmp_path / "runs")
        assert result.returncode == 0, result.stderr
        seen += [
            line.split()[2]
            for line in result.stdout.splitlines()
            if line.startswith("===")
        ]
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen))
