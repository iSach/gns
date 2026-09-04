#!/usr/bin/env bash
# Keep one training run alive.
#
#   setsid nohup bash experiments/supervise.sh <run-dir> <trainer args...> &
#
# Long runs on this box have been killed from outside the process more than
# once, with no traceback and the checkpoint intact.  The trainer resumes from
# `latest.pt`, so a supervisor that relaunches it turns a kill into a pause.  It
# exits for good once the run reaches its step budget, and gives up after a
# burst of immediate failures so a real crash does not spin.
set -uo pipefail
cd "$(dirname "$0")/.."

run_dir=$1; shift
PYTHON=${PYTHON:-.venv/bin/python}
: "${GNS_SUPERVISE_MAX_RESTARTS:=50}"

for attempt in $(seq 0 "$GNS_SUPERVISE_MAX_RESTARTS"); do
  started=$(date +%s)
  "$PYTHON" -m gns.cli.train --run-dir "$run_dir" "$@" && exit 0
  elapsed=$(( $(date +%s) - started ))
  if [ "$elapsed" -lt 120 ]; then
    echo "supervisor: $run_dir died after ${elapsed}s, treating as a real failure"
    exit 1
  fi
  echo "supervisor: $run_dir died after ${elapsed}s, restarting (attempt $((attempt + 1)))"
  sleep 10
done
echo "supervisor: $run_dir exceeded $GNS_SUPERVISE_MAX_RESTARTS restarts"
exit 1
