#!/usr/bin/env bash
# Stop any trainer running on this workstation, leaving checkpoints intact.
#
# The trainer resumes from latest.pt, so this is a pause, not a loss. Run it
# before submitting Slurm jobs that target the same run directories.
set -uo pipefail

mapfile -t pids < <(
  ps -eo pid,comm,args --no-headers |
    awk '$2 ~ /^python/ && /gns\.cli\.train/ {print $1}'
)
mapfile -t supervisors < <(
  ps -eo pid,comm,args --no-headers |
    awk '$2 ~ /^bash/ && /experiments\/supervise\.sh/ {print $1}'
)

if [ ${#supervisors[@]} -gt 0 ]; then
  echo "stopping ${#supervisors[@]} supervisor(s): ${supervisors[*]}"
  kill "${supervisors[@]}" 2>/dev/null || true
fi
if [ ${#pids[@]} -eq 0 ]; then
  echo "no local trainer running"
  exit 0
fi
echo "stopping ${#pids[@]} trainer process(es): ${pids[*]}"
kill "${pids[@]}" 2>/dev/null || true
sleep 5
ps -eo pid,comm,args --no-headers | awk '$2 ~ /^python/ && /gns\.cli\.train/ {print "still alive:", $1}'
