#!/usr/bin/env bash
# Fetch the released GNS datasets into $GNS_RAW_ROOT.
#
#   bash scripts/download_datasets.sh WaterRamps SandRamps Goop
#
# The files are the authors' own TFRecords, so they are the ground truth for a
# replication: a dataset re-derived from someone else's conversion can differ in
# ways that are invisible until the numbers do not match.
set -euo pipefail

: "${GNS_RAW_ROOT:?set GNS_RAW_ROOT to where the TFRecords should land}"
BUCKET=https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets

for dataset in "$@"; do
  mkdir -p "$GNS_RAW_ROOT/$dataset"
  for file in metadata.json train.tfrecord valid.tfrecord test.tfrecord; do
    target="$GNS_RAW_ROOT/$dataset/$file"
    echo "fetching $dataset/$file"
    curl -fL --retry 3 -C - -o "$target" "$BUCKET/$dataset/$file"
  done
done
