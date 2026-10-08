#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "usage: $0 PYTHON WORKER CONFIG OUTPUT RESULT MIN_FREE_MB" >&2
  exit 2
fi

python_bin="$1"
worker="$2"
config="$3"
output="$4"
result="$5"
min_free_mb="$6"

mkdir -p "$output"
echo "queued_at=$(date -Is)" | tee "$output/launch-state.log"
while true; do
  if /usr/bin/nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      | awk -v minimum="$min_free_mb" 'NF && ($1 + 0) < minimum {bad=1} END {exit bad}'; then
    echo "starting_at=$(date -Is)" | tee -a "$output/launch-state.log"
    exec "$python_bin" "$worker" \
      --config "$config" \
      --output "$output" \
      --result "$result" \
      --steps 1
  fi
  sleep 30
done
