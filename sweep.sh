#!/usr/bin/env bash
set -euo pipefail

cd /raid/praateekm/ray-serve-bench
source .venv/bin/activate

mkdir -p results

for c in 4 64 512; do
    echo "========== Concurrency: ${c} =========="
    bash run.sh --gpus 4,5,6,7 --concurrency "$c"
    echo ""
done

echo "========== Full sweep complete =========="
