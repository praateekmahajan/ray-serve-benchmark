#!/usr/bin/env bash
# Run vllm bench serve across multiple configurations.
#
# Each config is: name|model_type|engine_kwargs_json|autoscaling_config_json
#
# Results are organized as:
#   results/<timestamp>/<entry_name>/metrics.json
#
# Usage:
#   bash run.sh                          # run all configs
#   bash run.sh --gpus 4,5,6,7           # select GPUs
#   bash run.sh --concurrency 4          # override concurrency
#   bash run.sh --runs 3                 # multiple runs per config
#   bash run.sh --filter "dp4"           # only run configs matching substring
#   bash run.sh --env "RAY_SERVE_ENABLE_HA_PROXY=1,RAY_SERVE_THROUGHPUT_OPTIMIZED=1"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
GPUS=""
MAX_CONCURRENCY=64
NUM_PROMPTS=1000
INPUT_LEN=512
OUTPUT_LEN=256
MODEL_ID="google/gemma-3-12b-it"
RESULTS_DIR="results"
NUM_RUNS=1
FILTER=""
EXTRA_ENV=""
CLEANUP_WAIT=30

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift 2 ;;
        --concurrency|-c) MAX_CONCURRENCY="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --model) MODEL_ID="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --runs) NUM_RUNS="$2"; shift 2 ;;
        --filter) FILTER="$2"; shift 2 ;;
        --env) EXTRA_ENV="$2"; shift 2 ;;
        --cleanup-wait) CLEANUP_WAIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# Set extra environment variables (comma-separated KEY=VALUE pairs)
if [[ -n "$EXTRA_ENV" ]]; then
    IFS=',' read -ra ENV_PAIRS <<< "$EXTRA_ENV"
    for pair in "${ENV_PAIRS[@]}"; do
        export "${pair?}"
        echo "  Set env: ${pair}"
    done
fi

# ──────────────────────────────────────────────────────────────────────
# Configuration table
#
# Format: name|model_type|engine_kwargs|autoscaling_config
#
# model_type: ray-serve | vllm-direct
# engine_kwargs: JSON dict passed to vLLM engine
# autoscaling_config: JSON dict for Ray Serve autoscaling (ignored for vllm-direct)
# ──────────────────────────────────────────────────────────────────────

declare -a CONFIGS=(
    # ── DP4 (4 replicas on 4 GPUs, TP=1) ──
    "ray_serve_dp4|ray-serve|{\"tensor_parallel_size\": 1, \"limit_mm_per_prompt\": {\"image\": 0}}|{\"min_replicas\": 4, \"max_replicas\": 4}"
    "vllm_direct_async_dp4|vllm-direct|{\"tensor_parallel_size\": 1, \"data_parallel_size\": 4, \"limit_mm_per_prompt\": {\"image\": 0}}|"
    "vllm_direct_sync_dp4|vllm-direct|{\"tensor_parallel_size\": 1, \"data_parallel_size\": 4, \"limit_mm_per_prompt\": {\"image\": 0}, \"no_async_scheduling\": true}|"

    # ── TP4 (1 replica across 4 GPUs, TP=4) ──
    "ray_serve_tp4|ray-serve|{\"tensor_parallel_size\": 4, \"limit_mm_per_prompt\": {\"image\": 0}}|{\"min_replicas\": 1, \"max_replicas\": 1}"
    "vllm_direct_async_tp4|vllm-direct|{\"tensor_parallel_size\": 4, \"limit_mm_per_prompt\": {\"image\": 0}}|"
    "vllm_direct_sync_tp4|vllm-direct|{\"tensor_parallel_size\": 4, \"limit_mm_per_prompt\": {\"image\": 0}, \"no_async_scheduling\": true}|"
)

# ──────────────────────────────────────────────────────────────────────

# Filter configs if requested
FILTERED_CONFIGS=()
for cfg in "${CONFIGS[@]}"; do
    cfg_name="${cfg%%|*}"
    if [[ -z "$FILTER" ]] || [[ "$cfg_name" == *"$FILTER"* ]]; then
        FILTERED_CONFIGS+=("$cfg")
    fi
done

TOTAL=${#FILTERED_CONFIGS[@]}

# Create timestamped run directory
RUN_TIMESTAMP="$(date -Iseconds)"
RUN_DIR="${RESULTS_DIR}/c${MAX_CONCURRENCY}_${RUN_TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "=== Ray Serve Bench ==="
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Model: ${MODEL_ID}"
echo "Concurrency: ${MAX_CONCURRENCY}"
echo "Prompts: ${NUM_PROMPTS} (input=${INPUT_LEN}, output=${OUTPUT_LEN})"
echo "Configs: ${TOTAL} x ${NUM_RUNS} run(s)"
echo "Run dir: ${RUN_DIR}"
echo ""

RUN_IDX=0
for config_str in "${FILTERED_CONFIGS[@]}"; do
    IFS='|' read -r config_name model_type engine_kwargs autoscaling_config <<< "$config_str"

    for run in $(seq 1 "$NUM_RUNS"); do
        RUN_IDX=$((RUN_IDX + 1))

        if [[ "$NUM_RUNS" -gt 1 ]]; then
            entry_name="${config_name}_c${MAX_CONCURRENCY}_run${run}"
        else
            entry_name="${config_name}_c${MAX_CONCURRENCY}"
        fi

        result_path="${RUN_DIR}/${entry_name}"
        echo "=== [${RUN_IDX}/$((TOTAL * NUM_RUNS))] ${entry_name} ==="
        echo "  Type: ${model_type}"
        echo "  Results: ${result_path}"
        echo "  Start: $(date)"

        bench_args=(
            --results-dir="${result_path}"
            --model-type="${model_type}"
            --model-id="${MODEL_ID}"
            --engine-kwargs="${engine_kwargs}"
            --max-concurrency="${MAX_CONCURRENCY}"
            --num-prompts="${NUM_PROMPTS}"
            --input-len="${INPUT_LEN}"
            --output-len="${OUTPUT_LEN}"
        )
        if [[ -n "$autoscaling_config" ]]; then
            bench_args+=(--autoscaling-config="${autoscaling_config}")
        fi

        python bench_serve.py "${bench_args[@]}"

        echo "  End: $(date)"

        # Wait between runs for GPU memory cleanup
        if [[ "$RUN_IDX" -lt $((TOTAL * NUM_RUNS)) ]]; then
            echo "  Waiting ${CLEANUP_WAIT}s for GPU cleanup..."
            sleep "$CLEANUP_WAIT"
        fi

        echo ""
    done
done

# Generate run-level summary sorted by throughput (fastest first)
echo "=== Generating run summary ==="
python -c "
import json, sys
from pathlib import Path

run_dir = Path('${RUN_DIR}')
rows = []
for metrics_file in sorted(run_dir.glob('*/metrics.json')):
    entry = metrics_file.parent.name
    m = json.loads(metrics_file.read_text())
    if not m.get('is_success'):
        rows.append((entry, 0, 'FAILED', '', '', ''))
        continue
    rps = m.get('request_throughput', 0)
    ttft = m.get('mean_ttft_ms', 0)
    tpot = m.get('mean_tpot_ms', 0)
    e2el = m.get('mean_e2el_ms', 0)
    rows.append((entry, rps, f'{rps:.2f}', f'{ttft:.1f}', f'{tpot:.1f}', f'{e2el:.1f}'))

rows.sort(key=lambda r: -r[1])

lines = []
lines.append(f'Run: {run_dir.name}')
lines.append(f'Model: ${MODEL_ID}')
lines.append(f'Concurrency: ${MAX_CONCURRENCY}  Prompts: ${NUM_PROMPTS}  Input: ${INPUT_LEN}  Output: ${OUTPUT_LEN}')
lines.append('')
hdr = f'{\"Entry\":<45s} {\"req/s\":>8s} {\"TTFT\":>8s} {\"TPOT\":>8s} {\"E2EL\":>8s}'
lines.append(hdr)
lines.append('-' * len(hdr))
for entry, _, rps_s, ttft_s, tpot_s, e2el_s in rows:
    lines.append(f'{entry:<45s} {rps_s:>8s} {ttft_s:>8s} {tpot_s:>8s} {e2el_s:>8s}')
lines.append('')

summary = '\n'.join(lines)
print(summary)
(run_dir / 'summary.txt').write_text(summary)
"

echo ""
echo "=== All runs complete ==="
echo "Results: ${RUN_DIR}"
echo "Summary: ${RUN_DIR}/summary.txt"
