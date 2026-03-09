# ray-serve-bench

Benchmark Ray Serve vs vLLM direct serving using `vllm bench serve`. Isolates serving-layer overhead without any pipeline framework.

## Setup

```bash
uv venv -p 3.12 .venv
source .venv/bin/activate
uv sync --all-extras --all-groups
pre-commit install
```

## Usage

```bash
# Run all configs (DP4 + TP4, ray-serve + vllm-direct)
bash run.sh --gpus 4,5,6,7

# Override concurrency and number of runs
bash run.sh --gpus 0,1,2,3 --concurrency 4 --runs 3

# Filter to specific configs
bash run.sh --gpus 0,1,2,3 --filter dp4

# Set extra env vars (e.g. HAProxy)
bash run.sh --gpus 0,1,2,3 --env "RAY_SERVE_ENABLE_HA_PROXY=1,RAY_SERVE_THROUGHPUT_OPTIMIZED=1"
```

## Full concurrency sweep

`sweep.sh` runs `run.sh` across multiple concurrency levels (4, 64, 512) back-to-back:

```bash
# Run in a tmux session (takes a while — ~6 server startups per concurrency level)
tmux new -s sweep
bash sweep.sh 2>&1 | tee results/full_sweep.log
```

Edit the `for c in 4 64 512` loop in `sweep.sh` to change which concurrency levels are tested, and the `--gpus` flag to target different GPUs.

## Results

Results are written to `results/c<concurrency>_<timestamp>/`:

```
results/c64_2026-03-07T14:30:00+00:00/
├── summary.txt                        # all entries sorted by throughput
├── ray_serve_dp4_c64/
│   ├── summary.txt                    # human-readable per-entry summary
│   ├── metrics.json
│   ├── params.json
│   └── vllm_bench_raw/
└── vllm_direct_async_dp4_c64/
    └── ...
```

## Running a single entry directly

```bash
python bench_serve.py \
    --model-type ray-serve \
    --model-id google/gemma-3-12b-it \
    --engine-kwargs '{"tensor_parallel_size": 1}' \
    --autoscaling-config '{"min_replicas": 4, "max_replicas": 4}' \
    --max-concurrency 64 \
    --results-dir results/my_test
```
