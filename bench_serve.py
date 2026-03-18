"""Benchmark Ray Serve vs vLLM direct using vllm bench serve.

Isolates serving-layer overhead by removing any pipeline framework.
Uses vLLM's built-in benchmark_serving tool to send standardized requests
to an inference endpoint and collect latency/throughput metrics.

Usage:
    python bench_serve.py \
        --model-type ray-serve \
        --model-id google/gemma-3-12b-it \
        --engine-kwargs '{"tensor_parallel_size": 1}' \
        --autoscaling-config '{"min_replicas": 4, "max_replicas": 4}' \
        --max-concurrency 64 \
        --num-prompts 1000 \
        --results-dir results/ray_serve_dp4_c64
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from inference_server import start_inference_server
from ray_client import RayCluster
from utils import write_results, write_summary


def run_vllm_bench(
    endpoint: str,
    model_id: str,
    max_concurrency: int,
    num_prompts: int,
    input_len: int,
    output_len: int,
    result_dir: Path,
    backend: str = "openai-chat",
    dataset_name: str = "random",
) -> dict[str, Any]:
    """Run vllm bench serve and return parsed results."""
    result_dir.mkdir(parents=True, exist_ok=True)
    result_filename = "vllm_bench_result.json"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--base-url",
        endpoint.rsplit("/v1", 1)[0],
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        model_id,
        "--backend",
        backend,
        "--dataset-name",
        dataset_name,
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(max_concurrency),
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
        "--ignore-eos",
        "--disable-tqdm",
        "--temperature",
        "0",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        logger.error(f"vllm bench serve failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        msg = f"vllm bench serve exited with code {result.returncode}"
        raise RuntimeError(msg)

    logger.info(f"vllm bench serve stdout:\n{result.stdout}")

    result_file = result_dir / result_filename
    return json.loads(result_file.read_text())


# Metrics to extract from vllm bench serve results
_METRIC_KEYS = [
    "duration",
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "std_e2el_ms",
    "p99_e2el_ms",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark serving layer with vllm bench serve")
    parser.add_argument("--results-dir", required=True, help="Directory to write results")
    parser.add_argument("--model-type", required=True, choices=["ray-serve", "vllm-direct"])
    parser.add_argument("--model-id", default="google/gemma-3-12b-it")
    parser.add_argument("--engine-kwargs", type=json.loads, default="{}")
    parser.add_argument("--autoscaling-config", type=json.loads, default='{"min_replicas": 1, "max_replicas": 1}')
    parser.add_argument("--ingress-config", type=json.loads, default=None, help="Ingress deployment config JSON")
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--backend", default="openai-chat")
    parser.add_argument("--dataset-name", default="random")

    args = parser.parse_args()
    output_path = Path(args.results_dir)

    logger.info(f"=== Bench Serve Benchmark ===\n{vars(args)}")

    params = vars(args).copy()
    params["engine_kwargs"] = json.dumps(params["engine_kwargs"])
    params["autoscaling_config"] = json.dumps(params["autoscaling_config"])
    if params.get("ingress_config") is not None:
        params["ingress_config"] = json.dumps(params["ingress_config"])

    result_dict: dict[str, Any] = {"params": params, "metrics": {"is_success": False}}

    try:
        log_dir = output_path / "logs"
        cluster = None

        # Start Ray cluster for ray-serve model type
        if args.model_type == "ray-serve":
            t0_cluster = time.perf_counter()
            cluster = RayCluster(
                temp_dir=f"/tmp/ray_{uuid.uuid4().hex[:8]}",
                log_file=log_dir / "ray_cluster.log",
            )
            cluster.start()
            cluster_startup_s = time.perf_counter() - t0_cluster
            logger.info(f"Ray cluster started in {cluster_startup_s:.1f}s")

        try:
            t0_server = time.perf_counter()
            server = start_inference_server(
                args.model_type,
                args.model_id,
                args.engine_kwargs,
                args.autoscaling_config,
                ingress_config=args.ingress_config,
                log_dir=log_dir,
                cluster=cluster,
            )
            server_startup_s = time.perf_counter() - t0_server
            logger.info(f"Server startup took {server_startup_s:.1f}s (model ready: {server.startup_s:.1f}s)")

            try:
                t0_bench = time.perf_counter()
                bench_results = run_vllm_bench(
                    endpoint=server.endpoint,
                    model_id=args.model_id,
                    max_concurrency=args.max_concurrency,
                    num_prompts=args.num_prompts,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    result_dir=output_path / "vllm_bench_raw",
                    backend=args.backend,
                    dataset_name=args.dataset_name,
                )
                bench_duration_s = time.perf_counter() - t0_bench
                logger.info(f"Benchmark took {bench_duration_s:.1f}s")
            finally:
                server.stop()
        finally:
            if cluster is not None:
                cluster.stop()

        metrics = {
            "is_success": True,
            "serve_startup_s": server.startup_s,
            "model_type": args.model_type,
            "model_id": args.model_id,
            "num_prompts": args.num_prompts,
            "max_concurrency": args.max_concurrency,
            "input_len": args.input_len,
            "output_len": args.output_len,
        }

        for key in _METRIC_KEYS:
            if key in bench_results:
                metrics[key] = bench_results[key]

        result_dict["metrics"] = metrics
        logger.success(f"Benchmark completed: throughput={metrics.get('request_throughput', 'N/A')} req/s")
        return 0
    except Exception:
        logger.exception("Benchmark failed")
        return 1
    finally:
        write_results(result_dict, output_path)
        write_summary(result_dict, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
