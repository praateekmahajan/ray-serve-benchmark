"""Standalone inference server utilities (no nemo_curator dependency).

Supports two model types:
  - ray-serve: vLLM behind Ray Serve (OpenAI-compatible)
  - vllm-direct: standalone vLLM OpenAI server as a subprocess
"""

import atexit
import http
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

_VLLM_PORT = 8000
_HEALTH_TIMEOUT_S = 300


# ---------------------------------------------------------------------------
# Port utility
# ---------------------------------------------------------------------------

def get_free_port(start_port: int = _VLLM_PORT) -> int:
    """Find a free port starting from start_port."""
    for port in range(start_port, 65535):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    msg = f"No free port found starting from {start_port}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Server handle
# ---------------------------------------------------------------------------

@dataclass
class InferenceServer:
    """Handle returned by start_inference_server.

    Use ``endpoint`` and ``api_key`` to connect.
    Call ``stop()`` when done.
    """

    endpoint: str
    api_key: str = "none"
    startup_s: float = 0.0
    _stop_fn: Any = field(default=None, repr=False)

    def stop(self) -> None:
        if self._stop_fn is not None:
            self._stop_fn()
            self._stop_fn = None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def _wait_for_model_ready(port: int, timeout_s: int = _HEALTH_TIMEOUT_S) -> None:
    """Poll /v1/models until the server is ready."""
    models_url = f"http://localhost:{port}/v1/models"
    for attempt in range(timeout_s):
        try:
            resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
            if resp.status == http.HTTPStatus.OK:
                logger.info(f"Model server ready after {attempt + 1}s")
                return
        except Exception:
            pass
        time.sleep(1)
    msg = f"Model server did not become ready within {timeout_s}s"
    raise TimeoutError(msg)


# ---------------------------------------------------------------------------
# Engine kwargs → CLI args
# ---------------------------------------------------------------------------

def _engine_kwargs_to_vllm_args(engine_kwargs: dict[str, Any]) -> list[str]:
    """Convert engine_kwargs dict to vLLM CLI arguments."""
    args = []
    for key, value in engine_kwargs.items():
        cli_key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(cli_key)
        elif isinstance(value, (dict, list)):
            args.extend([cli_key, json.dumps(value)])
        else:
            args.extend([cli_key, str(value)])
    return args


# ---------------------------------------------------------------------------
# Ray Serve server
# ---------------------------------------------------------------------------

def _start_ray_serve(
    model_id: str,
    engine_kwargs: dict[str, Any],
    autoscaling_config: dict[str, Any],
    verbose: bool = False,
) -> InferenceServer:
    """Start Ray Serve with vLLM backend."""
    from ray import serve
    from ray.serve.llm import LLMConfig, build_openai_app

    os.environ.setdefault("NVIDIA_API_KEY", "none")

    quiet_env = None
    if not verbose:
        quiet_env = {
            "env_vars": {
                "VLLM_LOGGING_LEVEL": "WARNING",
                "RAY_SERVE_LOG_TO_STDERR": "0",
            },
        }

    llm_config = LLMConfig(
        model_loading_config={
            "model_id": model_id,
            "model_source": model_id,
        },
        deployment_config={"autoscaling_config": autoscaling_config},
        engine_kwargs=engine_kwargs,
        runtime_env=quiet_env,
    )

    build_args: dict[str, Any] = {"llm_configs": [llm_config]}
    if quiet_env:
        build_args["ingress_deployment_config"] = {
            "ray_actor_options": {"runtime_env": quiet_env},
        }

    port = get_free_port(_VLLM_PORT)
    app = build_openai_app(build_args)

    logger.info(f"Starting Ray Serve on port {port}: model={model_id}, engine_kwargs={engine_kwargs}")

    serve.start(http_options={"port": port})

    from ray.serve.schema import LoggingConfig

    logging_config = None
    if not verbose:
        logging_config = LoggingConfig(log_level="WARNING", enable_access_log=False)

    t0 = time.perf_counter()
    try:
        serve.run(app, name="default", blocking=False, logging_config=logging_config)
        _wait_for_model_ready(port)
    except Exception:
        try:
            serve.shutdown()
        except Exception:
            pass
        raise

    startup_s = time.perf_counter() - t0
    logger.info(f"Ray Serve started in {startup_s:.2f}s at http://localhost:{port}/v1")

    def _stop() -> None:
        logger.info("Shutting down Ray Serve")
        try:
            serve.shutdown()
        except Exception:
            logger.debug("serve.shutdown() failed (cluster may already be gone)")

    atexit.register(_stop)

    return InferenceServer(
        endpoint=f"http://localhost:{port}/v1",
        startup_s=startup_s,
        _stop_fn=_stop,
    )


# ---------------------------------------------------------------------------
# vLLM direct server
# ---------------------------------------------------------------------------

def _start_vllm_direct(
    model_id: str,
    engine_kwargs: dict[str, Any],
    log_dir: Path | None = None,
) -> InferenceServer:
    """Start standalone vLLM OpenAI-compatible server as a subprocess."""
    port = get_free_port(_VLLM_PORT)

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--host",
        "localhost",
        "--port",
        str(port),
        "--disable-log-requests",
        *_engine_kwargs_to_vllm_args(engine_kwargs),
    ]

    # Strip RAY_ env vars so vLLM starts its own process group for TP.
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAY_")}

    log_parent = Path(log_dir) if log_dir else Path("/tmp/ray-serve-bench")
    log_parent.mkdir(parents=True, exist_ok=True)
    log_file = log_parent / "vllm_server.log"

    logger.info(f"Starting vLLM server on port {port}: {' '.join(cmd)}")
    logger.info(f"vLLM server log: {log_file}")

    log_fh = log_file.open("w")
    process = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True, env=env,
    )

    t0 = time.perf_counter()
    _wait_for_model_ready(port)
    startup_s = time.perf_counter() - t0
    logger.info(f"vLLM server started in {startup_s:.2f}s on port {port}")

    def _stop() -> None:
        logger.info("Stopping vLLM server...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        log_fh.close()

    return InferenceServer(
        endpoint=f"http://localhost:{port}/v1",
        startup_s=startup_s,
        _stop_fn=_stop,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_inference_server(
    model_type: str,
    model_id: str,
    engine_kwargs: dict[str, Any] | None = None,
    autoscaling_config: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
    verbose: bool = False,
) -> InferenceServer:
    """Start an inference server and return a handle.

    Args:
        model_type: "ray-serve" or "vllm-direct"
        model_id: HuggingFace model ID
        engine_kwargs: vLLM engine kwargs (tensor_parallel_size, etc.)
        autoscaling_config: Ray Serve autoscaling config (only for ray-serve)
        log_dir: Directory for server logs
        verbose: Keep full logging if True

    Returns:
        InferenceServer handle. Call .stop() when done.
    """
    engine_kwargs = engine_kwargs or {}
    autoscaling_config = autoscaling_config or {"min_replicas": 1, "max_replicas": 1}

    if model_type == "ray-serve":
        return _start_ray_serve(model_id, engine_kwargs, autoscaling_config, verbose=verbose)
    elif model_type == "vllm-direct":
        return _start_vllm_direct(model_id, engine_kwargs, log_dir=log_dir)
    else:
        msg = f"Unknown model_type: {model_type}. Must be 'ray-serve' or 'vllm-direct'."
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def write_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Write benchmark results to params.json and metrics.json."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if "params" in results:
        params_path = output_path / "params.json"
        params_data = {}
        if params_path.exists():
            params_data = json.loads(params_path.read_text())
        params_data.update(results["params"])
        params_path.write_text(json.dumps(params_data, indent=2, default=str))

    if "metrics" in results:
        metrics_path = output_path / "metrics.json"
        metrics_data = {}
        if metrics_path.exists():
            metrics_data = json.loads(metrics_path.read_text())
        metrics_data.update(results["metrics"])
        metrics_path.write_text(json.dumps(metrics_data, indent=2, default=str))
