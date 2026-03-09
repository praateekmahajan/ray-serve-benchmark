"""Inference server management for Ray Serve and vLLM direct.

Supports two model types:
  - ray-serve: Ray cluster + Ray Serve + vLLM backend
  - vllm-direct: standalone vLLM OpenAI-compatible server subprocess
"""

import atexit
import contextlib
import http
import json
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ray_client import RayCluster, get_free_port

_VLLM_PORT = 8000
_HEALTH_TIMEOUT_S = 300


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
            with urllib.request.urlopen(models_url, timeout=5) as resp:
                if resp.status == http.HTTPStatus.OK:
                    body = json.loads(resp.read())
                    models = body.get("data", [])
                    if models:
                        model_ids = [m.get("id", "?") for m in models]
                        logger.info(f"Model server ready after {attempt + 1}s, models: {model_ids}")
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
        elif isinstance(value, dict | list):
            args.extend([cli_key, json.dumps(value)])
        else:
            args.extend([cli_key, str(value)])
    return args


# ---------------------------------------------------------------------------
# Ray Serve server
# ---------------------------------------------------------------------------


def _start_ray_serve(
    cluster: RayCluster,
    model_id: str,
    engine_kwargs: dict[str, Any],
    autoscaling_config: dict[str, Any],
    verbose: bool = False,
) -> InferenceServer:
    """Deploy Ray Serve with vLLM on an already-running Ray cluster."""
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
        with contextlib.suppress(Exception):
            serve.shutdown()
        raise

    startup_s = time.perf_counter() - t0
    logger.info(f"Ray Serve started in {startup_s:.2f}s at http://localhost:{port}/v1")

    def _stop() -> None:
        logger.info("Shutting down Ray Serve")
        with contextlib.suppress(Exception):
            serve.shutdown()

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
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

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

    t0 = time.perf_counter()
    try:
        _wait_for_model_ready(port)
    except Exception:
        _stop()
        raise
    startup_s = time.perf_counter() - t0
    logger.info(f"vLLM server started in {startup_s:.2f}s on port {port}")

    atexit.register(_stop)

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
    cluster: RayCluster | None = None,
) -> InferenceServer:
    """Start an inference server and return a handle.

    Args:
        model_type: "ray-serve" or "vllm-direct"
        model_id: HuggingFace model ID
        engine_kwargs: vLLM engine kwargs (tensor_parallel_size, etc.)
        autoscaling_config: Ray Serve autoscaling config (only for ray-serve)
        log_dir: Directory for server logs
        verbose: Keep full logging if True
        cluster: A running RayCluster (required for ray-serve)

    Returns:
        InferenceServer handle. Call .stop() when done.
    """
    engine_kwargs = engine_kwargs or {}
    autoscaling_config = autoscaling_config or {"min_replicas": 1, "max_replicas": 1}

    if model_type == "ray-serve":
        if cluster is None:
            msg = "A running RayCluster must be provided for ray-serve model type"
            raise ValueError(msg)
        return _start_ray_serve(cluster, model_id, engine_kwargs, autoscaling_config, verbose=verbose)
    elif model_type == "vllm-direct":
        return _start_vllm_direct(model_id, engine_kwargs, log_dir=log_dir)
    else:
        msg = f"Unknown model_type: {model_type}. Must be 'ray-serve' or 'vllm-direct'."
        raise ValueError(msg)
