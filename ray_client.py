"""Ray cluster lifecycle management.

Starts a local Ray cluster via ``ray start --head`` and tears it down
by killing the process group. Respects an existing cluster if
``RAY_ADDRESS`` is already set.
"""

import contextlib
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

_RAY_PORT = 6379
_RAY_DASHBOARD_PORT = 8265
_RAY_CLUSTER_TIMEOUT_S = 300


def get_free_port(start_port: int) -> int:
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


@dataclass
class RayCluster:
    """Manage a local Ray cluster via ``ray start --head`` / process kill.

    Usage::

        cluster = RayCluster()
        cluster.start()
        # ... use Ray ...
        cluster.stop()

    Or as a context manager::

        with RayCluster() as cluster:
            ...
    """

    num_gpus: int | None = None
    num_cpus: int | None = None
    ray_port: int = _RAY_PORT
    dashboard_port: int = _RAY_DASHBOARD_PORT
    temp_dir: str = "/tmp/ray-serve-bench"
    log_file: str | Path | None = None

    _process: subprocess.Popen | None = field(init=False, default=None, repr=False)
    _log_fh: Any = field(init=False, default=None, repr=False)
    _external: bool = field(init=False, default=False, repr=False)

    def start(self) -> None:
        if os.environ.get("RAY_ADDRESS"):
            logger.info(f"Ray already running at {os.environ['RAY_ADDRESS']}, skipping cluster start")
            self._external = True
            return

        self.ray_port = get_free_port(self.ray_port)
        self.dashboard_port = get_free_port(self.dashboard_port)

        ip_address = socket.gethostbyname(socket.gethostname())

        cmd = [
            "ray",
            "start",
            "--head",
            "--node-ip-address",
            ip_address,
            "--port",
            str(self.ray_port),
            "--dashboard-port",
            str(self.dashboard_port),
            "--temp-dir",
            self.temp_dir,
            "--disable-usage-stats",
            "--block",
        ]
        if self.num_gpus is not None:
            cmd.extend(["--num-gpus", str(self.num_gpus)])
        if self.num_cpus is not None:
            cmd.extend(["--num-cpus", str(self.num_cpus)])

        logger.info(f"Starting Ray cluster: {' '.join(cmd)}")

        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_file, "w")  # noqa: SIM115

        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        os.environ["RAY_ADDRESS"] = f"{ip_address}:{self.ray_port}"

        try:
            self._wait_responsive()
        except Exception:
            self.stop()
            raise
        logger.info(f"Ray cluster ready at {os.environ['RAY_ADDRESS']}")

    def _wait_responsive(self) -> None:
        """Poll ``ray status`` until the cluster is responsive."""
        deadline = time.monotonic() + _RAY_CLUSTER_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    ["ray", "status"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and "No cluster status" not in result.stdout:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        msg = f"Ray cluster did not become responsive within {_RAY_CLUSTER_TIMEOUT_S}s"
        raise TimeoutError(msg)

    def stop(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping Ray cluster")
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait()
        except (ProcessLookupError, OSError):
            pass
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        os.environ.pop("RAY_ADDRESS", None)
        self._process = None
        logger.info("Ray cluster stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
