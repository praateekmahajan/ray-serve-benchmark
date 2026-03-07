"""Results writing and summary utilities."""

import json
from pathlib import Path
from typing import Any


def write_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Write benchmark results to params.json and metrics.json."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for key in ("params", "metrics"):
        if key not in results:
            continue
        path = output_path / f"{key}.json"
        try:
            existing = json.loads(path.read_text())
        except FileNotFoundError:
            existing = {}
        existing.update(results[key])
        path.write_text(json.dumps(existing, indent=2, default=str))


def write_summary(results: dict[str, Any], output_path: str | Path) -> None:
    """Write a human-readable summary.txt for quick inspection."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = results.get("metrics", {})
    params = results.get("params", {})

    lines = []
    lines.append(f"Model:       {metrics.get('model_id', params.get('model_id', 'N/A'))}")
    lines.append(f"Backend:     {metrics.get('model_type', params.get('model_type', 'N/A'))}")
    lines.append(f"Concurrency: {metrics.get('max_concurrency', params.get('max_concurrency', 'N/A'))}")
    lines.append(f"Prompts:     {metrics.get('num_prompts', params.get('num_prompts', 'N/A'))}")
    lines.append(f"Input/Output:{metrics.get('input_len', 'N/A')} / {metrics.get('output_len', 'N/A')} tokens")
    lines.append("")

    if not metrics.get("is_success", False):
        lines.append("Status: FAILED")
    else:
        lines.append("Status: SUCCESS")
        lines.append(f"Startup:     {metrics.get('serve_startup_s', 0):.1f}s")
        lines.append(f"Duration:    {metrics.get('duration', 0):.1f}s")
        lines.append(f"Completed:   {metrics.get('completed', 'N/A')} requests")
        lines.append("")
        lines.append("Throughput:")
        lines.append(f"  Requests:  {metrics.get('request_throughput', 0):.2f} req/s")
        lines.append(f"  Output:    {metrics.get('output_throughput', 0):.1f} tok/s")
        lines.append(f"  Total:     {metrics.get('total_token_throughput', 0):.1f} tok/s")
        lines.append("")
        lines.append("Latency (ms):          mean     median   p99      std")
        for label, prefix in [("  TTFT", "ttft"), ("  TPOT", "tpot"), ("  ITL", "itl"), ("  E2EL", "e2el")]:
            mean = metrics.get(f"mean_{prefix}_ms", 0)
            median = metrics.get(f"median_{prefix}_ms", 0)
            p99 = metrics.get(f"p99_{prefix}_ms", 0)
            std = metrics.get(f"std_{prefix}_ms", 0)
            lines.append(f"{label:18s} {mean:8.1f}   {median:8.1f}   {p99:8.1f}   {std:8.1f}")

    lines.append("")
    (output_path / "summary.txt").write_text("\n".join(lines))
