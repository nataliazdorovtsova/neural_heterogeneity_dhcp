"""Run provenance snapshots.

Pattern borrowed from mr09's studies: every run writes a JSON record of
how it was configured, hashed so you can verify later that you are looking at
the run you think you are.

The part worth copying is that thresholds are recorded as {value, method}
pairs rather than bare numbers. Six months on, that is the difference between
"the cutoff was 0.05" and "the cutoff was the 95th percentile of the training
fold" — which is exactly what a review comment asks for.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import paths


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.PROJECT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _environment() -> dict:
    import sys
    import jax, numpy, scipy
    return {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "jax_x64": bool(jax.config.jax_enable_x64),
        "git_commit": _git_commit(),
    }


def threshold(value, method: str, **detail) -> dict:
    """Record a threshold together with the rule that produced it.

        threshold(4.49, "95th percentile of training fold", n=569)
    """
    return {"value": value, "method": method, **detail}


def snapshot(name: str, config: dict, results: dict,
             thresholds: dict | None = None,
             inputs: list | None = None,
             out_dir: Path | None = None) -> Path:
    """Write one run record. Returns the path written.

    `inputs` are data files whose contents get hashed, so a change in the
    upstream stack is detectable without re-reading it.
    """
    out_dir = Path(out_dir or (paths.RESULTS / "provenance"))
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    record = {
        "snapshot_timestamp": stamp,
        "spec_name": name,
        "spec_digest": _digest(config),
        "input_digests": {str(Path(p).name): file_digest(p) for p in (inputs or [])},
        "environment": _environment(),
        "config": config,
        "thresholds": thresholds or {},
        "results": results,
    }
    out = out_dir / f"{name}_{stamp}.json"
    out.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return out


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def verify(path, config: dict) -> bool:
    """True if the recorded run used exactly this configuration."""
    return load(path)["spec_digest"] == _digest(config)
