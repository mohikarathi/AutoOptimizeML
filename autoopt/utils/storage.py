"""Storage and persistence utilities for reproducible run artifacts."""

import os
import json
import time
from typing import Dict, Any, Optional, List


DEFAULT_RUNS_DIR = os.path.expanduser("~/.autoopt_runs")


def save_run_artifact(run_data: Dict[str, Any], runs_dir: str = DEFAULT_RUNS_DIR) -> str:
    """Save an optimization run dictionary as a timestamped JSON file."""
    os.makedirs(runs_dir, exist_ok=True)
    run_id = run_data.get("run_id", f"run_{int(time.time())}")
    filename = f"{run_id}.json"
    filepath = os.path.join(runs_dir, filename)

    with open(filepath, "w") as f:
        json.dump(run_data, f, indent=2)

    # Also update 'latest.json' symlink / copy
    latest_path = os.path.join(runs_dir, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(run_data, f, indent=2)

    return filepath


def load_run_artifact(run_id_or_path: str, runs_dir: str = DEFAULT_RUNS_DIR) -> Optional[Dict[str, Any]]:
    """Load a run artifact JSON by run ID ('latest', 'run_123') or explicit file path."""
    if os.path.exists(run_id_or_path):
        with open(run_id_or_path, "r") as f:
            return json.load(f)

    if not run_id_or_path.endswith(".json"):
        run_id_or_path = f"{run_id_or_path}.json"

    target = os.path.join(runs_dir, run_id_or_path)
    if os.path.exists(target):
        with open(target, "r") as f:
            return json.load(f)

    return None


def list_runs(runs_dir: str = DEFAULT_RUNS_DIR) -> List[str]:
    """List all saved run JSON files."""
    if not os.path.exists(runs_dir):
        return []
    files = [f for f in os.listdir(runs_dir) if f.endswith(".json") and f != "latest.json"]
    return sorted(files, reverse=True)
