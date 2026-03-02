"""
Smart run loading utilities with R2 sync support.
"""

import os
import subprocess
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openact_core.io.run import Run


def load_run(
    path: str,
    r2_sync: bool = True,
    local_base: str = "/data/runs",
    require_complete: bool = False,
) -> "Run":
    """
    Smart run loader supporting local paths and R2 run names.
    
    Args:
        path: Local path or R2 run name
        r2_sync: If True, auto-pull from R2 if not found locally
        local_base: Base directory for local runs
        require_complete: Require _SUCCESS marker
        
    Returns:
        Loaded Run object
        
    Examples:
        # Local path
        run = load_run("/data/runs/gsm8k_v1")
        
        # R2 run name (auto-pull)
        run = load_run("gsm8k_v1")
        
        # Mount point path
        run = load_run("~/mnt/openact/gsm8k_v1")
    """
    from openact_core import Run
    
    # Expand user directory
    path = os.path.expanduser(path)
    
    # Check if it's a path (contains / or exists)
    if "/" in path or Path(path).exists():
        return Run(path, require_complete=require_complete)
    
    # Treat as run name
    local_path = Path(local_base) / path
    
    if local_path.exists():
        return Run(local_path, require_complete=require_complete)
    
    if r2_sync:
        print(f"Run '{path}' not found locally. Pulling from R2...")
        
        result = subprocess.run(
            ["openact-sync", "pull", path],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to pull from R2: {result.stderr}\n"
                f"Available runs: openact-sync list"
            )
        
        return Run(local_path, require_complete=require_complete)
    
    raise FileNotFoundError(
        f"Run '{path}' not found at {local_path}. "
        "Use r2_sync=True to pull from R2, or check run name."
    )


def list_local_runs(local_base: str = "/data/runs") -> list:
    """List all local runs."""
    base = Path(local_base)
    if not base.exists():
        return []
    
    runs = []
    for run_dir in base.iterdir():
        if run_dir.is_dir() and (run_dir / "manifest.json").exists():
            is_complete = (run_dir / "_SUCCESS").exists()
            runs.append({
                "name": run_dir.name,
                "path": str(run_dir),
                "is_complete": is_complete,
            })
    
    return sorted(runs, key=lambda x: x["name"])


def list_r2_runs() -> list:
    """List runs on R2 (requires openact-sync)."""
    result = subprocess.run(
        ["openact-sync", "list"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list R2 runs: {result.stderr}")
    
    # Parse output
    runs = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("✓") or line.startswith("⚠"):
            parts = line.split()
            if len(parts) >= 2:
                runs.append({
                    "name": parts[1],
                    "is_complete": line.startswith("✓"),
                    "size": parts[2] if len(parts) > 2 else "unknown",
                })
    
    return runs