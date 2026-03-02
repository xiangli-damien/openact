# packages/openact-core/src/openact_core/cli/sync.py
"""OpenAct R2 Sync CLI."""
import argparse
import subprocess
import sys
import os
from pathlib import Path

LOCAL_BASE = Path(os.environ.get("OPENACT_RUNS_DIR", "/data/runs"))
R2_BUCKET = "r2:autoact-data/runs"


def rclone(args: list, dry_run: bool = False) -> int:
    cmd = ["rclone"] + (["--dry-run"] if dry_run else []) + args
    print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def push(run_name: str, dry_run: bool = False) -> int:
    local = LOCAL_BASE / run_name
    if not local.exists():
        print(f"[ERROR] Not found: {local}")
        return 1
    if not (local / "_SUCCESS").exists():
        print(f"[WARN] Run incomplete (no _SUCCESS)")
    return rclone([
        "sync", str(local), f"{R2_BUCKET}/{run_name}",
        "--progress", "--transfers", "16", "-v"
    ], dry_run)


def pull(run_name: str, dry_run: bool = False) -> int:
    local = LOCAL_BASE / run_name
    local.parent.mkdir(parents=True, exist_ok=True)
    return rclone([
        "sync", f"{R2_BUCKET}/{run_name}", str(local),
        "--progress", "--transfers", "16", "-v"
    ], dry_run)


def ls() -> int:
    return subprocess.run(["rclone", "lsf", "--dirs-only", R2_BUCKET]).returncode


def main():
    parser = argparse.ArgumentParser(prog="openact-sync")
    sub = parser.add_subparsers(dest="cmd")
    
    p = sub.add_parser("push")
    p.add_argument("run_name")
    p.add_argument("--dry-run", action="store_true")
    
    p = sub.add_parser("pull")
    p.add_argument("run_name")
    p.add_argument("--dry-run", action="store_true")
    
    sub.add_parser("list")
    
    args = parser.parse_args()
    if args.cmd == "push":
        sys.exit(push(args.run_name, args.dry_run))
    elif args.cmd == "pull":
        sys.exit(pull(args.run_name, args.dry_run))
    elif args.cmd == "list":
        sys.exit(ls())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()