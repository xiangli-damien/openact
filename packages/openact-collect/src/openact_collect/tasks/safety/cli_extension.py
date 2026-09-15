import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openact_collect.schema import GenerationProfile, DEFAULT_SAFETY_PROFILES, build_profiles_from_cli
def add_safety_arguments(parser: argparse.ArgumentParser) -> None:
    safety_group = parser.add_argument_group("safety options")
    safety_group.add_argument(
        "--profiles",
        type=str,
        default=None,
        help=(
            "Generation profiles for multi-temperature collection. "
            "Format: 'greedy:temp=0.0 warm:temp=0.7,n_gen=2 hot:temp=1.0,n_gen=2'. "
            "If omitted, uses default profiles (greedy + warm + hot)."
        ),
    )
    safety_group.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Don't load attack artifacts (goal prompts only).",
    )
    safety_group.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Directory containing attack artifacts (JSONL/JSON files).",
    )
    safety_group.add_argument(
        "--artifact-methods",
        type=str,
        default=None,
        help="Comma-separated list of artifact methods to include (e.g. 'GCG,PAIR').",
    )
    safety_group.add_argument(
        "--safety-split",
        type=str,
        default=None,
        choices=["all", "harmful", "benign"],
        help="Which safety split to collect (default: all).",
    )
def resolve_safety_task_kwargs(args: argparse.Namespace) -> Dict:
    kwargs = {}
    if args.profiles:
        kwargs["profiles"] = build_profiles_from_cli(args.profiles)
    else:
        kwargs["profiles"] = dict(DEFAULT_SAFETY_PROFILES)
    is_jbb = str(getattr(args, 'task', '')).lower() == 'jbb'
    if is_jbb:
        kwargs['include_artifacts'] = bool(getattr(args, 'artifact_dir', None)) and not getattr(args, 'no_artifacts', False)
    if is_jbb and getattr(args, "artifact_dir", None):
        from openact_collect.tasks.safety.artifacts import (
            DirectoryArtifactLoader,
        )
        methods = None
        if getattr(args, "artifact_methods", None):
            methods = [m.strip() for m in args.artifact_methods.split(",")]
        kwargs["artifact_loader"] = DirectoryArtifactLoader(
            artifacts_dir=args.artifact_dir, methods=methods
        )
    elif is_jbb and getattr(args, "artifact_methods", None):
        methods = [m.strip() for m in args.artifact_methods.split(",")]
        kwargs["artifact_methods"] = methods
    split = getattr(args, 'safety_split', None) or getattr(args, 'split', None)
    if split:
        kwargs['split'] = split
    return kwargs
_SAFETY_TASK_NAMES = {"jbb", "advbench", "xstest"}
def is_safety_task(task_name: str) -> bool:
    return task_name.lower() in _SAFETY_TASK_NAMES
def get_safety_capture_defaults() -> Dict:
    return {
        "save_per_token": False,
        "save_mean_states": True,
        "save_prompt_last": True,
        "compute_online_metrics": True,
    }
