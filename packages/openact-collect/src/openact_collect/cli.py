import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("openact.cli")


def parse_layers(layers_str: str) -> Optional[List[int]]:
    """Parse comma-separated layer indices string into a list of ints.

    Returns None if the input is empty, otherwise a list of integers.
    """
    if not layers_str:
        return None
    return [int(x.strip()) for x in layers_str.split(",")]


def _configure_logging(output_dir: str, level: int = logging.INFO) -> None:
    """Set up root logging with both console and file handlers."""
    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    handlers = [logging.StreamHandler()]
    # Ensure output directory exists before creating file handler
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(out_path / "collect.log"))
    logging.basicConfig(level=level, format=log_format, handlers=handlers)


def main():
    parser = argparse.ArgumentParser(
        prog="openact",
        description="OpenAct: LLM Activation Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ---- collect subcommand with grouped arguments ----
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect activation data",
        description="Run data collection with specified model and task.",
    )

    # Required arguments
    collect_parser.add_argument(
        "--model", "-m", required=True,
        help="Model name or path (e.g., 'Qwen/Qwen2-7B-Instruct')",
    )
    collect_parser.add_argument(
        "--task", "-t", required=True,
        help="Task name (e.g., 'gsm8k', 'mmlu', 'math', 'arc_challenge', 'truthfulqa')",
    )
    collect_parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for the run",
    )

    # Data options
    data_group = collect_parser.add_argument_group("data options")
    data_group.add_argument(
        "--max-samples", "-n", type=int, default=None,
        help="Maximum number of samples to process",
    )
    data_group.add_argument(
        "--split", default="test",
        help="Dataset split to use (default: test)",
    )
    data_group.add_argument(
        "--template", default=None,
        help="Prompt template name (task-specific)",
    )
    data_group.add_argument(
        "--language", default=None,
        help="Language code for multilingual tasks (e.g., 'en', 'zh', 'ja')",
    )

    # Model options
    model_group = collect_parser.add_argument_group("model options")
    model_group.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto",
        help="Model dtype (default: auto)",
    )
    model_group.add_argument(
        "--device-map", default="auto",
        help="Device map for model loading (default: auto)",
    )

    # Capture options
    capture_group = collect_parser.add_argument_group("capture options")
    capture_group.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices to capture (e.g., '-1,-2,-3')",
    )
    capture_group.add_argument(
        "--no-hidden-states", action="store_true",
        help="Don't capture hidden states",
    )
    capture_group.add_argument(
        "--hidden-dtype", choices=["float16", "float32"], default="float16",
        help="Dtype for stored hidden states (default: float16)",
    )

    # Generation options
    gen_group = collect_parser.add_argument_group("generation options")
    gen_group.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    gen_group.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    gen_group.add_argument(
        "--top-p", type=float, default=1.0,
        help="Top-p sampling (default: 1.0)",
    )
    gen_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    gen_group.add_argument(
        "--queue-size", type=int, default=8,
        help="Async writer queue size (default: 8; use a smaller value for all-layers per-token to avoid OOM)",
    )

    # Safety arguments (optional extension)
    try:
        from openact_collect.tasks.safety.cli_extension import add_safety_arguments
        add_safety_arguments(collect_parser)
    except ImportError:
        pass

    # ---- list subcommand ----
    list_parser = subparsers.add_parser("list", help="List available tasks")
    list_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed task information",
    )

    # ---- info subcommand ----
    info_parser = subparsers.add_parser("info", help="Show information about a run")
    info_parser.add_argument("run_dir", help="Path to run directory")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)


def cmd_collect(args):
    from openact_collect.engine.model_manager import ModelRunner
    from openact_collect.engine.collector import CollectionRunner
    from openact_collect.tasks.registry import TaskRegistry
    from openact_collect.schema import CaptureSpec, GenerationSpec

    # Configure logging with file output in the run directory
    _configure_logging(args.output)

    logger.info("OpenAct Data Collection")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Output: {args.output}")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples}")

    task_kwargs = {"max_samples": args.max_samples, "split": args.split}
    if args.template:
        task_kwargs["template"] = args.template
    if args.language:
        task_kwargs["language"] = args.language

    if args.task.lower() in ("jbb", "advbench", "xstest"):
        try:
            from openact_collect.tasks.safety.cli_extension import resolve_safety_task_kwargs
            safety_kwargs = resolve_safety_task_kwargs(args)
            task_kwargs.update(safety_kwargs)
        except ImportError:
            raise ImportError(
                f"Safety tasks require additional dependencies. "
                f"Please ensure safety task modules are available."
            )

    try:
        task = TaskRegistry.create(args.task, **task_kwargs)
    except ValueError as e:
        logger.error(f"Task creation failed: {e}")
        sys.exit(1)

    logger.info("Loading model...")
    model_runner = ModelRunner(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
    )

    capture_spec = CaptureSpec(
        hidden_states=not args.no_hidden_states,
        hidden_states_layers=parse_layers(args.layers) if args.layers else None,
        hidden_states_dtype=args.hidden_dtype,
    )
    generation_spec = GenerationSpec(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    runner = CollectionRunner(
        model_manager=model_runner,
        task=task,
        output_dir=args.output,
        capture_spec=capture_spec,
        generation_spec=generation_spec,
        queue_size=args.queue_size,
    )

    logger.info("Starting collection...")
    stats = runner.run()

    logger.info("Collection complete!")
    logger.info("=" * 50)
    logger.info(f"Total processed: {stats['processed']}")
    logger.info(f"Successful: {stats['ok']}")
    logger.info(f"Errors: {stats['error'] + stats['timeout']}")
    logger.info(f"Duration: {stats['duration_seconds']:.1f}s")
    logger.info(f"Output: {args.output}")


def cmd_list(args):
    from openact_collect.tasks.registry import TaskRegistry

    tasks = TaskRegistry.list_with_info()
    if not tasks:
        print("No tasks registered.")
        return

    print("Available tasks:")
    print("-" * 40)
    for name, info in sorted(tasks.items()):
        if args.verbose:
            print(f"\n{name}:")
            print(f"  Source: {info['source']}")
            print(f"  Language: {info['language']}")
            if info["doc"]:
                doc_line = info["doc"].strip().split("\n")[0]
                print(f"  Description: {doc_line}")
        else:
            print(f"  {name}")


def cmd_info(args):
    from openact_core import Run

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    try:
        run = Run(run_dir)
    except Exception as e:
        print(f"Error loading run: {e}")
        sys.exit(1)

    print(f"Run Information: {run.run_dir.name}")
    print("=" * 50)
    manifest = run.manifest

    print(f"\nModel:")
    print(f"  Name: {manifest.model.name}")
    print(f"  Layers: {manifest.model.n_layers}")
    print(f"  Hidden dim: {manifest.model.hidden_dim}")

    print(f"\nDataset:")
    print(f"  Name: {manifest.dataset.name}")
    print(f"  Source: {manifest.dataset.source}")
    print(f"  Split: {manifest.dataset.split}")

    print(f"\nStatistics:")
    print(f"  Total samples: {run.stats.n_samples_total}")
    print(f"  Successful: {run.stats.n_samples_ok}")
    print(f"  Errors: {run.stats.n_samples_error}")
    print(f"  Total tokens: {run.stats.n_tokens_total}")
    if run.stats.duration_seconds:
        print(f"  Duration: {run.stats.duration_seconds:.1f}s")

    print(f"\nCapture:")
    capture_config = manifest.capture_config
    print(f"  Hidden states: {capture_config.get('hidden_states', False)}")
    if capture_config.get("hidden_states_layers"):
        print(f"  Layers: {capture_config.get('hidden_states_layers')}")


if __name__ == "__main__":
    main()