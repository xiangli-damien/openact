"""
OpenAct-Eval command-line interface.

Commands
--------
  eval           Evaluate a run (parser / LLM judge / auto).
  label-safety   Label a safety run with LlamaGuard.
  compare        Compare multiple evaluators on the same run.
  list-evaluators List available evaluators.
  list-parsers    List available answer parsers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        prog="openact-eval",
        description="OpenAct Eval: evaluation and labeling toolkit for LLM activation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # -- eval ----------------------------------------------------------------
    eval_parser = subparsers.add_parser("eval", help="Evaluate a run", aliases=["run"])
    eval_parser.add_argument("run_dir", help="Path to run directory")
    eval_parser.add_argument(
        "--task", "-t", default=None,
        help="Task name (auto-detected from manifest if omitted)",
    )
    eval_parser.add_argument(
        "--evaluator", "-e", default="auto",
        choices=["auto", "parser", "llm"],
        help="Evaluator type (default: auto)",
    )
    eval_parser.add_argument(
        "--label", "-l", default="correctness",
        help="Label file name (default: correctness)",
    )
    eval_parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save labels to disk",
    )
    eval_parser.add_argument(
        "--llm-backend", default="openai",
        help="LLM backend for judge evaluator (default: openai)",
    )
    eval_parser.add_argument(
        "--llm-model", default="gpt-4o-mini",
        help="LLM model for judge evaluator (default: gpt-4o-mini)",
    )
    eval_parser.add_argument(
        "--api-key", default=None,
        help="API key for LLM backend (prefers OPENAI_API_KEY env var)",
    )
    eval_parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )

    # -- label-safety --------------------------------------------------------
    safety_parser = subparsers.add_parser(
        "label-safety",
        help="Label a safety run with LlamaGuard",
    )
    safety_parser.add_argument("run_dir", help="Path to run directory")
    safety_parser.add_argument(
        "--task", "-t", default=None,
        help="Task name (auto-detected from manifest if omitted)",
    )
    safety_parser.add_argument(
        "--guard-model", default="meta-llama/Llama-Guard-3-8B",
        help="LlamaGuard model name or local path (default: meta-llama/Llama-Guard-3-8B)",
    )
    safety_parser.add_argument(
        "--device-map", default="auto",
        help="Device map for model loading (default: auto)",
    )
    safety_parser.add_argument(
        "--dtype", default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
    )
    safety_parser.add_argument(
        "--max-new-tokens", type=int, default=32,
        help="Max new tokens for guard generation (default: 32)",
    )
    safety_parser.add_argument(
        "--label-request", action="store_true", default=False,
        help="Additionally classify the user request as safe/unsafe",
    )
    safety_parser.add_argument(
        "--label", "-l", default="safety",
        help="Label file name (default: safety)",
    )
    safety_parser.add_argument(
        "--write-npy", action="store_true", default=True,
        help="Write per-sample .npy arrays (default: True)",
    )
    safety_parser.add_argument(
        "--no-write-npy", dest="write_npy", action="store_false",
    )
    safety_parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )

    # -- compare -------------------------------------------------------------
    compare_parser = subparsers.add_parser(
        "compare", help="Compare evaluators on a run"
    )
    compare_parser.add_argument("run_dir", help="Path to run directory")
    compare_parser.add_argument("--task", "-t", default=None, help="Task name")

    # -- list ----------------------------------------------------------------
    subparsers.add_parser("list-evaluators", help="List available evaluators")
    subparsers.add_parser("list-parsers", help="List available parsers")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command in ("eval", "run"):
        cmd_eval(args)
    elif args.command == "label-safety":
        cmd_label_safety(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "list-evaluators":
        cmd_list_evaluators()
    elif args.command == "list-parsers":
        cmd_list_parsers()


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def cmd_eval(args):
    from openact_core import Run
    from openact_eval.pipeline import EvalPipeline
    from openact_eval.evaluators.parser_evaluator import ParserEvaluator
    from openact_eval.evaluators.llm_judge import LLMJudgeEvaluator
    from openact_eval.evaluators.registry import auto_select_evaluator

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    run = Run(run_dir)
    task_name = args.task or run.manifest.dataset.name
    if not task_name:
        print(
            "Error: Cannot detect task name. Use --task to specify.",
            file=sys.stderr,
        )
        sys.exit(1)

    evaluator = None
    if args.evaluator == "parser":
        evaluator = ParserEvaluator(task_name=task_name)
    elif args.evaluator == "llm":
        evaluator = LLMJudgeEvaluator(
            backend=args.llm_backend,
            model=args.llm_model,
            api_key=args.api_key,
        )
    else:
        evaluator = auto_select_evaluator(
            task_name,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
        )

    pipeline = EvalPipeline(
        run=run,
        evaluator=evaluator,
        label_name=args.label,
        save_labels=not args.no_save,
    )

    if not args.json:
        print("OpenAct Evaluation")
        print("=" * 60)
        print(f"Run: {run_dir.name}")
        print(f"Task: {task_name}")
        print(f"Evaluator: {evaluator.name}")
        print(f"Valid samples: {run.n_valid}")
        print()

    summary = pipeline.run_pipeline(progress=not args.json)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        metrics = summary["metrics"]
        print()
        print("Results")
        print("-" * 40)
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Correct: {metrics['n_correct']}")
        print(f"  Incorrect: {metrics['n_incorrect']}")
        print(f"  Errors: {metrics['n_error']}")
        print(f"  Evaluated: {metrics['n_evaluated']}/{metrics['n_total']}")
        if summary.get("label_path"):
            print(f"\n  Labels saved to: {summary['label_path']}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# label-safety
# ---------------------------------------------------------------------------
def cmd_label_safety(args):
    import numpy as np
    from openact_core import Run
    from openact_eval.evaluators.safety_evaluators import LlamaGuardEvaluator
    from openact_eval.metrics_safety import compute_safety_metrics, format_safety_report

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    run = Run(run_dir)
    task_name = args.task or run.manifest.dataset.name
    if not task_name:
        print(
            "Error: Cannot detect task name. Use --task to specify.",
            file=sys.stderr,
        )
        sys.exit(1)

    evaluator = LlamaGuardEvaluator(
        model_name=args.guard_model,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        label_request=args.label_request,
    )

    if not args.json:
        print()
        print("=" * 60)
        print("  SAFETY LABELING WITH LLAMAGUARD")
        print("=" * 60)
        print(f"  Run:          {run_dir.name}")
        print(f"  Task:         {task_name}")
        print(f"  Guard model:  {args.guard_model}")
        print(f"  Label request: {args.label_request}")
        print(f"  Valid samples: {run.n_valid}")
        print("=" * 60)
        print()

    # run evaluation
    result = evaluator.evaluate(run=run, only_valid=True, progress=not args.json)

    # save parquet labels
    labels_dir = run_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    label_path = result.save_labels(run_dir, args.label)

    # optionally save per-sample .npy arrays
    npy_dir: Optional[Path] = None
    if args.write_npy:
        npy_dir = labels_dir / "npy"
        npy_dir.mkdir(exist_ok=True)
        n_total = len(run)

        is_safe_arr = np.full((n_total,), -1, dtype=np.int8)
        unsafe_prob_arr = np.full((n_total,), np.nan, dtype=np.float32)
        safe_prob_arr = np.full((n_total,), np.nan, dtype=np.float32)
        is_refusal_arr = np.full((n_total,), -1, dtype=np.int8)

        for rec in result.records:
            idx = rec.sample_idx
            if 0 <= idx < n_total:
                meta = rec.meta
                if meta.get("is_safe") is not None:
                    is_safe_arr[idx] = int(meta["is_safe"])
                if meta.get("unsafe_prob") is not None:
                    unsafe_prob_arr[idx] = float(meta["unsafe_prob"])
                if meta.get("safe_prob") is not None:
                    safe_prob_arr[idx] = float(meta["safe_prob"])
                if meta.get("is_refusal") is not None:
                    is_refusal_arr[idx] = int(meta["is_refusal"])

        np.save(npy_dir / "is_safe.npy", is_safe_arr)
        np.save(npy_dir / "unsafe_prob.npy", unsafe_prob_arr)
        np.save(npy_dir / "safe_prob.npy", safe_prob_arr)
        np.save(npy_dir / "is_refusal.npy", is_refusal_arr)

    # compute safety metrics
    safety_metrics = compute_safety_metrics(result)

    if args.json:
        output = {
            "run_path": str(run_dir),
            "task": task_name,
            "guard_model": args.guard_model,
            "label_path": label_path,
            "npy_dir": str(npy_dir) if npy_dir else None,
            **safety_metrics.to_dict(),
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        report = format_safety_report(safety_metrics)
        print(report)
        print(f"\n  Labels saved to: {label_path}")
        if npy_dir:
            print(f"  Numpy arrays:   {npy_dir}")
        print()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def cmd_compare(args):
    from openact_core import Run
    from openact_eval.evaluators.parser_evaluator import ParserEvaluator
    from openact_eval.evaluators.llm_judge import LLMJudgeEvaluator
    from openact_eval.metrics import compare_evaluators

    run_dir = Path(args.run_dir)
    run = Run(run_dir)
    task_name = args.task or run.manifest.dataset.name

    print("OpenAct Evaluator Comparison")
    print("=" * 60)
    print(f"Run: {run_dir.name}")
    print(f"Task: {task_name}")
    print()

    results = {}

    try:
        parser_eval = ParserEvaluator(task_name=task_name)
        print(f"Running: {parser_eval.name} ...")
        results["parser"] = parser_eval.evaluate(run)
    except ValueError:
        print(f"  No parser available for '{task_name}'")

    try:
        llm_eval = LLMJudgeEvaluator()
        print(f"Running: {llm_eval.name} ...")
        results["llm_judge"] = llm_eval.evaluate(run)
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"  LLM judge unavailable: {e}")

    if not results:
        print("No evaluators could run.")
        sys.exit(1)

    comparison = compare_evaluators(results)

    print()
    print("Comparison")
    print("-" * 40)
    for name, data in comparison.items():
        if "accuracy" in data:
            print(f"  {name}: accuracy={data['accuracy']:.1%}")
        elif "agreement" in data:
            print(f"  {name}: agreement={data['agreement']:.1%}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# list helpers
# ---------------------------------------------------------------------------
def cmd_list_evaluators():
    from openact_eval.evaluators.registry import EvaluatorRegistry

    print("Built-in evaluators:")
    print("  parser          - Deterministic parser-based evaluation")
    print("  llm_judge       - LLM-as-judge evaluation (OpenAI / Anthropic)")
    print("  llamaguard      - LlamaGuard safety classification (GPU)")
    print("  refusal_heuristic - Keyword-based refusal detection (CPU fallback)")
    print()

    registered = EvaluatorRegistry.list_with_info()
    if registered:
        print("Registered evaluators:")
        for name, info in registered.items():
            print(f"  {name}: {info['doc']}")


def cmd_list_parsers():
    from openact_core.tasks.parsers import list_parsers

    parsers = list_parsers()
    print("Available parsers:")
    for p in parsers:
        print(f"  {p}")


if __name__ == "__main__":
    main()