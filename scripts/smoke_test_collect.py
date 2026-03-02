#!/usr/bin/env python3
"""
OpenAct Smoke Test: Full Collection + Evaluation for All Parser-Based Tasks
============================================================================
Covers all tasks with eval_strategy='parser', excluding LLM-judge and safety tasks.

Tasks included:
  - gsm8k          (numeric, en)
  - mgsm           (numeric, 11 languages)
  - math           (math, en)
  - theoremqa      (type_aware, en)
  - mmlu           (mc4, en)
  - arc_challenge  (mc5, en)
  - commonsenseqa  (mc5, en)
  - belebele       (mc4, 18 languages)

Excluded (LLM judge): truthfulqa, humaneval, ifeval
Excluded (safety):    jbb, advbench, xstest
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test")

# ──────────────────────────────────────────────────────────────
# Task definitions
# ──────────────────────────────────────────────────────────────

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

BELEBELE_LANGUAGES = [
    "en", "zh", "ja", "de", "fr", "es", "ru", "ar",
    "hi", "ko", "pt", "it", "th", "vi", "bn", "sw", "te", "tr",
]

# Each entry: (task_name, task_kwargs, eval_task_name, description)
# task_kwargs will be merged with common kwargs (max_samples, etc.)
TASK_SPECS: List[Tuple[str, Dict[str, Any], str, str]] = []

def build_task_specs(
    max_samples: int = 5,
    mgsm_languages: Optional[List[str]] = None,
    belebele_languages: Optional[List[str]] = None,
    skip_multilingual: bool = False,
) -> List[Tuple[str, Dict[str, Any], str, str]]:
    """Build the full list of (task_name, kwargs, eval_name, description) tuples."""

    mgsm_langs = mgsm_languages or MGSM_LANGUAGES
    belebele_langs = belebele_languages or BELEBELE_LANGUAGES
    specs = []

    # ── Single-language tasks ──────────────────────────────────
    specs.append((
        "gsm8k", {"split": "test", "template": "zot"},
        "gsm8k", "GSM8K (grade-school math, en)"
    ))
    specs.append((
        "math", {"split": "test", "template": "zot"},
        "math", "MATH (competition math, en)"
    ))
    specs.append((
        "theoremqa", {"split": "test", "template": "zot"},
        "theoremqa", "TheoremQA (theorem-based, en)"
    ))
    specs.append((
        "mmlu", {"split": "test", "template": "zot", "max_per_subject": 1},
        "mmlu", "MMLU (multitask, en)"
    ))
    specs.append((
        "arc_challenge", {"split": "test", "template": "zot"},
        "arc_challenge", "ARC-Challenge (science QA, en)"
    ))
    specs.append((
        "commonsenseqa", {"split": "validation", "template": "zot"},
        "commonsenseqa", "CommonsenseQA (reasoning, en)"
    ))

    # ── Multilingual: MGSM ─────────────────────────────────────
    if not skip_multilingual:
        for lang in mgsm_langs:
            specs.append((
                "mgsm", {"split": "test", "template": "zot", "language": lang},
                "mgsm", f"MGSM (math, {lang})"
            ))

    # ── Multilingual: Belebele ──────────────────────────────────
    if not skip_multilingual:
        for lang in belebele_langs:
            specs.append((
                "belebele", {"split": "test", "template": "zot", "language": lang},
                "belebele", f"Belebele (reading comprehension, {lang})"
            ))

    # ── English-only MGSM/Belebele if multilingual skipped ─────
    if skip_multilingual:
        specs.append((
            "mgsm", {"split": "test", "template": "zot", "language": "en"},
            "mgsm", "MGSM (math, en-only)"
        ))
        specs.append((
            "belebele", {"split": "test", "template": "zot", "language": "en"},
            "belebele", "Belebele (reading comprehension, en-only)"
        ))

    return specs


# ──────────────────────────────────────────────────────────────
# Result tracking
# ──────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_name: str
    description: str
    language: str
    status: str = "pending"           # pending | collected | evaluated | error
    n_samples: int = 0
    n_ok: int = 0
    n_error: int = 0
    accuracy: Optional[float] = None
    n_correct: Optional[int] = None
    n_evaluated: Optional[int] = None
    collect_time_s: float = 0.0
    eval_time_s: float = 0.0
    output_dir: str = ""
    error_msg: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Collection
# ──────────────────────────────────────────────────────────────

def collect_task(
    task_name: str,
    task_kwargs: Dict[str, Any],
    model_name: str,
    output_dir: Path,
    max_samples: int,
    dtype: str,
    device_map: str,
    max_tokens: int,
    hidden_dtype: str,
    layers: Optional[str],
    queue_size: int,
) -> Dict[str, Any]:
    """
    Run collection for a single task configuration.
    Returns stats dict from CollectionRunner.run().
    """
    from openact_collect.engine.model_manager import ModelRunner
    from openact_collect.engine.collector import CollectionRunner
    from openact_collect.tasks.registry import TaskRegistry
    from openact_collect.schema import CaptureSpec, GenerationSpec

    # Merge max_samples into task kwargs
    merged_kwargs = {**task_kwargs, "max_samples": max_samples}

    # Create task
    task = TaskRegistry.create(task_name, **merged_kwargs)

    # Load model (caller should pass a pre-loaded ModelRunner for efficiency)
    # Here we accept model_name and create fresh — see run_all for shared loading
    model_runner = ModelRunner(
        model_name_or_path=model_name,
        dtype=dtype,
        device_map=device_map,
    )
    if not model_runner._loaded:
        model_runner.load()

    # Specs
    capture_layers = None
    if layers:
        capture_layers = [int(x.strip()) for x in layers.split(",")]

    capture_spec = CaptureSpec(
        hidden_states=True,
        hidden_states_layers=capture_layers,
        hidden_states_dtype=hidden_dtype,
        save_per_token=True,
        save_mean_states=True,
        save_prompt_last=True,
        compute_online_metrics=True,
    )
    generation_spec = GenerationSpec(
        max_new_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=42,
    )

    runner = CollectionRunner(
        model_manager=model_runner,
        task=task,
        output_dir=str(output_dir),
        capture_spec=capture_spec,
        generation_spec=generation_spec,
        queue_size=queue_size,
    )

    stats = runner.run()
    return stats


def collect_task_shared_model(
    task_name: str,
    task_kwargs: Dict[str, Any],
    model_runner,  # pre-loaded ModelRunner
    output_dir: Path,
    max_samples: int,
    max_tokens: int,
    hidden_dtype: str,
    layers: Optional[str],
    queue_size: int,
) -> Dict[str, Any]:
    """
    Run collection using a shared (already loaded) ModelRunner.
    """
    from openact_collect.engine.collector import CollectionRunner
    from openact_collect.tasks.registry import TaskRegistry
    from openact_collect.schema import CaptureSpec, GenerationSpec

    merged_kwargs = {**task_kwargs, "max_samples": max_samples}
    task = TaskRegistry.create(task_name, **merged_kwargs)

    capture_layers = None
    if layers:
        capture_layers = [int(x.strip()) for x in layers.split(",")]

    capture_spec = CaptureSpec(
        hidden_states=True,
        hidden_states_layers=capture_layers,
        hidden_states_dtype=hidden_dtype,
        save_per_token=True,
        save_mean_states=True,
        save_prompt_last=True,
        compute_online_metrics=True,
    )
    generation_spec = GenerationSpec(
        max_new_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=42,
    )

    runner = CollectionRunner(
        model_manager=model_runner,
        task=task,
        output_dir=str(output_dir),
        capture_spec=capture_spec,
        generation_spec=generation_spec,
        queue_size=queue_size,
    )

    stats = runner.run()
    return stats


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_run(
    run_dir: Path,
    eval_task_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a completed run using parser-based evaluation.
    Returns summary dict.
    """
    from openact_core import Run
    from openact_eval.evaluators.parser_evaluator import ParserEvaluator
    from openact_eval.pipeline import EvalPipeline

    run = Run(run_dir)
    evaluator = ParserEvaluator(task_name=eval_task_name)
    pipeline = EvalPipeline(
        run=run,
        evaluator=evaluator,
        label_name="correctness",
        save_labels=True,
        only_valid=True,
    )
    summary = pipeline.run_pipeline(progress=False)
    return summary


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

def validate_run(run_dir: Path) -> Dict[str, Any]:
    """Run structural validation on a completed run."""
    from openact_core.cli.validate import validate_run as _validate
    return _validate(run_dir, verbose=True)


# ──────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────

def run_all(
    model_name: str,
    output_root: Path,
    max_samples: int = 5,
    dtype: str = "auto",
    device_map: str = "auto",
    max_tokens: int = 2048,
    hidden_dtype: str = "float16",
    layers: Optional[str] = None,
    queue_size: int = 8,
    skip_multilingual: bool = False,
    skip_eval: bool = False,
    skip_validate: bool = False,
    mgsm_languages: Optional[List[str]] = None,
    belebele_languages: Optional[List[str]] = None,
    continue_on_error: bool = True,
) -> List[TaskResult]:
    """
    Run collection + evaluation for all parser-based tasks.
    """
    from openact_collect.engine.model_manager import ModelRunner

    specs = build_task_specs(
        max_samples=max_samples,
        mgsm_languages=mgsm_languages,
        belebele_languages=belebele_languages,
        skip_multilingual=skip_multilingual,
    )

    logger.info("=" * 70)
    logger.info("OpenAct Smoke Test — Full Collection + Evaluation")
    logger.info("=" * 70)
    logger.info(f"Model:            {model_name}")
    logger.info(f"Max samples/task: {max_samples}")
    logger.info(f"Total tasks:      {len(specs)}")
    logger.info(f"Output root:      {output_root}")
    logger.info(f"dtype:            {dtype}")
    logger.info(f"hidden_dtype:     {hidden_dtype}")
    logger.info(f"layers:           {layers or 'all'}")
    logger.info(f"max_tokens:       {max_tokens}")
    logger.info(f"queue_size:       {queue_size}")
    logger.info(f"skip_multilingual:{skip_multilingual}")
    logger.info(f"skip_eval:        {skip_eval}")
    logger.info(f"skip_validate:    {skip_validate}")
    logger.info("=" * 70)

    # ── Load model once ────────────────────────────────────────
    logger.info("Loading model (shared across all tasks) ...")
    t0 = time.time()
    model_runner = ModelRunner(
        model_name_or_path=model_name,
        dtype=dtype,
        device_map=device_map,
    )
    model_runner.load()
    model_load_time = time.time() - t0
    logger.info(f"Model loaded in {model_load_time:.1f}s")

    # ── Derive short model name for directory ──────────────────
    model_short = Path(model_name).name
    results: List[TaskResult] = []
    overall_start = time.time()

    for idx, (task_name, task_kwargs, eval_name, description) in enumerate(specs, 1):
        lang = task_kwargs.get("language", "en")
        # Build unique output directory
        lang_suffix = f"_{lang}" if lang != "en" or task_name in ("mgsm", "belebele") else ""
        run_name = f"{task_name}{lang_suffix}"
        run_output_dir = output_root / model_short / run_name

        result = TaskResult(
            task_name=task_name,
            description=description,
            language=lang,
            output_dir=str(run_output_dir),
        )

        logger.info("")
        logger.info(f"[{idx}/{len(specs)}] {description}")
        logger.info(f"  Output: {run_output_dir}")

        # ── Step 1: Collect ────────────────────────────────────
        try:
            t_collect = time.time()
            stats = collect_task_shared_model(
                task_name=task_name,
                task_kwargs=task_kwargs,
                model_runner=model_runner,
                output_dir=run_output_dir,
                max_samples=max_samples,
                max_tokens=max_tokens,
                hidden_dtype=hidden_dtype,
                layers=layers,
                queue_size=queue_size,
            )
            result.collect_time_s = time.time() - t_collect
            result.n_samples = stats.get("processed", 0)
            result.n_ok = stats.get("ok", 0)
            result.n_error = stats.get("error", 0) + stats.get("timeout", 0)
            result.status = "collected"
            logger.info(
                f"  Collected: {result.n_ok}/{result.n_samples} OK "
                f"({result.n_error} errors) in {result.collect_time_s:.1f}s"
            )
        except Exception as e:
            result.status = "error"
            result.error_msg = f"Collection failed: {traceback.format_exc()[-500:]}"
            logger.error(f"  COLLECTION ERROR: {e}")
            if not continue_on_error:
                raise
            results.append(result)
            continue

        # ── Step 2: Validate ───────────────────────────────────
        if not skip_validate:
            try:
                val = validate_run(run_output_dir)
                if not val["is_valid"]:
                    logger.warning(f"  Validation FAILED: {val['errors']}")
                    result.extra["validation_errors"] = val["errors"]
                else:
                    logger.info(f"  Validation: OK (warnings={val['n_warnings']})")
                    if val["warnings"]:
                        result.extra["validation_warnings"] = val["warnings"]
            except Exception as e:
                logger.warning(f"  Validation error: {e}")

        # ── Step 3: Evaluate ───────────────────────────────────
        if not skip_eval:
            try:
                t_eval = time.time()
                eval_summary = evaluate_run(
                    run_dir=run_output_dir,
                    eval_task_name=eval_name,
                )
                result.eval_time_s = time.time() - t_eval
                result.accuracy = eval_summary.get("accuracy")
                result.n_correct = eval_summary.get("n_correct")
                result.n_evaluated = eval_summary.get("n_evaluated")
                result.status = "evaluated"

                acc_str = f"{result.accuracy:.1%}" if result.accuracy is not None else "N/A"
                logger.info(
                    f"  Evaluated: accuracy={acc_str} "
                    f"({result.n_correct}/{result.n_evaluated}) "
                    f"in {result.eval_time_s:.1f}s"
                )
            except Exception as e:
                logger.error(f"  EVAL ERROR: {e}")
                result.extra["eval_error"] = str(e)
                if not continue_on_error:
                    raise

        results.append(result)

    # ── Unload model ───────────────────────────────────────────
    model_runner.unload()
    overall_time = time.time() - overall_start

    # ── Summary report ─────────────────────────────────────────
    _print_summary(results, model_name, overall_time, model_load_time)

    # ── Save JSON report ───────────────────────────────────────
    report = _build_report(results, model_name, max_samples, overall_time, model_load_time)
    report_path = output_root / model_short / "smoke_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved: {report_path}")

    return results


def _print_summary(
    results: List[TaskResult],
    model_name: str,
    overall_time: float,
    model_load_time: float,
):
    """Print a formatted summary table."""
    print("\n")
    print("=" * 90)
    print("  SMOKE TEST SUMMARY")
    print("=" * 90)
    print(f"  Model: {model_name}")
    print(f"  Total time: {overall_time:.1f}s (model load: {model_load_time:.1f}s)")
    print()
    print(f"  {'Task':<30s} {'Lang':>4s} {'Status':<12s} {'OK/N':>8s} {'Acc':>8s} {'Time':>8s}")
    print("  " + "-" * 80)

    n_collected = 0
    n_evaluated = 0
    n_errors = 0

    for r in results:
        lang = r.language
        status = r.status.upper()
        ok_n = f"{r.n_ok}/{r.n_samples}" if r.n_samples > 0 else "-"
        acc = f"{r.accuracy:.1%}" if r.accuracy is not None else "-"
        t = f"{r.collect_time_s + r.eval_time_s:.1f}s"

        status_marker = "✓" if r.status == "evaluated" else ("⚠" if r.status == "collected" else "✗")
        print(f"  {status_marker} {r.description:<28s} {lang:>4s} {status:<12s} {ok_n:>8s} {acc:>8s} {t:>8s}")

        if r.status in ("collected", "evaluated"):
            n_collected += 1
        if r.status == "evaluated":
            n_evaluated += 1
        if r.status == "error":
            n_errors += 1

    print("  " + "-" * 80)
    print(f"  Collected: {n_collected}/{len(results)}  |  Evaluated: {n_evaluated}/{len(results)}  |  Errors: {n_errors}/{len(results)}")
    print("=" * 90)

    # Print errors if any
    error_results = [r for r in results if r.error_msg]
    if error_results:
        print("\n  ERRORS:")
        for r in error_results:
            print(f"    {r.description}: {r.error_msg[:200]}")
        print()


def _build_report(
    results: List[TaskResult],
    model_name: str,
    max_samples: int,
    overall_time: float,
    model_load_time: float,
) -> Dict[str, Any]:
    """Build a JSON-serializable report dict."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": model_name,
        "max_samples_per_task": max_samples,
        "overall_time_s": round(overall_time, 2),
        "model_load_time_s": round(model_load_time, 2),
        "n_tasks": len(results),
        "n_collected": sum(1 for r in results if r.status in ("collected", "evaluated")),
        "n_evaluated": sum(1 for r in results if r.status == "evaluated"),
        "n_errors": sum(1 for r in results if r.status == "error"),
        "tasks": [
            {
                "task_name": r.task_name,
                "description": r.description,
                "language": r.language,
                "status": r.status,
                "n_samples": r.n_samples,
                "n_ok": r.n_ok,
                "n_error": r.n_error,
                "accuracy": r.accuracy,
                "n_correct": r.n_correct,
                "n_evaluated": r.n_evaluated,
                "collect_time_s": round(r.collect_time_s, 2),
                "eval_time_s": round(r.eval_time_s, 2),
                "output_dir": r.output_dir,
                "error_msg": r.error_msg,
                "extra": r.extra,
            }
            for r in results
        ],
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAct Smoke Test: collect + evaluate all parser-based tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke with 5 samples, English only
  python smoke_test_collect.py --model Qwen/Qwen2-7B-Instruct --max-samples 5 --skip-multilingual

  # Full multilingual smoke, 10 samples each
  python smoke_test_collect.py --model Qwen/Qwen2-7B-Instruct --max-samples 10

  # Only specific MGSM languages
  python smoke_test_collect.py --model Qwen/Qwen2-7B-Instruct --mgsm-langs en,zh,ja

  # Collect only (skip evaluation)
  python smoke_test_collect.py --model Qwen/Qwen2-7B-Instruct --skip-eval

  # Last 3 layers only, smaller queue for memory
  python smoke_test_collect.py --model Qwen/Qwen2-7B-Instruct --layers=-1,-2,-3 --queue-size 4

  # Full dataset (no max-samples cap)
  python smoke_test_collect.py --model Qwen/Qwen2-7B-Instruct --max-samples 0
        """,
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="Model name or path (e.g. 'Qwen/Qwen2-7B-Instruct')",
    )
    parser.add_argument(
        "--output", "-o", default="./runs",
        help="Root output directory (default: ./runs)",
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=5,
        help="Max samples per task (0 = no limit). Default: 5",
    )
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--hidden-dtype", choices=["float16", "float32"], default="float16",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices (e.g. '-1,-2,-3'). Default: all layers",
    )
    parser.add_argument("--queue-size", type=int, default=8)

    parser.add_argument("--skip-multilingual", action="store_true",
                        help="Skip multilingual variants (MGSM/Belebele), only run English")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip parser-based evaluation after collection")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip structural validation of runs")

    parser.add_argument(
        "--mgsm-langs", type=str, default=None,
        help=f"Comma-separated MGSM languages. Default: all ({','.join(MGSM_LANGUAGES)})",
    )
    parser.add_argument(
        "--belebele-langs", type=str, default=None,
        help=f"Comma-separated Belebele languages. Default: all ({','.join(BELEBELE_LANGUAGES)})",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop immediately on first error (default: continue)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output)
    max_samples = args.max_samples if args.max_samples > 0 else None

    mgsm_langs = None
    if args.mgsm_langs:
        mgsm_langs = [l.strip() for l in args.mgsm_langs.split(",")]

    belebele_langs = None
    if args.belebele_langs:
        belebele_langs = [l.strip() for l in args.belebele_langs.split(",")]

    results = run_all(
        model_name=args.model,
        output_root=output_root,
        max_samples=max_samples,
        dtype=args.dtype,
        device_map=args.device_map,
        max_tokens=args.max_tokens,
        hidden_dtype=args.hidden_dtype,
        layers=args.layers,
        queue_size=args.queue_size,
        skip_multilingual=args.skip_multilingual,
        skip_eval=args.skip_eval,
        skip_validate=args.skip_validate,
        mgsm_languages=mgsm_langs,
        belebele_languages=belebele_langs,
        continue_on_error=not args.stop_on_error,
    )

    # Exit code: 0 if all OK, 1 if any errors
    n_errors = sum(1 for r in results if r.status == "error")
    sys.exit(1 if n_errors > 0 else 0)


if __name__ == "__main__":
    main()