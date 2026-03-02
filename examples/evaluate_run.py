"""
Example: Evaluate an OpenAct run using the eval pipeline.

Usage:
    python examples/evaluate_run.py <run_dir> [--task TASK] [--evaluator TYPE]

This demonstrates three ways to evaluate:
    1. Auto-selection (recommended)
    2. Explicit parser-based evaluation
    3. LLM-as-judge evaluation
"""

import argparse
import json
from pathlib import Path

from openact_core import Run
from openact_eval import (
    EvalPipeline,
    ParserEvaluator,
    compute_metrics,
)
from openact_eval.evaluators.registry import auto_select_evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate an OpenAct run")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument(
        "--task", "-t", default=None,
        help="Task name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--evaluator", "-e", default="auto",
        choices=["auto", "parser", "llm"],
        help="Evaluator type",
    )
    parser.add_argument(
        "--label", "-l", default="correctness",
        help="Label file name",
    )
    args = parser.parse_args()

    print("OpenAct Evaluation Example")
    print("=" * 60)

    # --- Method 1: EvalPipeline (recommended) ---
    print("\n[Method 1] Using EvalPipeline (auto-select evaluator)")
    print("-" * 40)

    pipeline = EvalPipeline(
        run_path=args.run_dir,
        label_name=args.label,
    )

    summary = pipeline.run_pipeline()
    print(f"  Accuracy: {summary['accuracy']:.1%}")
    print(f"  Correct: {summary['n_correct']}/{summary['n_evaluated']}")
    print(f"  Labels saved to: {summary.get('label_path', 'N/A')}")

    # --- Method 2: Explicit evaluator ---
    print("\n[Method 2] Explicit ParserEvaluator")
    print("-" * 40)

    run = Run(args.run_dir)
    task_name = args.task or run.manifest.dataset.name

    try:
        evaluator = ParserEvaluator(task_name=task_name)
        result = evaluator.evaluate(run)
        metrics = compute_metrics(result)

        print(f"  Evaluator: {evaluator.name}")
        print(f"  Accuracy: {metrics.accuracy:.1%}")
        print(f"  Correct: {metrics.n_correct}/{metrics.n_evaluated}")
        print(f"  Errors: {metrics.n_error}")

        # Access individual records
        for record in result.records[:3]:
            print(
                f"  Sample {record.sample_idx}: "
                f"correct={record.is_correct}, "
                f"extracted='{record.extracted_answer}', "
                f"gt='{record.ground_truth}'"
            )
    except ValueError as e:
        print(f"  Parser not available: {e}")

    # --- Method 3: Using the Run's existing labels ---
    print("\n[Method 3] Reading labels back from saved file")
    print("-" * 40)

    run = Run(args.run_dir)  # reload to pick up new labels
    for sample in list(run.iter_valid())[:3]:
        is_correct = sample.get_label("is_correct")
        extracted = sample.get_label("extracted_answer")
        print(
            f"  Sample {sample.sample_idx}: "
            f"correct={is_correct}, extracted='{extracted}'"
        )

    # Summary of filtering
    correct_samples = run.filter_by_label("is_correct", True)
    incorrect_samples = run.filter_by_label("is_correct", False)
    print(f"\n  Correct samples: {len(correct_samples)}")
    print(f"  Incorrect samples: {len(incorrect_samples)}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
