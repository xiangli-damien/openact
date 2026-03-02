"""
Example: Add correctness labels to an OpenAct run.

This script demonstrates the decoupled architecture:
  - ``openact-core`` provides ``get_parser()`` for answer extraction
  - No need to import ``openact-collect`` (torch/transformers)

Usage:
    python add_labels.py runs/gsm8k_example
    python add_labels.py runs/gsm8k_example --task gsm8k --output correctness
"""

import argparse
import pandas as pd
from pathlib import Path

from openact_core import Run
from openact_eval import ParserEvaluator


def main():
    parser = argparse.ArgumentParser(description="Add labels to OpenAct run")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument(
        "--task", "-t", default=None,
        help="Task name for parser selection (auto-detected from manifest if omitted)",
    )
    parser.add_argument(
        "--output", "-o", default="correctness",
        help="Label file name (without extension)",
    )
    args = parser.parse_args()

    print("OpenAct Labeling Example")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Load run                                                            #
    # ------------------------------------------------------------------ #
    run = Run(args.run_dir)
    print(f"Run: {run.run_dir.name}")
    print(f"Valid samples: {run.n_valid}")

    # ------------------------------------------------------------------ #
    #  Resolve parser from task name                                       #
    # ------------------------------------------------------------------ #
    task_name = args.task
    if task_name is None:
        # Auto-detect from manifest
        task_name = run.manifest.dataset.name
        if not task_name:
            raise ValueError(
                "Cannot auto-detect task name from manifest. "
                "Please specify --task explicitly."
            )
    print(f"Task: {task_name}")

    evaluator = ParserEvaluator(task_name=task_name)
    print(f"Evaluator: {evaluator.name}")

    # ------------------------------------------------------------------ #
    #  Extract answers and compute correctness                             #
    # ------------------------------------------------------------------ #
    print(f"\nExtracting answers and computing correctness...")
    labels_data = []
    correct_count = 0

    for sample in run.iter_valid():
        idx = sample.sample_idx

        # Use evaluator to get evaluation record
        record = evaluator.evaluate_sample(sample)
        is_correct = record.is_correct if record.is_correct is not None else False

        if is_correct:
            correct_count += 1

        labels_data.append({
            "sample_idx": idx,
            "extracted_answer": record.extracted_answer,
            "normalized_answer": record.normalized_answer,
            "ground_truth": record.ground_truth,
            "normalized_ground_truth": record.normalized_ground_truth,
            "is_correct": is_correct,
        })

    # ------------------------------------------------------------------ #
    #  Save labels                                                         #
    # ------------------------------------------------------------------ #
    labels_df = pd.DataFrame(labels_data)
    labels_dir = run.run_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    output_path = labels_dir / f"{args.output}.parquet"
    labels_df.to_parquet(output_path, index=False)

    total = len(labels_data)
    accuracy = correct_count / total * 100 if total > 0 else 0

    print(f"\nLabeling complete!")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Saved to: {output_path}")

    # ------------------------------------------------------------------ #
    #  Verify label access                                                 #
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("Verifying label access...")
    print(f"{'=' * 60}")

    # Reload to test label reading through the Run API.
    run = Run(args.run_dir)
    sample = run[0]
    is_correct = sample.get_label("is_correct")
    extracted = sample.get_label("extracted_answer")
    print(f"\nSample 0:")
    print(f"  Response: {sample.response_text[:100]}...")
    print(f"  Extracted answer: {extracted}")
    print(f"  Is correct: {is_correct}")

    print(f"\nFiltering by correctness:")
    correct_samples = list(run.filter_by_label("is_correct", True))
    incorrect_samples = list(run.filter_by_label("is_correct", False))
    print(f"  Correct samples: {len(correct_samples)}")
    print(f"  Incorrect samples: {len(incorrect_samples)}")


if __name__ == "__main__":
    main()
