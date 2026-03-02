#!/usr/bin/env python3
"""
Check prompts and answers for collected runs.

Reads each run's data.parquet (and labels/correctness.parquet if present),
prints or saves a readable report: prompt, response, ground_truth, and eval
fields so you can verify format and alignment before full-scale collection.

Usage:
  python scripts/check_prompts_and_answers.py
  python scripts/check_prompts_and_answers.py --path smoke_test_runs/Qwen2-7B-Instruct
  python scripts/check_prompts_and_answers.py --path runs --max-per-task 2 --output report.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def find_run_dirs(root: Path):
    """Yield (run_name, run_dir) for each subdir that contains data.parquet."""
    root = Path(root)
    if not root.is_dir():
        return
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "data.parquet").exists():
            yield d.name, d


def load_labels(run_dir: Path):
    """Load labels parquet if present (e.g. labels/correctness.parquet)."""
    for name in ("correctness", "labels"):
        p = run_dir / "labels" / f"{name}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    return None


def truncate(s: str, max_len: int = 600, suffix: str = "…") -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)] + suffix


def run(
    root: Path,
    max_per_task: int = 1,
    output_path: Optional[Path] = None,
    max_prompt_len: int = 800,
    max_response_len: int = 500,
    max_gt_len: int = 200,
) -> None:
    runs = list(find_run_dirs(root))
    if not runs:
        print(f"No run dirs with data.parquet under {root}", file=sys.stderr)
        sys.exit(1)

    lines: list[str] = []
    out = lambda s: lines.append(s)

    for run_name, run_dir in runs:
        df = pd.read_parquet(run_dir / "data.parquet")
        labels_df = load_labels(run_dir)

        n_show = min(max_per_task, len(df))
        out("")
        out("=" * 80)
        out(f"  {run_name}  (n_samples={len(df)}, showing {n_show})")
        out("=" * 80)

        for i in range(n_show):
            row = df.iloc[i]
            idx = row.get("sample_idx", i)
            out("")
            out("-" * 60)
            out(f"  Sample index: {idx}")
            out("-" * 60)
            out("  PROMPT (to model):")
            out(truncate(row.get("prompt_text"), max_prompt_len))
            out("")
            out("  RESPONSE (model output):")
            out(truncate(row.get("response_text"), max_response_len))
            out("")
            out("  GROUND TRUTH:")
            out(truncate(row.get("ground_truth"), max_gt_len))
            if labels_df is not None and "sample_idx" in labels_df.columns:
                match = labels_df[labels_df["sample_idx"] == idx]
                if not match.empty:
                    lrow = match.iloc[0]
                    out("")
                    out("  EVAL:")
                    out(f"    extracted_answer  = {lrow.get('extracted_answer', '')}")
                    out(f"    normalized_answer = {lrow.get('normalized_answer', '')}")
                    out(f"    is_correct        = {lrow.get('is_correct', '')}")
            out("")

    report = "\n".join(lines)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report written to {output_path}", file=sys.stderr)
    else:
        print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Check prompts and answers for collected runs",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=Path("runs/Qwen2-7B-Instruct"),
        help="Root dir containing run subdirs (each with data.parquet)",
    )
    parser.add_argument(
        "--max-per-task",
        "-n",
        type=int,
        default=1,
        help="Max samples to show per run (default: 1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write report to this file instead of stdout",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=800,
        help="Truncate prompt to this many chars (default: 800)",
    )
    parser.add_argument(
        "--max-response-len",
        type=int,
        default=500,
        help="Truncate response to this many chars (default: 500)",
    )
    parser.add_argument(
        "--max-gt-len",
        type=int,
        default=200,
        help="Truncate ground_truth to this many chars (default: 200)",
    )
    args = parser.parse_args()
    run(
        root=args.path,
        max_per_task=args.max_per_task,
        output_path=args.output,
        max_prompt_len=args.max_prompt_len,
        max_response_len=args.max_response_len,
        max_gt_len=args.max_gt_len,
    )


if __name__ == "__main__":
    main()
