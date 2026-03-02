"""OpenAct Run Comparison CLI.

Compare results across multiple runs (different models on the same task).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from openact_core.comparison import (
    compare_runs,
    compare_by_sample_id,
    find_disagreements,
    aggregate_across_tasks,
)


def main():
    parser = argparse.ArgumentParser(
        prog="openact-compare",
        description="Compare OpenAct runs across models or configurations."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comparison commands")
    
    # Summary command
    summary_parser = subparsers.add_parser(
        "summary",
        help="Generate summary comparison table"
    )
    summary_parser.add_argument(
        "runs",
        nargs="+",
        help="Run directory paths to compare"
    )
    summary_parser.add_argument(
        "--accuracy-column",
        help="Label column containing accuracy values"
    )
    summary_parser.add_argument(
        "--format",
        choices=["table", "markdown", "json", "csv"],
        default="table",
        help="Output format (default: table)"
    )
    summary_parser.add_argument(
        "-o", "--output",
        help="Output file path (stdout if not specified)"
    )
    
    # Responses command
    responses_parser = subparsers.add_parser(
        "responses",
        help="Side-by-side response comparison"
    )
    responses_parser.add_argument(
        "runs",
        nargs="+",
        help="Run directory paths to compare"
    )
    responses_parser.add_argument(
        "--sample-ids",
        nargs="*",
        help="Specific sample IDs to compare (all if not specified)"
    )
    responses_parser.add_argument(
        "-n", "--limit",
        type=int,
        help="Limit number of samples"
    )
    responses_parser.add_argument(
        "-o", "--output",
        help="Output file path (stdout if not specified)"
    )
    
    # Disagreements command
    disagree_parser = subparsers.add_parser(
        "disagreements",
        help="Find samples where models disagree"
    )
    disagree_parser.add_argument(
        "runs",
        nargs="+",
        help="Run directory paths to compare"
    )
    disagree_parser.add_argument(
        "--label-column",
        default="correct",
        help="Label column for correctness (default: correct)"
    )
    disagree_parser.add_argument(
        "-o", "--output",
        help="Output file path (stdout if not specified)"
    )
    
    # Aggregate command
    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate results across multiple tasks"
    )
    agg_parser.add_argument(
        "runs_dir",
        help="Directory containing run directories organized by task"
    )
    agg_parser.add_argument(
        "--accuracy-column",
        help="Label column containing accuracy values"
    )
    agg_parser.add_argument(
        "--format",
        choices=["table", "markdown", "csv"],
        default="table",
        help="Output format (default: table)"
    )
    agg_parser.add_argument(
        "-o", "--output",
        help="Output file path (stdout if not specified)"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "summary":
        _cmd_summary(args)
    elif args.command == "responses":
        _cmd_responses(args)
    elif args.command == "disagreements":
        _cmd_disagreements(args)
    elif args.command == "aggregate":
        _cmd_aggregate(args)


def _cmd_summary(args):
    """Execute summary comparison command."""
    runs = [Path(r) for r in args.runs]
    result = compare_runs(runs, accuracy_column=args.accuracy_column)
    
    if args.format == "markdown":
        output = result.to_markdown()
    elif args.format == "json":
        output = result.to_json()
    elif args.format == "csv":
        output = result.to_dataframe().to_csv(index=False)
    else:  # table
        output = result.to_dataframe().to_string(index=False)
    
    _write_output(output, args.output)


def _cmd_responses(args):
    """Execute responses comparison command."""
    runs = [Path(r) for r in args.runs]
    df = compare_by_sample_id(runs)
    
    if args.sample_ids:
        df = df[df["sample_id"].isin(args.sample_ids)]
    
    if args.limit:
        df = df.head(args.limit)
    
    output = df.to_string(index=False)
    _write_output(output, args.output)


def _cmd_disagreements(args):
    """Execute disagreements command."""
    runs = [Path(r) for r in args.runs]
    df = find_disagreements(runs, label_column=args.label_column)
    
    if df.empty:
        output = "No disagreements found."
    else:
        output = f"Found {len(df)} disagreements:\n\n{df.to_string(index=False)}"
    
    _write_output(output, args.output)


def _cmd_aggregate(args):
    """Execute aggregate comparison command."""
    runs_dir = Path(args.runs_dir)
    
    # Discover runs organized by task
    runs_by_task = {}
    
    for task_dir in runs_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        task_runs = []
        for run_dir in task_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "manifest.json").exists():
                task_runs.append(run_dir)
        
        if task_runs:
            runs_by_task[task_dir.name] = task_runs
    
    if not runs_by_task:
        print(f"No runs found in {runs_dir}", file=sys.stderr)
        sys.exit(1)
    
    df = aggregate_across_tasks(runs_by_task, accuracy_column=args.accuracy_column)
    
    if args.format == "markdown":
        output = df.to_markdown()
    elif args.format == "csv":
        output = df.to_csv()
    else:  # table
        output = df.to_string()
    
    _write_output(output, args.output)


def _write_output(content: str, output_path: Optional[str]):
    """Write content to file or stdout."""
    if output_path:
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Output written to {output_path}")
    else:
        print(content)


if __name__ == "__main__":
    main()
