"""
Cross-run comparison utilities for OpenAct.

Provides tools for comparing results from different runs (e.g., different
models on the same task) and generating comparison tables/reports.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json

import numpy as np
import pandas as pd

from openact_core.io.run import Run
from openact_core.schema.status import SampleStatus


@dataclass
class RunSummary:
    """Summary statistics for a single run."""
    run_path: str
    model_name: str
    task_name: str
    n_samples: int
    n_ok: int
    n_errors: int
    accuracy: Optional[float] = None
    mean_response_length: Optional[float] = None
    mean_processing_time: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Fraction of samples with OK status."""
        return self.n_ok / self.n_samples if self.n_samples > 0 else 0.0


@dataclass 
class ComparisonResult:
    """Result of comparing multiple runs."""
    task_name: str
    runs: List[RunSummary]
    per_sample_df: Optional[pd.DataFrame] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert run summaries to a comparison DataFrame."""
        rows = []
        for run in self.runs:
            rows.append({
                "run_path": run.run_path,
                "model": run.model_name,
                "task": run.task_name,
                "n_samples": run.n_samples,
                "n_ok": run.n_ok,
                "n_errors": run.n_errors,
                "success_rate": run.success_rate,
                "accuracy": run.accuracy,
                "mean_response_len": run.mean_response_length,
                "mean_time_s": run.mean_processing_time,
                **run.custom_metrics,
            })
        return pd.DataFrame(rows)
    
    def to_markdown(self) -> str:
        """Generate a markdown comparison table."""
        df = self.to_dataframe()
        
        # Format numeric columns
        for col in ["success_rate", "accuracy"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        
        for col in ["mean_response_len", "mean_time_s"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        
        return df.to_markdown(index=False)
    
    def to_json(self) -> str:
        """Generate JSON representation."""
        return json.dumps({
            "task": self.task_name,
            "runs": [
                {
                    "model": r.model_name,
                    "n_samples": r.n_samples,
                    "success_rate": r.success_rate,
                    "accuracy": r.accuracy,
                    **r.custom_metrics,
                }
                for r in self.runs
            ]
        }, indent=2, default=str)


def summarize_run(
    run: Union[Run, str, Path],
    accuracy_column: Optional[str] = None,
) -> RunSummary:
    """Generate summary statistics for a run.
    
    Args:
        run: Run instance or path.
        accuracy_column: Column name in labels containing accuracy/correctness.
    
    Returns:
        RunSummary with computed statistics.
    """
    if isinstance(run, (str, Path)):
        run = Run(run)
    
    df = run.to_dataframe()
    manifest = run.manifest
    
    n_samples = len(df)
    n_ok = (df["status"] == SampleStatus.OK.value).sum() if "status" in df.columns else len(df)
    n_errors = n_samples - n_ok
    
    # Mean response length
    mean_response_len = None
    if "response_text" in df.columns:
        mean_response_len = df["response_text"].str.len().mean()
    elif "n_response_tokens" in df.columns:
        mean_response_len = df["n_response_tokens"].mean()
    
    # Mean processing time
    mean_time = None
    if "processing_time" in df.columns:
        mean_time = df["processing_time"].mean()
    
    # Accuracy from labels
    accuracy = None
    if accuracy_column:
        try:
            labels = run.get_all_labels(accuracy_column)
            # Handle boolean or numeric labels
            if labels.dtype == bool:
                accuracy = labels.mean()
            elif np.issubdtype(labels.dtype, np.number):
                accuracy = labels.mean()
        except Exception:
            pass
    
    return RunSummary(
        run_path=str(run.run_dir),
        model_name=manifest.model.name if manifest.model else "unknown",
        task_name=manifest.dataset.name if manifest.dataset else "unknown",
        n_samples=n_samples,
        n_ok=n_ok,
        n_errors=n_errors,
        accuracy=accuracy,
        mean_response_length=mean_response_len,
        mean_processing_time=mean_time,
    )


def compare_runs(
    runs: List[Union[Run, str, Path]],
    accuracy_column: Optional[str] = None,
    include_per_sample: bool = False,
) -> ComparisonResult:
    """Compare multiple runs on the same task.
    
    Args:
        runs: List of Run instances or paths.
        accuracy_column: Column name in labels containing accuracy/correctness.
        include_per_sample: If True, include per-sample comparison DataFrame.
    
    Returns:
        ComparisonResult with summaries and optionally per-sample data.
    """
    summaries = []
    per_sample_dfs = []
    
    for run_input in runs:
        if isinstance(run_input, (str, Path)):
            run = Run(run_input)
        else:
            run = run_input
        
        summary = summarize_run(run, accuracy_column=accuracy_column)
        summaries.append(summary)
        
        if include_per_sample:
            df = run.to_dataframe()
            df["_run_model"] = summary.model_name
            df["_run_path"] = summary.run_path
            per_sample_dfs.append(df)
    
    # Determine common task name
    task_names = set(s.task_name for s in summaries)
    task_name = task_names.pop() if len(task_names) == 1 else "/".join(sorted(task_names))
    
    # Merge per-sample data if requested
    per_sample_df = None
    if include_per_sample and per_sample_dfs:
        per_sample_df = pd.concat(per_sample_dfs, ignore_index=True)
    
    return ComparisonResult(
        task_name=task_name,
        runs=summaries,
        per_sample_df=per_sample_df,
    )


def compare_by_sample_id(
    runs: List[Union[Run, str, Path]],
    sample_id_column: str = "sample_id",
    response_column: str = "response_text",
) -> pd.DataFrame:
    """Create a side-by-side comparison of responses by sample ID.
    
    Useful for qualitative analysis of how different models respond
    to the same inputs.
    
    Args:
        runs: List of Run instances or paths.
        sample_id_column: Column containing sample identifiers.
        response_column: Column containing responses to compare.
    
    Returns:
        DataFrame with sample_id and response columns from each run.
    """
    dfs = []
    model_names = []
    
    for run_input in runs:
        if isinstance(run_input, (str, Path)):
            run = Run(run_input)
        else:
            run = run_input
        
        df = run.to_dataframe()
        model_name = run.manifest.model.name if run.manifest.model else f"run_{len(dfs)}"
        model_names.append(model_name)
        
        # Select relevant columns
        cols = [sample_id_column]
        if response_column in df.columns:
            cols.append(response_column)
        if "ground_truth" in df.columns and "ground_truth" not in cols:
            cols.append("ground_truth")
        
        df_subset = df[cols].copy()
        df_subset = df_subset.rename(columns={
            response_column: f"response_{model_name}",
        })
        dfs.append(df_subset)
    
    # Merge on sample_id
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=sample_id_column, how="outer", suffixes=("", "_dup"))
        # Remove duplicate ground_truth columns
        dup_cols = [c for c in result.columns if c.endswith("_dup")]
        result = result.drop(columns=dup_cols, errors="ignore")
    
    return result


def find_disagreements(
    runs: List[Union[Run, str, Path]],
    label_column: str = "correct",
    sample_id_column: str = "sample_id",
) -> pd.DataFrame:
    """Find samples where runs disagree on correctness.
    
    Args:
        runs: List of Run instances or paths.
        label_column: Column in labels containing correctness.
        sample_id_column: Column containing sample identifiers.
    
    Returns:
        DataFrame with samples where at least one run differs.
    """
    results = []
    model_names = []
    
    for run_input in runs:
        if isinstance(run_input, (str, Path)):
            run = Run(run_input)
        else:
            run = run_input
        
        model_name = run.manifest.model.name if run.manifest.model else f"run_{len(results)}"
        model_names.append(model_name)
        
        df = run.to_dataframe()
        
        # Try to get labels
        try:
            labels = run.get_all_labels(label_column)
            df[f"correct_{model_name}"] = labels
        except Exception:
            df[f"correct_{model_name}"] = None
        
        results.append(df[[sample_id_column, f"correct_{model_name}"]].copy())
    
    # Merge all results
    merged = results[0]
    for df in results[1:]:
        merged = merged.merge(df, on=sample_id_column, how="outer")
    
    # Find disagreements
    correct_cols = [f"correct_{m}" for m in model_names]
    
    def has_disagreement(row):
        vals = [row[c] for c in correct_cols if pd.notna(row[c])]
        if len(vals) < 2:
            return False
        return len(set(vals)) > 1
    
    merged["disagreement"] = merged.apply(has_disagreement, axis=1)
    
    return merged[merged["disagreement"]].drop(columns=["disagreement"])


def aggregate_across_tasks(
    runs_by_task: Dict[str, List[Union[Run, str, Path]]],
    accuracy_column: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate comparison across multiple tasks.
    
    Args:
        runs_by_task: Dict mapping task name to list of runs for that task.
        accuracy_column: Column name for accuracy labels.
    
    Returns:
        DataFrame with models as rows and tasks as columns.
    """
    all_results = []
    
    for task_name, runs in runs_by_task.items():
        comparison = compare_runs(runs, accuracy_column=accuracy_column)
        for run_summary in comparison.runs:
            all_results.append({
                "model": run_summary.model_name,
                "task": task_name,
                "accuracy": run_summary.accuracy,
                "success_rate": run_summary.success_rate,
                "n_samples": run_summary.n_samples,
            })
    
    df = pd.DataFrame(all_results)
    
    # Pivot to get models as rows and tasks as columns
    if "accuracy" in df.columns and df["accuracy"].notna().any():
        pivot = df.pivot(index="model", columns="task", values="accuracy")
    else:
        pivot = df.pivot(index="model", columns="task", values="success_rate")
    
    # Add mean across tasks
    pivot["mean"] = pivot.mean(axis=1)
    
    return pivot.sort_values("mean", ascending=False)
