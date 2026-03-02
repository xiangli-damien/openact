#!/usr/bin/env python3
"""
采集完成后运行此脚本，生成所有数据集的汇总分析报告。
用法: python analyze_all_runs.py runs/
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from openact_core import Run, SampleStatus
from openact_core.tasks import get_parser, list_parsers

def analyze_single_run(run_dir: Path) -> dict:
    """分析单个 run，返回汇总信息。"""
    try:
        run = Run(run_dir)
    except Exception as e:
        return {"path": str(run_dir), "error": str(e)}

    stats = run.stats
    manifest = run.manifest
    
    result = {
        "path": str(run_dir),
        "task": manifest.dataset.name,
        "model": manifest.model.name,
        "n_total": stats.n_samples_total,
        "n_valid": stats.n_samples_ok,
        "n_error": stats.n_samples_error,
        "n_tokens": stats.n_tokens_total,
        "duration_s": stats.duration_seconds,
        "is_complete": run.is_complete,
        "n_layers": run.n_layers,
        "hidden_dim": run.hidden_dim,
    }

    # 计算每样本平均 token 数
    if stats.n_samples_ok > 0:
        token_lengths = [s.n_tokens for s in run.iter_valid()]
        result["avg_tokens"] = np.mean(token_lengths)
        result["min_tokens"] = int(np.min(token_lengths))
        result["max_tokens"] = int(np.max(token_lengths))
    
    # 计算 accuracy（如果有 labels）
    labels_dir = run_dir / "labels"
    if labels_dir.exists() and list(labels_dir.glob("*.parquet")):
        try:
            labels = run.get_all_labels("is_correct", only_valid=True)
            n_correct = sum(1 for v in labels if v is True or v == True)
            n_total_labeled = len(labels)
            result["accuracy"] = n_correct / n_total_labeled if n_total_labeled > 0 else 0
            result["n_correct"] = n_correct
            result["n_labeled"] = n_total_labeled
        except Exception:
            pass

    # Hidden states 形状验证
    try:
        sample = run[0]
        hs = sample.hidden_states
        result["hs_shape"] = list(hs.shape)
        result["hs_dtype"] = str(hs.dtype)
        
        # 检查 alignment
        aligner = sample.aligner
        issues = aligner.validate()
        result["alignment_issues"] = len(issues)
    except Exception as e:
        result["hs_error"] = str(e)

    return result


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_all_runs.py <runs_dir>")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"目录不存在: {base_dir}")
        sys.exit(1)

    # 找到所有 run 目录（有 manifest.json 的）
    run_dirs = sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    ])

    if not run_dirs:
        print(f"在 {base_dir} 下未找到任何 run")
        sys.exit(1)

    print("=" * 80)
    print(f"  OpenAct 全数据集分析报告")
    print(f"  扫描目录: {base_dir}")
    print(f"  发现 {len(run_dirs)} 个 run")
    print("=" * 80)

    results = []
    for run_dir in run_dirs:
        print(f"\n分析: {run_dir.name} ...", end=" ", flush=True)
        info = analyze_single_run(run_dir)
        results.append(info)
        if "error" in info:
            print(f"❌ {info['error']}")
        else:
            acc_str = f", acc={info['accuracy']:.1%}" if "accuracy" in info else ""
            print(f"✓ {info['n_valid']}/{info['n_total']} valid{acc_str}")

    # ─── 汇总表格 ────────────────────────────────────────────────────────
    print("\n")
    print("=" * 80)
    print("  汇总表格")
    print("=" * 80)

    # 表头
    header = f"{'Task':<16} {'Valid':>6} {'Error':>6} {'Tokens':>10} {'Avg Tok':>8} {'Acc':>8} {'Time':>8} {'Status':>8}"
    print(header)
    print("-" * len(header))

    total_valid = 0
    total_error = 0
    total_tokens = 0

    for r in results:
        if "error" in r:
            print(f"{r.get('task', '?'):<16} {'ERROR':>50}")
            continue

        task = r.get("task", "?")
        n_valid = r.get("n_valid", 0)
        n_error = r.get("n_error", 0)
        n_tokens = r.get("n_tokens", 0)
        avg_tok = r.get("avg_tokens", 0)
        acc = r.get("accuracy")
        duration = r.get("duration_s")
        complete = "✓" if r.get("is_complete") else "⚠"

        acc_str = f"{acc:.1%}" if acc is not None else "-"
        dur_str = f"{duration:.0f}s" if duration else "-"

        print(f"{task:<16} {n_valid:>6} {n_error:>6} {n_tokens:>10,} {avg_tok:>8.0f} {acc_str:>8} {dur_str:>8} {complete:>8}")

        total_valid += n_valid
        total_error += n_error
        total_tokens += n_tokens

    print("-" * len(header))
    print(f"{'TOTAL':<16} {total_valid:>6} {total_error:>6} {total_tokens:>10,}")

    # ─── Hidden States 验证 ───────────────────────────────────────────────
    print("\n")
    print("=" * 80)
    print("  Hidden States 验证")
    print("=" * 80)

    for r in results:
        if "error" in r:
            continue
        task = r.get("task", "?")
        hs_shape = r.get("hs_shape", "N/A")
        hs_dtype = r.get("hs_dtype", "N/A")
        align_issues = r.get("alignment_issues", "N/A")
        print(f"  {task:<16} shape={hs_shape}  dtype={hs_dtype}  alignment_issues={align_issues}")

    # ─── Accuracy 排名 ───────────────────────────────────────────────────
    acc_results = [(r["task"], r["accuracy"]) for r in results if "accuracy" in r]
    if acc_results:
        print("\n")
        print("=" * 80)
        print("  Accuracy 排名")
        print("=" * 80)
        acc_results.sort(key=lambda x: x[1], reverse=True)
        for (task, acc) in acc_results:
            bar = "█" * int(acc * 40)
            print(f"  {task:<16} {acc:>6.1%}  {bar}")

    # ─── 保存 JSON 报告 ──────────────────────────────────────────────────
    report_path = base_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()