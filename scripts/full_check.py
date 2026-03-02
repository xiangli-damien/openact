#!/usr/bin/env python3
"""
OpenAct 完整检查脚本：结构验证 + 内容抽查 + Hidden States 校验。

整合 validation_run、check_prompts_and_answers、smoke_test_hidden_states 的逻辑，
对 runs 下所有任务做一次性完整检查。

Usage:
  python scripts/full_check.py runs/Qwen2-7B-Instruct
  python scripts/full_check.py runs --fix
  python scripts/full_check.py runs --skip-hidden-states
  python scripts/full_check.py runs --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def find_run_dirs(root: Path) -> List[Tuple[str, Path]]:
    """找到所有包含 data.parquet 的 run 目录。返回 (run_name, path) 列表。

    支持两种结构：
      - runs/Qwen2-7B-Instruct  → 直接子目录 gsm8k, math, ... 为 run
      - runs                    → 先找 model 子目录，再找其下的 task run
    """
    root = Path(root)
    if not root.is_dir():
        return []
    runs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "data.parquet").exists():
            runs.append((d.name, d))
        else:
            # 可能是 runs/ 结构，d 为 model 子目录
            for sd in sorted(d.iterdir()):
                if sd.is_dir() and (sd / "data.parquet").exists():
                    runs.append((f"{d.name}/{sd.name}", sd))
    return runs


def validate_single_run(run_dir: Path, fix: bool = False) -> Dict[str, Any]:
    """单 run 结构验证（复用 validation_run 逻辑）。"""
    from openact_collect.engine.completion_marker import CompletionMarker

    results: Dict[str, Any] = {
        "run_name": run_dir.name,
        "checks": {},
        "errors": [],
        "warnings": [],
        "n_samples": None,
        "model": "unknown",
        "dataset": "unknown",
    }

    # 1. 必需文件
    required = ["manifest.json", "data.parquet", "tensors.zarr"]
    for fname in required:
        exists = (run_dir / fname).exists()
        results["checks"][f"file_{fname}"] = exists
        if not exists:
            results["errors"].append(f"Missing: {fname}")

    # 2. 完成标记
    is_complete = CompletionMarker.check_complete(run_dir)
    results["checks"]["has_success_marker"] = is_complete
    if not is_complete:
        results["warnings"].append("No _SUCCESS marker")
        if fix:
            CompletionMarker.mark_complete(run_dir)
            results["checks"]["has_success_marker"] = True

    # 3. Zarr 元数据
    zarr_path = run_dir / "tensors.zarr"
    if zarr_path.exists():
        try:
            import zarr

            z = zarr.open_group(str(zarr_path), mode="r")
            zmetadata_path = zarr_path / ".zmetadata"
            has_consolidated = zmetadata_path.exists()
            results["checks"]["zarr_consolidated"] = has_consolidated
            if not has_consolidated:
                results["warnings"].append("Zarr metadata not consolidated")
                if fix:
                    zarr.consolidate_metadata(str(zarr_path))
                    results["checks"]["zarr_consolidated"] = True

            if "tokens/sample_ptr" in z and "tokens/ids" in z:
                sample_ptr = z["tokens/sample_ptr"][:]
                n_exp = int(sample_ptr[-1])
                n_act = len(z["tokens/ids"])
                match = n_exp == n_act
                results["checks"]["token_count_match"] = match
                if not match:
                    results["errors"].append(
                        f"Token count mismatch: {n_exp} vs {n_act}"
                    )
        except Exception as e:
            results["errors"].append(f"Zarr error: {e}")

    # 4. Run 加载与 alignment
    if (run_dir / "manifest.json").exists():
        try:
            from openact_core import Run

            run = Run(run_dir, require_complete=False)
            results["n_samples"] = len(run)
            results["model"] = run.manifest.model.name or "unknown"
            results["dataset"] = run.manifest.dataset.name or "unknown"

            # 抽样检查 offset 覆盖
            indices = [0, len(run) // 2, len(run) - 1]
            indices = [i for i in indices if 0 <= i < len(run)]
            alignment_ok = True
            for idx in indices:
                try:
                    sample = run[idx]
                    if (
                        sample.token_offsets is not None
                        and len(sample.token_offsets) > 0
                        and len(sample.response_text or "") > 0
                    ):
                        cov = sample.token_offsets[-1, 1] / len(
                            sample.response_text
                        )
                        if cov < 0.5:
                            results["warnings"].append(
                                f"Sample {idx}: low offset coverage ({cov:.1%})"
                            )
                            alignment_ok = False
                except Exception as e:
                    results["warnings"].append(f"Sample {idx}: {e}")
            results["checks"]["alignment_ok"] = alignment_ok
        except Exception as e:
            results["errors"].append(f"Run load error: {e}")

    results["is_valid"] = len(results["errors"]) == 0
    return results


def check_content_summary(run_dir: Path) -> Dict[str, Any]:
    """内容摘要：样本数、labels、accuracy 等。"""
    info: Dict[str, Any] = {"n_samples": 0, "has_labels": False, "accuracy": None}
    try:
        df = pd.read_parquet(run_dir / "data.parquet")
        info["n_samples"] = len(df)
    except Exception:
        return info

    labels_path = run_dir / "labels" / "correctness.parquet"
    if labels_path.exists():
        try:
            labels_df = pd.read_parquet(labels_path)
            info["has_labels"] = True
            if "is_correct" in labels_df.columns:
                n_correct = (labels_df["is_correct"] == True).sum()
                n_total = len(labels_df)
                info["accuracy"] = (
                    round(n_correct / n_total, 4) if n_total > 0 else None
                )
        except Exception:
            pass
    return info


def check_hidden_states(run_dir: Path, max_samples: int = 2) -> Tuple[bool, List[str]]:
    """Hidden states 校验。返回 (passed, messages)。"""
    try:
        from openact_core import Run

        run = Run(run_dir, require_complete=False)
        if "hidden_states/per_token" not in run._zarr:
            return True, ["No per_token hidden states; skip HS check"]

        model = run.manifest.model
        n_layers = model.n_layers or 0
        hidden_dim = model.hidden_dim or 0
        errors = []
        checked = 0
        for sample in run.iter_valid():
            if checked >= max_samples:
                break
            try:
                hs = sample.hidden_states
                pl = sample.prompt_last_hidden_states
                n_tokens = sample.n_tokens
                gen_only = sample.generated_only_hidden_states

                if len(hs.shape) != 3:
                    errors.append(
                        f"Sample {sample.index}: hs.shape={hs.shape} (expected 3D)"
                    )
                elif n_layers and hs.shape[1] != n_layers:
                    errors.append(
                        f"Sample {sample.index}: hs.shape[1]={hs.shape[1]} != n_layers={n_layers}"
                    )
                if n_tokens > 0 and len(gen_only) != n_tokens:
                    errors.append(
                        f"Sample {sample.index}: gen_only.len={len(gen_only)} != n_tokens={n_tokens}"
                    )
                checked += 1
            except Exception as e:
                errors.append(f"Sample {sample.index}: {e}")
        if checked == 0:
            return True, ["No valid samples with HS"]
        return len(errors) == 0, errors if errors else [f"HS OK ({checked} samples)"]
    except Exception as e:
        return False, [str(e)]


def load_smoke_report(root: Path) -> Optional[Dict]:
    """加载 smoke_test_report.json（若存在）。"""
    candidates = [
        root / "smoke_test_report.json",
        root.parent / "smoke_test_report.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def run_full_check(
    root: Path,
    fix: bool = False,
    skip_hidden_states: bool = False,
    hs_max_samples: int = 2,
) -> Dict[str, Any]:
    """执行完整检查。"""
    root = Path(root)
    runs = find_run_dirs(root)
    if not runs:
        return {
            "root": str(root),
            "error": f"No run dirs found under {root}",
            "runs": [],
        }

    report: Dict[str, Any] = {
        "root": str(root),
        "n_runs": len(runs),
        "runs": [],
        "summary": {"valid": 0, "invalid": 0, "with_labels": 0},
    }

    smoke_report = load_smoke_report(root)
    if smoke_report:
        report["smoke_test_report"] = smoke_report

    for run_name, run_dir in runs:
        # 1. 结构验证
        val = validate_single_run(run_dir, fix=fix)
        # 2. 内容摘要
        content = check_content_summary(run_dir)
        val["content"] = content
        # 3. Hidden states（可选）
        if not skip_hidden_states:
            hs_ok, hs_msgs = check_hidden_states(run_dir, max_samples=hs_max_samples)
            val["hidden_states_ok"] = hs_ok
            val["hidden_states_messages"] = hs_msgs

        report["runs"].append(val)
        if val["is_valid"]:
            report["summary"]["valid"] += 1
        else:
            report["summary"]["invalid"] += 1
        if content.get("has_labels"):
            report["summary"]["with_labels"] += 1

    report["summary"]["all_valid"] = report["summary"]["invalid"] == 0
    return report


def print_report(report: Dict[str, Any]) -> None:
    """打印人类可读的报告。"""
    if "error" in report:
        print(f"Error: {report['error']}")
        return

    root = report["root"]
    n_runs = report["n_runs"]
    print()
    print("=" * 70)
    print(f"  OpenAct 完整检查报告")
    print(f"  目录: {root}")
    print(f"  Run 数量: {n_runs}")
    print("=" * 70)

    # 汇总表
    print()
    header = f"{'Run':<20} {'Valid':>6} {'Samples':>8} {'Labels':>6} {'Acc':>8} {'HS':>6}"
    print(header)
    print("-" * len(header))

    for r in report["runs"]:
        run_name = r.get("run_name", "?")
        valid = "✓" if r.get("is_valid") else "✗"
        n = r.get("n_samples") or 0
        has_labels = "✓" if r.get("content", {}).get("has_labels") else "-"
        acc = r.get("content", {}).get("accuracy")
        acc_str = f"{acc:.1%}" if acc is not None else "-"
        hs_ok = r.get("hidden_states_ok")
        hs_str = "✓" if hs_ok is True else ("✗" if hs_ok is False else "-")
        print(f"{run_name:<20} {valid:>6} {n:>8} {has_labels:>6} {acc_str:>8} {hs_str:>6}")

    print("-" * len(header))
    s = report["summary"]
    print(f"Valid: {s['valid']}/{n_runs}  |  With labels: {s['with_labels']}/{n_runs}")
    print()

    # 错误与警告
    has_issues = False
    for r in report["runs"]:
        if r.get("errors"):
            has_issues = True
            print(f"  [{r['run_name']}] ERRORS:")
            for e in r["errors"]:
                print(f"    ✗ {e}")
        if r.get("warnings"):
            has_issues = True
            print(f"  [{r['run_name']}] Warnings:")
            for w in r["warnings"][:5]:
                print(f"    ⚠ {w}")
            if len(r["warnings"]) > 5:
                print(f"    ... and {len(r['warnings'])-5} more")
        if r.get("hidden_states_ok") is False and r.get("hidden_states_messages"):
            has_issues = True
            print(f"  [{r['run_name']}] Hidden states:")
            for m in r["hidden_states_messages"]:
                print(f"    ✗ {m}")

    if smoke_report := report.get("smoke_test_report"):
        print()
        print("=" * 70)
        print("  Smoke Test 报告")
        print("=" * 70)
        print(f"  Model: {smoke_report.get('model', '?')}")
        print(f"  Total time: {smoke_report.get('overall_time_s', '?')}s")
        print(f"  Collected: {smoke_report.get('n_collected', '?')}/{smoke_report.get('n_tasks', '?')}")
        print()

    print("=" * 70)
    if report["summary"]["all_valid"] and not has_issues:
        print("  ✓ 全部检查通过")
    else:
        print("  ✗ 存在错误或警告，请查看上方详情")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="OpenAct 完整检查：结构 + 内容 + Hidden States",
    )
    parser.add_argument(
        "path",
        type=Path,
        default=Path("runs"),
        nargs="?",
        help="runs 根目录，如 runs 或 runs/Qwen2-7B-Instruct",
    )
    parser.add_argument("--fix", action="store_true", help="尝试修复（如补齐 _SUCCESS、consolidate zarr）")
    parser.add_argument(
        "--skip-hidden-states",
        action="store_true",
        help="跳过 hidden states 校验（更快）",
    )
    parser.add_argument(
        "--hs-max-samples",
        type=int,
        default=2,
        help="Hidden states 检查的样本数 (default: 2)",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")

    args = parser.parse_args()
    path = args.path if args.path.exists() else Path("runs")

    report = run_full_check(
        path,
        fix=args.fix,
        skip_hidden_states=args.skip_hidden_states,
        hs_max_samples=args.hs_max_samples,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    ok = report.get("summary", {}).get("all_valid", False) and "error" not in report
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
