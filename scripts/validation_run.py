#!/usr/bin/env python3
"""
Validate OpenAct run integrity and alignment correctness.

Usage:
    python scripts/validate_run.py /data/runs/my_run
    python scripts/validate_run.py /data/runs/my_run --fix
    python scripts/validate_run.py /data/runs/my_run --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np


def validate_run(run_dir: Path, fix: bool = False) -> Dict[str, Any]:
    """
    Comprehensive run validation.
    
    Args:
        run_dir: Path to run directory
        fix: Whether to attempt fixing issues
        
    Returns:
        Validation results dictionary
    """
    from openact_core import Run
    from openact_collect.engine.completion_marker import CompletionMarker
    
    results: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "checks": {},
        "errors": [],
        "warnings": [],
    }
    
    # 1. Check required files
    print("Checking required files...")
    required = ["manifest.json", "data.parquet", "tensors.zarr"]
    for fname in required:
        exists = (run_dir / fname).exists()
        results["checks"][f"file_{fname}"] = exists
        if not exists:
            results["errors"].append(f"Missing: {fname}")
    
    # 2. Check completion marker
    print("Checking completion marker...")
    is_complete = CompletionMarker.check_complete(run_dir)
    results["checks"]["has_success_marker"] = is_complete
    
    if not is_complete:
        results["warnings"].append("No _SUCCESS marker")
        if fix:
            print("  Marking as complete...")
            CompletionMarker.mark_complete(run_dir)
            results["checks"]["has_success_marker"] = True
    
    # 3. Check Zarr metadata consolidation
    print("Checking Zarr metadata...")
    zarr_path = run_dir / "tensors.zarr"
    if zarr_path.exists():
        import zarr
        try:
            z = zarr.open_group(str(zarr_path), mode='r')
            
            zmetadata_path = zarr_path / ".zmetadata"
            has_consolidated = zmetadata_path.exists()
            results["checks"]["zarr_consolidated"] = has_consolidated
            
            if not has_consolidated:
                results["warnings"].append("Zarr metadata not consolidated")
                if fix:
                    print("  Consolidating metadata...")
                    zarr.consolidate_metadata(str(zarr_path))
                    results["checks"]["zarr_consolidated"] = True
            
            # Check array consistency
            if "tokens/sample_ptr" in z and "tokens/ids" in z:
                sample_ptr = z["tokens/sample_ptr"][:]
                n_tokens_expected = int(sample_ptr[-1])
                n_tokens_actual = len(z["tokens/ids"])
                
                match = n_tokens_expected == n_tokens_actual
                results["checks"]["token_count_match"] = match
                if not match:
                    results["errors"].append(
                        f"Token count mismatch: {n_tokens_expected} vs {n_tokens_actual}"
                    )
            
        except Exception as e:
            results["errors"].append(f"Zarr validation error: {e}")
    
    # 4. Validate alignment (sampling)
    print("Checking alignment...")
    try:
        run = Run(run_dir, require_complete=False)
        n_samples = len(run)
        results["n_samples"] = n_samples
        
        # Sample check
        sample_indices = [0, n_samples // 2, n_samples - 1]
        sample_indices = [i for i in sample_indices if 0 <= i < n_samples]
        
        alignment_ok = True
        for idx in sample_indices:
            try:
                sample = run[idx]
                
                if sample.token_offsets is not None and len(sample.token_offsets) > 0:
                    offsets = sample.token_offsets
                    text = sample.response_text
                    
                    if len(text) > 0:
                        coverage = offsets[-1, 1] / len(text)
                        if coverage < 0.5:
                            results["warnings"].append(
                                f"Sample {idx}: Low offset coverage ({coverage:.2%})"
                            )
                            alignment_ok = False
            except Exception as e:
                results["warnings"].append(f"Sample {idx}: {e}")
        
        results["checks"]["alignment_ok"] = alignment_ok
        
    except Exception as e:
        results["errors"].append(f"Run loading error: {e}")
    
    # 5. Extract statistics
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            results["stats"] = manifest.get("stats", {})
            results["model"] = manifest.get("model", {}).get("name", "unknown")
            results["dataset"] = manifest.get("dataset", {}).get("name", "unknown")
        except Exception:
            pass
    
    # Summary
    results["is_valid"] = len(results["errors"]) == 0
    results["n_errors"] = len(results["errors"])
    results["n_warnings"] = len(results["warnings"])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate OpenAct Run")
    parser.add_argument("run_dir", type=Path, help="Run directory to validate")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if not args.run_dir.exists():
        print(f"Error: {args.run_dir} does not exist")
        sys.exit(1)
    
    results = validate_run(args.run_dir, fix=args.fix)
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print(f"Validation Results: {args.run_dir}")
        print("=" * 60)
        
        print(f"\nModel: {results.get('model', 'unknown')}")
        print(f"Dataset: {results.get('dataset', 'unknown')}")
        print(f"Samples: {results.get('n_samples', 'unknown')}")
        
        print(f"\nChecks:")
        for check, passed in results["checks"].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        if results["warnings"]:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for w in results["warnings"]:
                print(f"  ⚠ {w}")
        
        if results["errors"]:
            print(f"\nErrors ({len(results['errors'])}):")
            for e in results["errors"]:
                print(f"  ✗ {e}")
        
        print("\n" + "=" * 60)
        if results["is_valid"]:
            print("✓ VALID")
        else:
            print("✗ INVALID")
        print("=" * 60)
    
    sys.exit(0 if results["is_valid"] else 1)


if __name__ == "__main__":
    main()