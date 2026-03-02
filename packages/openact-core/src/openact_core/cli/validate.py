"""OpenAct Run Validation CLI.

Validates run directories for data integrity and completeness.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import zarr


def validate_run(run_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    """Validate a run directory for data integrity.
    
    Args:
        run_dir: Path to run directory.
        verbose: If True, include detailed info in output.
    
    Returns:
        Validation result dict with 'is_valid', 'errors', 'warnings' keys.
    """
    run_dir = Path(run_dir)
    
    if not run_dir.exists():
        return {
            "is_valid": False,
            "errors": [f"Run directory does not exist: {run_dir}"],
            "warnings": [],
            "details": {},
        }
    
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}
    
    # 1. Check required files
    required_files = ["manifest.json", "data.parquet"]
    for fname in required_files:
        fpath = run_dir / fname
        if not fpath.exists():
            errors.append(f"Missing required file: {fname}")
        elif verbose:
            details[fname] = {"exists": True, "size_bytes": fpath.stat().st_size}
    
    # 2. Check _SUCCESS marker
    success_file = run_dir / "_SUCCESS"
    if success_file.exists():
        details["completed"] = True
        try:
            with open(success_file) as f:
                completion_info = json.load(f)
                details["completion_info"] = completion_info
        except Exception as e:
            warnings.append(f"Could not read _SUCCESS file: {e}")
    else:
        warnings.append("Run not marked as complete (no _SUCCESS file)")
        details["completed"] = False
    
    # 3. Validate manifest
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            required_fields = ["model", "dataset"]
            for field in required_fields:
                if field not in manifest:
                    errors.append(f"Manifest missing required field: {field}")
            
            if verbose:
                details["manifest"] = {
                    "model": manifest.get("model", {}).get("name"),
                    "dataset": manifest.get("dataset", {}).get("name"),
                    "n_samples": manifest.get("stats", {}).get("n_samples_total"),
                }
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in manifest.json: {e}")
        except Exception as e:
            errors.append(f"Error reading manifest.json: {e}")
    
    # 4. Validate parquet
    parquet_path = run_dir / "data.parquet"
    if parquet_path.exists():
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_path)
            n_rows = table.num_rows
            
            required_columns = ["sample_idx", "status", "response_text"]
            missing_cols = [col for col in required_columns if col not in table.column_names]
            if missing_cols:
                errors.append(f"Parquet missing required columns: {missing_cols}")
            
            if verbose:
                details["parquet"] = {
                    "n_rows": n_rows,
                    "columns": table.column_names,
                }
        except ImportError:
            warnings.append("pyarrow not installed; skipping parquet validation")
        except Exception as e:
            errors.append(f"Error reading data.parquet: {e}")
    
    # 5. Validate Zarr
    zarr_path = run_dir / "tensors.zarr"
    if not zarr_path.exists():
        errors.append("Missing tensors.zarr directory")
    else:
        try:
            z = zarr.open_group(str(zarr_path), mode='r')
            
            # Check for consolidated metadata
            zmetadata_path = zarr_path / ".zmetadata"
            if not zmetadata_path.exists():
                warnings.append("Zarr metadata not consolidated (may be slow for remote access)")
            
            # Check token arrays
            if "tokens" in z:
                tokens_group = z["tokens"]
                
                # Check required arrays
                required_arrays = ["ids", "sample_ptr"]
                for arr_name in required_arrays:
                    if arr_name not in tokens_group:
                        errors.append(f"Missing Zarr array: tokens/{arr_name}")
                
                # Validate consistency
                if "ids" in tokens_group and "sample_ptr" in tokens_group:
                    ids = tokens_group["ids"]
                    sample_ptr = tokens_group["sample_ptr"][:]
                    
                    n_tokens = len(ids)
                    expected_tokens = int(sample_ptr[-1]) if len(sample_ptr) > 0 else 0
                    
                    if n_tokens != expected_tokens:
                        errors.append(
                            f"Token count mismatch: ids has {n_tokens}, "
                            f"sample_ptr indicates {expected_tokens}"
                        )
                    
                    # Check monotonicity
                    for i in range(1, len(sample_ptr)):
                        if sample_ptr[i] < sample_ptr[i - 1]:
                            errors.append(f"Non-monotonic sample_ptr at index {i}")
                            break
                    
                    n_samples = len(sample_ptr) - 1 if len(sample_ptr) > 0 else 0
                    
                    if verbose:
                        details["zarr_tokens"] = {
                            "n_tokens": n_tokens,
                            "n_samples": n_samples,
                        }
            else:
                errors.append("Missing Zarr group: tokens")
            
            # Check hidden states
            if "hidden_states" in z:
                hs_group = z["hidden_states"]
                if verbose:
                    details["zarr_hidden_states"] = {
                        "arrays": list(hs_group.keys()),
                    }
            else:
                warnings.append("No hidden_states group in Zarr (may be intentional)")
            
        except Exception as e:
            errors.append(f"Error reading tensors.zarr: {e}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "details": details if verbose else {},
    }


def validate_multiple_runs(run_dirs: List[Path], verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """Validate multiple run directories.
    
    Args:
        run_dirs: List of run directory paths.
        verbose: If True, include detailed info.
    
    Returns:
        Dict mapping run path to validation result.
    """
    results = {}
    for run_dir in run_dirs:
        results[str(run_dir)] = validate_run(run_dir, verbose=verbose)
    return results


def print_validation_result(run_dir: str, result: Dict[str, Any], show_details: bool = False) -> None:
    """Print validation result in human-readable format."""
    status = "✓ VALID" if result["is_valid"] else "✗ INVALID"
    print(f"\n{run_dir}: {status}")
    
    if result["errors"]:
        print("  Errors:")
        for err in result["errors"]:
            print(f"    - {err}")
    
    if result["warnings"]:
        print("  Warnings:")
        for warn in result["warnings"]:
            print(f"    - {warn}")
    
    if show_details and result.get("details"):
        print("  Details:")
        for key, value in result["details"].items():
            if isinstance(value, dict):
                print(f"    {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v}")
            else:
                print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        prog="openact-validate",
        description="Validate OpenAct run directories for data integrity."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Run directory path(s) to validate"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed validation info"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    run_dirs = [Path(d) for d in args.run_dirs]
    
    # If a single directory containing multiple runs is given, expand it
    if len(run_dirs) == 1 and run_dirs[0].is_dir():
        # Check if this is a run directory itself
        if (run_dirs[0] / "manifest.json").exists():
            pass  # It's a run directory
        else:
            # It's a directory of runs
            run_dirs = sorted([
                d for d in run_dirs[0].iterdir()
                if d.is_dir() and (d / "manifest.json").exists()
            ])
            if not run_dirs:
                print(f"No run directories found in {args.run_dirs[0]}", file=sys.stderr)
                sys.exit(1)
    
    results = validate_multiple_runs(run_dirs, verbose=args.verbose)
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for run_dir, result in results.items():
            print_validation_result(run_dir, result, show_details=args.verbose)
        
        # Summary
        n_valid = sum(1 for r in results.values() if r["is_valid"])
        n_total = len(results)
        n_warnings = sum(r["n_warnings"] for r in results.values())
        
        print(f"\n{'='*50}")
        print(f"Summary: {n_valid}/{n_total} runs valid, {n_warnings} warnings")
    
    # Exit code
    all_valid = all(r["is_valid"] for r in results.values())
    if args.strict:
        all_valid = all_valid and all(r["n_warnings"] == 0 for r in results.values())
    
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
