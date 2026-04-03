import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def validate_run(run_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    if not run_dir.exists():
        return {
            'is_valid': False,
            'errors': [f'Run directory does not exist: {run_dir}'],
            'warnings': [],
            'n_errors': 1,
            'n_warnings': 0,
            'details': {},
        }

    for fname in ('manifest.json', 'data.parquet', 'tensors.zarr'):
        if not (run_dir / fname).exists():
            errors.append(f'Missing required path: {fname}')

    success_path = run_dir / '_SUCCESS'
    if success_path.exists():
        details['completed'] = True
        try:
            details['completion_info'] = json.loads(success_path.read_text(encoding='utf-8'))
        except Exception:
            warnings.append('Could not parse _SUCCESS as JSON')
    else:
        details['completed'] = False
        warnings.append('Run not marked as complete (missing _SUCCESS)')

    if not errors:
        try:
            from openact_core.io.run import Run

            run = Run(run_dir)
            issues = run.validate()
            if issues:
                errors.extend(issues)
            if verbose:
                details['run'] = {
                    'n_samples': len(run),
                    'n_valid': run.n_valid,
                    'model': run.manifest.model.name,
                    'dataset': run.manifest.dataset.name,
                    'is_complete': run.is_complete,
                }
        except Exception as exc:
            errors.append(f'Failed to load run: {exc}')

    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'n_errors': len(errors),
        'n_warnings': len(warnings),
        'details': details if verbose else {},
    }


def validate_multiple_runs(run_dirs: List[Path], verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    return {str(Path(d)): validate_run(Path(d), verbose=verbose) for d in run_dirs}


def print_validation_result(run_dir: str, result: Dict[str, Any], show_details: bool = False) -> None:
    status = '✓ VALID' if result['is_valid'] else '✗ INVALID'
    print(f'\n{run_dir}: {status}')
    if result['errors']:
        print('  Errors:')
        for err in result['errors']:
            print(f'    - {err}')
    if result['warnings']:
        print('  Warnings:')
        for warn in result['warnings']:
            print(f'    - {warn}')
    if show_details and result.get('details'):
        print('  Details:')
        for key, value in result['details'].items():
            print(f'    {key}: {value}')


def main() -> None:
    parser = argparse.ArgumentParser(prog='openact-validate', description='Validate OpenAct run directories.')
    parser.add_argument('run_dirs', nargs='+', help='Run directory path(s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show extra details')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--strict', action='store_true', help='Treat warnings as errors')
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    if len(run_dirs) == 1 and run_dirs[0].is_dir() and not (run_dirs[0] / 'manifest.json').exists():
        run_dirs = sorted([d for d in run_dirs[0].iterdir() if d.is_dir() and (d / 'manifest.json').exists()])
        if not run_dirs:
            print(f'No run directories found in {args.run_dirs[0]}', file=sys.stderr)
            sys.exit(1)

    results = validate_multiple_runs(run_dirs, verbose=args.verbose)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for run_dir, result in results.items():
            print_validation_result(run_dir, result, show_details=args.verbose)
        n_valid = sum(1 for r in results.values() if r['is_valid'])
        n_total = len(results)
        n_warnings = sum(r['n_warnings'] for r in results.values())
        print(f"\n{'=' * 50}")
        print(f'Summary: {n_valid}/{n_total} valid, {n_warnings} warnings')

    all_valid = all(r['is_valid'] for r in results.values())
    if args.strict:
        all_valid = all_valid and all(r['n_warnings'] == 0 for r in results.values())
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
