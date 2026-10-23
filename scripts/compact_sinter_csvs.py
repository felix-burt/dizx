"""Collapse per-batch rows in sinter CSVs into one row per strong_id.

Usage:
    python scripts/compact_sinter_csvs.py <path> [<path> ...] [--apply]

Each <path> is a CSV file or a directory (searched recursively for *.csv).
run_threshold_experiment compacts each CSV after every collection, so this is
for results produced elsewhere (e.g. synced from the cluster by older code).
See dizx.sinter_csv for how rewrites are verified.

Dry run by default; pass --apply to rewrite files.
"""
import argparse
import sys
from pathlib import Path

from dizx.sinter_csv import compact_sinter_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--apply', action='store_true', help='Rewrite files (default: dry run)')
    args = parser.parse_args()

    files = []
    for p in map(Path, args.paths):
        files += sorted(p.rglob('*.csv')) if p.is_dir() else [p]

    before = after = 0
    failures = []
    for f in files:
        try:
            b, a = compact_sinter_csv(f, apply=args.apply)
        except (RuntimeError, ValueError, KeyError, StopIteration) as e:
            failures.append(f'{f}: {e}')
            continue
        if a != b:
            print(f'  {f}: {b / 1e6:.1f}MB -> {a / 1e6:.2f}MB')
        before += b
        after += a

    print(f'\n{len(files)} files: {before / 1e9:.2f}GB -> {after / 1e9:.2f}GB')
    for msg in failures:
        print(f'FAILED {msg}')
    if not args.apply:
        print('Dry run only. Re-run with --apply to rewrite files.')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
