"""Compaction of sinter result CSVs to one row per strong_id.

sinter.collect appends one row per finished batch, so long runs accumulate
10^4-10^5 rows per task, each repeating the full json_metadata. Folding them
(sum of shots/errors/discards/seconds/custom_counts) leaves every reader's
view unchanged -- sinter.read_stats_from_csv_files and resume both fold by
strong_id anyway -- while shrinking files ~100x. Only the per-batch timing
breakdown is lost.
"""
import collections
import csv
import json
import sys
from pathlib import Path

import sinter

csv.field_size_limit(sys.maxsize)


def _raw_totals(path: Path) -> tuple[int, dict]:
    """Row count and per-strong_id totals from a plain csv parse, independent of sinter."""
    totals = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()])
    rows = 0
    with open(path, newline='') as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(reader)]
        idx = {h: i for i, h in enumerate(header)}
        for row in reader:
            if not row:
                continue
            rows += 1
            t = totals[row[idx['strong_id']].strip()]
            t[0] += int(row[idx['shots']])
            t[1] += int(row[idx['errors']])
            t[2] += int(row[idx['discards']])
            if 'custom_counts' in idx and len(row) > idx['custom_counts'] and row[idx['custom_counts']].strip():
                t[3].update(json.loads(row[idx['custom_counts']]))
    return rows, {k: (s, e, d, +c) for k, (s, e, d, c) in totals.items()}


def _stat_totals(stats: list[sinter.TaskStats]) -> dict:
    return {s.strong_id: (s.shots, s.errors, s.discards, +collections.Counter(s.custom_counts))
            for s in stats}


def is_sinter_csv(path: Path) -> bool:
    with open(path) as f:
        header = [h.strip() for h in f.readline().split(',')]
    return {'shots', 'errors', 'strong_id', 'json_metadata'} <= set(header)


def compact_sinter_csv(path: str | Path, *, apply: bool = True) -> tuple[int, int]:
    """Fold a sinter CSV to one row per strong_id. Returns (bytes_before, bytes_after).

    Files that aren't sinter CSVs, or are already compact, are left alone
    (after == before). The folded content is verified against an independent
    raw parse of the original, and the written copy is re-parsed, before the
    original is touched; a mismatch raises RuntimeError and leaves it intact.

    The original is rewritten in place (same inode) rather than swapped via
    rename, so another process with the file open for append (sinter opens
    save_resume_filepath with mode 'a', i.e. O_APPEND) keeps writing to the
    live file instead of an orphaned one. If the file grows while being
    folded, compaction is abandoned for this call.
    """
    path = Path(path)
    size = path.stat().st_size
    if not is_sinter_csv(path):
        return size, size
    rows, expected = _raw_totals(path)
    if rows <= len(expected):
        return size, size

    stats = sinter.read_stats_from_csv_files(path)
    if _stat_totals(stats) != expected:
        raise RuntimeError(f'{path}: sinter fold disagrees with raw totals; left untouched')

    content = sinter.CSV_HEADER + '\n' + ''.join(s.to_csv_line() + '\n' for s in stats)
    # Written alongside first: it doubles as a backup of the folded data should
    # the in-place rewrite below be interrupted.
    backup = path.with_name(path.name + '.compact.tmp')
    backup.write_text(content)
    rows_new, written = _raw_totals(backup)
    if written != expected or rows_new != len(expected):
        backup.unlink()
        raise RuntimeError(f'{path}: rewritten file fails verification; left untouched')
    if not apply:
        backup.unlink()
        return size, len(content.encode())

    with open(path, 'r+') as f:
        f.seek(0, 2)
        if f.tell() != size:
            backup.unlink()
            return size, size
        f.seek(0)
        f.write(content)
        f.truncate()
    backup.unlink()
    return size, path.stat().st_size
