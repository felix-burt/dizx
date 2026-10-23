import sinter
import stim

from dizx.sinter_csv import compact_sinter_csv


def _collect(path, max_shots):
    tasks = [
        sinter.Task(
            circuit=stim.Circuit.generated('repetition_code:memory', distance=d, rounds=3,
                                           after_clifford_depolarization=0.05),
            decoder='pymatching',
            json_metadata={'d': d},
        )
        for d in (3, 5)
    ]
    sinter.collect(tasks=tasks, num_workers=2, max_shots=max_shots, max_batch_size=100,
                   decoders=['pymatching'], save_resume_filepath=path)


def _collect_batched(path):
    # Resume in increments so each task gets several appended rows.
    for shots in (300, 600, 1000):
        _collect(path, shots)


def _totals(path):
    return {s.strong_id: (s.shots, s.errors, s.discards)
            for s in sinter.read_stats_from_csv_files(path)}


def _rows(path):
    return len(path.read_text().strip().splitlines()) - 1


def test_compact_preserves_totals_and_resumes(tmp_path):
    path = tmp_path / 'stats.csv'
    _collect_batched(path)
    before = _totals(path)
    assert _rows(path) > 2

    compact_sinter_csv(path)
    assert _rows(path) == 2
    assert _totals(path) == before
    assert not list(tmp_path.glob('*.tmp'))

    # Idempotent, and sinter still resumes from the compacted file.
    size = path.stat().st_size
    assert compact_sinter_csv(path) == (size, size)
    _collect(path, 3000)
    assert all(s.shots >= 3000 for s in sinter.read_stats_from_csv_files(path))


def test_compact_keeps_concurrent_appender_attached(tmp_path):
    path = tmp_path / 'stats.csv'
    _collect_batched(path)
    before = _totals(path)
    appender = open(path, 'a')  # as sinter holds save_resume_filepath
    extra = next(iter(sinter.read_stats_from_csv_files(path))).with_edits(shots=7, errors=1, seconds=0.0)

    compact_sinter_csv(path)
    appender.write(extra.to_csv_line() + '\n')
    appender.close()

    after = _totals(path)
    sid = extra.strong_id
    assert after[sid] == (before[sid][0] + 7, before[sid][1] + 1, before[sid][2])


def test_dry_run_leaves_file_untouched(tmp_path):
    path = tmp_path / 'stats.csv'
    _collect_batched(path)
    text = path.read_text()
    b, a = compact_sinter_csv(path, apply=False)
    assert a < b
    assert path.read_text() == text
