"""Recompute the published historical summaries without loading GPU models."""
from pathlib import Path
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / 'results/image-60-runs.json').read_text(encoding='utf-8'))
    ids = set()
    count = 0
    for batch in data['batches']:
        rows = batch['jobs']
        assert len(rows) == batch['total_success'] == 20
        assert len({row['seed'] for row in rows}) == 20
        assert len({row['prompt'] for row in rows}) == 20
        durations = []
        for row in rows:
            assert row['prompt_id'] not in ids
            ids.add(row['prompt_id'])
            assert row['state'] == 'success'
            assert any('REMOTE |' in r and 'FALLBACK' not in r for r in row['remote_receipt'])
            duration = (row['execution_success_ms'] - row['execution_start_ms']) / 1000
            assert math.isclose(duration, row['execution_s'], abs_tol=0.001)
            assert row['image_size'] == [512, 512]
            assert len(row['image_sha256']) == 64
            durations.append(duration)
        warm = durations[1:]
        recorded = batch['after_first_execution']
        assert math.isclose(statistics.mean(warm), recorded['mean_s'], abs_tol=1e-6)
        assert math.isclose(min(warm), recorded['min_s'], abs_tol=1e-6)
        assert math.isclose(max(warm), recorded['max_s'], abs_tol=1e-6)
        assert math.isclose(statistics.pstdev(warm), recorded['std_s'], abs_tol=1e-6)
        assert math.isclose(statistics.mean(durations), batch['all_execution']['mean_s'], abs_tol=1e-6)
        print(f"{batch['worker_gpu']}: 20/20 records; first {durations[0]:.3f}s; warm mean {statistics.mean(warm):.3f}s")
        count += len(rows)
    sources = json.loads((ROOT / 'assets/image-60-contact-sheet.sources.json').read_text(encoding='utf-8'))
    assert len(sources) == count == 60
    mapping = {(b['worker_gpu'], r['index']): r['image_sha256'] for b in data['batches'] for r in b['jobs']}
    assert all(mapping[(s['worker_gpu'], s['index'])] == s['image_sha256'] for s in sources)
    print('PASS: 60 internally consistent archived records and montage hash mappings.')
    print('This check does not rerun generation or authenticate unpublished raw files.')


if __name__ == '__main__':
    main()
