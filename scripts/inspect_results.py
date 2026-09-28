"""Read archived observations; no model, GPU or network access."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
data=json.loads((root/'results/h3-runs.json').read_text(encoding='utf-8'))
print('Worker | TE host | execution seconds | TE milliseconds')
for r in data['runs']:
    print(f"{r['worker_gpu']} | {r['te_gpu']} | {r['history_elapsed_s']:.3f} | {r['remote_te_encode_ms']:.2f}")
print('\nLimits:')
for limit in data['limits']:print('- '+limit)
