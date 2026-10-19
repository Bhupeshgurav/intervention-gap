"""Feature control requested by the independent review (addendum to information_audit_v1).

Question: did the generic MLP fail to use concentration history because the
derivative signal was hard to extract from raw readings? We refit the frozen
wider-interval, q=0.9 setting with the same data, seeds and evaluator, adding a
mode whose inputs are [current concentration, two successive differences].
The 'history' mode is refit too, to check that the replay matches the saved run.
"""
import json
from pathlib import Path

import numpy as np

from research.models import fit_predictor
from research.process.run_study import CA, A
from research.process.run_information_audit import prepare, outcome

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'results/information_audit_v1'
OUT = ROOT / 'results/information_audit_v1_difference_control.json'
LO, HI, Q = 320, 340, 0.9


def features(hist, action, mode):
    if mode == 'history':
        observed = hist
    elif mode == 'history_diff':
        observed = np.column_stack([hist[:, -1], hist[:, -1] - hist[:, -2], hist[:, -2] - hist[:, -3]])
    return np.column_stack([observed, action])


def main():
    if OUT.exists():
        raise FileExistsError('Preserve frozen addendum')
    config = json.loads((AUDIT / 'config.json').read_text())
    saved = json.loads((AUDIT / 'metrics.json').read_text())
    ev = np.load(AUDIT / f'evaluator_{HI}.npz')
    eh, truth = np.repeat(ev['history'], 3, axis=0), ev['truth']
    ea = np.tile(A, 1000)
    rows = []
    for seed in config['seeds']:
        rng = np.random.default_rng(seed)
        reset = np.column_stack([rng.choice(CA, 512), rng.uniform(LO, HI, 512)])
        state, hist = prepare(reset)
        uniforms = rng.random(512)
        preferred = 2 - np.clip(((state[:, 1] - LO) / ((HI - LO) / 3)).astype(int), 0, 2)
        offset = np.where(uniforms < Q, 0, np.where(uniforms < (1 + Q) / 2, 1, 2))
        act = A[(preferred + offset) % 3]
        y, _ = outcome(state, act)
        for mode in ['history', 'history_diff']:
            m = fit_predictor('mlp', features(hist, act, mode), y, seed=seed, epochs=config['epochs'])
            pred = m.predict(features(eh, ea, mode)).reshape(1000, 3, 3)
            choice = pred[:, :, 0].argmin(1)
            excess = float((truth[np.arange(1000), choice, 0] - truth[:, :, 0].min(1)).mean())
            original = [r['excess_full_state_oracle'] for r in saved if r['range_high'] == HI and r['seed'] == seed
                        and abs(r['q'] - Q) < 1e-9 and r['model'] == 'mlp' and r['mode'] == 'history' and r['cost'] == 'squared']
            rows.append(dict(seed=seed, mode=mode, cost='squared', excess_full_state_oracle=excess,
                             saved_history_excess=original[0] if mode == 'history' else None))
            print(rows[-1], flush=True)
    OUT.write_text(json.dumps(dict(note='Addendum control after independent review; exploratory; same data/evaluator as '
                                        'information_audit_v1 wider interval, q=0.9, MLP, squared cost. Full-state oracle.',
                                   rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
