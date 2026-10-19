"""Matched-information reference rules for the frozen information audit (addendum).

Adds the rows the independent review found missing: how well do simple fixed
rules do against the same full-state oracle used in information_audit_v1?
Reads saved evaluator truth only; trains nothing; does not modify v1 files.
"""
import json
from pathlib import Path

import numpy as np

from research.process.run_study import CA, A

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'results/information_audit_v1'
OUT = ROOT / 'results/information_audit_v1_references.json'
COSTS = ['squared', 'absolute', 'capped_squared']


def main():
    config = json.loads((AUDIT / 'config.json').read_text())
    rows = []
    for lo, hi in config['ranges']:
        truth = np.load(AUDIT / f'evaluator_{hi}.npz')['truth']  # (1000, 3 actions, 3 costs)
        rng = np.random.default_rng(config['evaluator_seed'])
        reset_concentration = rng.choice(CA, 1000)  # identical first draw to the frozen evaluator
        oracle = truth.min(1)
        for j, cost in enumerate(COSTS):
            fixed = truth[:, :, j].mean(0)
            best_fixed = int(fixed.argmin())
            per_stratum = np.empty(1000)
            stratum_actions = []
            for c in CA:
                m = reset_concentration == c
                a = int(truth[m, :, j].mean(0).argmin())
                stratum_actions.append(float(A[a]))
                per_stratum[m] = truth[m, a, j]
            rows.append(dict(range=[lo, hi], cost=cost,
                             best_fixed_command=float(A[best_fixed]),
                             best_fixed_excess=float((truth[:, best_fixed, j] - oracle[:, j]).mean()),
                             per_stratum_commands=stratum_actions,
                             per_stratum_excess=float((per_stratum - oracle[:, j]).mean()),
                             note='Reference rules chosen with evaluator truth (in-sample upper bound for '
                                  'stratum-only information); not learned models.'))
    OUT.write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        print(r['range'], r['cost'], r['best_fixed_command'], round(r['best_fixed_excess'], 2),
              r['per_stratum_commands'], round(r['per_stratum_excess'], 2))


if __name__ == '__main__':
    main()
