"""Tables and figure for results/exchange_rate_v2 (reads saved outputs only)."""
import json
import os
from pathlib import Path

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results/exchange_rate_v2'


def load():
    return json.loads((RES / 'summary.json').read_text())


def pick(summary, **k):
    return [e for e in summary if all(e.get(a) == b for a, b in k.items())]


def tables(summary):
    lines = ['## Part A: operator logs vs randomized interventions (squared cost, 100 seeds)', '',
             '| Data | Size | Method | Mean regret K² | Median | 90th pct | Seeds > 5 K² |', '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for q in [0.9, 0.6]:
        for e in sorted(pick(summary, part='A', source='logs', q=q), key=lambda e: (e['size'], e['method'])):
            s = e['squared']
            lines.append(f"| logs, q={q} | {e['size']} | {e['method']} | {s['mean']:.2f} | {s['median']:.2f} | {s['p90']:.2f} | {s['frac_over_5']:.0%} |")
    for e in sorted(pick(summary, part='A', source='interventions'), key=lambda e: (e['method'], e['size'])):
        s = e['squared']
        lines.append(f"| interventions | {e['size']} | {e['method']} | {s['mean']:.2f} | {s['median']:.2f} | {s['p90']:.2f} | {s['frac_over_5']:.0%} |")
    lines += ['', '## Part B: acquisition, 100 seeds (squared cost)', '',
              '| Budget | Method | Mean | Median | 90th pct | Seeds > 5 K² | Mean diff vs stratified [95% CI] | W/T/L vs stratified |',
              '| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |']
    for e in sorted(pick(summary, part='B'), key=lambda e: (e['size'], e['method'])):
        s = e['squared']
        v = e.get('vs_stratified_squared')
        diff = f"{v['mean_diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" if v else '—'
        wtl = f"{v['wins']}/{v['ties']}/{v['losses']}" if v else '—'
        lines.append(f"| {e['size']} | {e['method']} | {s['mean']:.2f} | {s['median']:.2f} | {s['p90']:.2f} | {s['frac_over_5']:.0%} | {diff} | {wtl} |")
    return '\n'.join(lines)


def figure(summary):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    styles = {('logs', 'naive'): ('#9a9a9a', 'o', 'Logs, naive regression'),
              ('logs', 'snipw'): ('#c0392b', 's', 'Logs, self-normalized IPW (true propensities)'),
              ('logs', 'dr'): ('#e67e22', '^', 'Logs, doubly robust (true propensities)'),
              ('interventions', 'stratified'): ('#1f5fa8', 'D', 'Randomized interventions, stratified')}
    for (src, m), (c, mk, label) in styles.items():
        rows = pick(summary, part='A', source=src, method=m) if src == 'interventions' else pick(summary, part='A', source=src, method=m, q=0.9)
        rows = sorted(rows, key=lambda e: e['size'])
        x = [e['size'] for e in rows]
        med = [e['squared']['median'] for e in rows]
        mean = [e['squared']['mean'] for e in rows]
        ax.plot(x, mean, marker=mk, color=c, label=label, lw=1.6)
        ax.plot(x, med, marker=mk, color=c, lw=0.8, ls=':', alpha=0.8)
    floor = json.loads((RES / 'naive_bias_floor.json').read_text())['0.9']['regret']
    ax.axhline(floor, color='#9a9a9a', ls='--', lw=1)
    ax.annotate(f'Naive logs with unlimited data still miss: {floor:.1f} K²', (330, floor + 30), fontsize=7.5, color='#555555')
    ax.set_xscale('log')
    ax.set_xlabel('Outcomes collected (log scale)')
    ax.set_ylabel('Decision regret, K² (solid = mean, dotted = median)')
    ax.set_title('240 randomized trials ≈ 512–2,048 operator logs (operator preference 0.9, 100 seeds)', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    out = ROOT / 'results/figures/exchange_rate.png'
    fig.savefig(out)
    return out


if __name__ == '__main__':
    s = load()
    text = tables(s)
    (RES / 'tables.md').write_text(text + '\n')
    print(text)
    print(figure(s))
