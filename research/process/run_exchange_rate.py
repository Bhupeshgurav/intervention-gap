"""Logs-versus-experiments study (v2). Protocol: research/EXCHANGE_RATE_PROTOCOL.md.

Part A asks how many operator-logged episodes, analysed with off-policy
estimators that are even given the operator's true propensities, it takes to
choose actions as well as a few randomized interventions.
Part B repeats the acquisition comparison with 100 seeds and adds top-two
Thompson sampling as a fair adaptive comparator.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from research.process.run_study import CA, A
from research.process.run_information_audit import outcome
from research.process.run_acquisition import probabilities

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/exchange_rate_v2'
LO, HI = 320.0, 340.0
COSTS = ['squared', 'absolute', 'capped_squared']
CONFIG = dict(
    protocol='research/EXCHANGE_RATE_PROTOCOL.md',
    interval=[LO, HI],
    part_a=dict(seeds=list(range(1001, 1101)), q=[0.9, 0.6], log_sizes=[512, 2048, 8192],
                estimators=['naive', 'ipw', 'snipw', 'dr'],
                intervention_budgets=[15, 30, 60, 120, 240], intervention_rules=['stratified', 'random']),
    part_b=dict(seeds=list(range(2001, 2101)), budgets=[30, 60, 120, 240], batch=15, initial_per_cell=2,
                methods=['passive', 'random', 'stratified', 'uncertainty', 'decision', 'top_two_ts'],
                top_two_beta=0.5),
    bootstrap=dict(resamples=10000, seed=0),
    reference_nodes=[488, 976],
)


# ---------------------------------------------------------------- reference
def reference_means():
    """Expected cost per (concentration, command, cost) averaged over hidden temperature."""
    previous = None
    for nodes in CONFIG['reference_nodes']:
        temps = LO + (np.arange(nodes) + 0.5) * (HI - LO) / nodes
        states = np.array([[c, t] for c in CA for t in temps])
        truth, max_t = outcome(np.repeat(states, 3, axis=0), np.tile(A, len(states)))
        means = truth.reshape(5, nodes, 3, 3).mean(1)
        if previous is not None:
            delta = np.abs(means - previous).max(axis=(0, 1))
            if np.any(delta > np.array([1, .1, .5])):
                raise RuntimeError(f'Reference quadrature unconverged: {delta}')
        previous = means
    return means, delta, max_t


def regret(estimates, mu):
    """Matched-information regret per cost; estimates shape (5, 3, n_costs)."""
    out = []
    for j in range(mu.shape[2]):
        choice = estimates[:, :, j].argmin(1)
        out.append(float(np.mean(mu[np.arange(5), choice, j] - mu[:, :, j].min(1))))
    return out


# ---------------------------------------------------------------- part A
def operator_commands(temps, uniforms, q):
    preferred = 2 - np.clip(((temps - LO) / ((HI - LO) / 3)).astype(int), 0, 2)
    offset = np.where(uniforms < q, 0, np.where(uniforms < (1 + q) / 2, 1, 2))
    action_index = (preferred + offset) % 3
    propensity = np.where(offset == 0, q, (1 - q) / 2)
    return action_index, propensity


def off_policy_estimates(ctx, act, prop, y):
    """Return dict estimator -> (5, 3, n_costs) estimated mean cost; empty cells -> +inf."""
    est = {k: np.full((5, 3, y.shape[1]), np.inf) for k in CONFIG['part_a']['estimators']}
    empty = 0
    for c in range(5):
        in_c = ctx == c
        n_c = in_c.sum()
        for a in range(3):
            cell = in_c & (act == a)
            if cell.sum() == 0:
                empty += 1
                continue
            w = 1.0 / prop[cell]
            naive = y[cell].mean(0)
            est['naive'][c, a] = naive
            est['ipw'][c, a] = (w[:, None] * y[cell]).sum(0) / n_c
            est['snipw'][c, a] = (w[:, None] * y[cell]).sum(0) / w.sum()
            est['dr'][c, a] = naive + (w[:, None] * (y[cell] - naive)).sum(0) / n_c
    return est, empty


def intervention_estimates(rng, budget, rule, cache):
    """Cell means from `budget` randomized interventions; cache supplies fresh outcomes."""
    if rule == 'stratified':
        cells = np.repeat(np.arange(15), budget // 15)
    else:
        cells = rng.integers(0, 15, budget)
    y = cache(cells)
    est = np.full((5, 3, 3), np.inf)
    for cell in range(15):
        hit = cells == cell
        if hit.any():
            est[cell // 3, cell % 3] = y[hit].mean(0)
    return est


def part_a(mu):
    cfg = CONFIG['part_a']
    rows = []
    nmax = max(cfg['log_sizes'])
    for q in cfg['q']:
        for seed in cfg['seeds']:
            rng = np.random.default_rng([seed, int(q * 100)])
            ctx = rng.integers(0, 5, nmax)
            temps = rng.uniform(LO, HI, nmax)
            act, prop = operator_commands(temps, rng.random(nmax), q)
            y, _ = outcome(np.column_stack([CA[ctx], temps]), A[act])
            for n in cfg['log_sizes']:
                est, empty = off_policy_estimates(ctx[:n], act[:n], prop[:n], y[:n])
                for name, e in est.items():
                    rows.append(dict(part='A', source='logs', q=q, seed=seed, size=n, method=name,
                                     empty_cells=empty, regret=regret(e, mu)))
        print(f'Part A logs done for q={q}', flush=True)
    # Intervention comparators do not depend on q; one draw per seed and rule.
    for seed in cfg['seeds']:
        for rule in cfg['intervention_rules']:
            rng = np.random.default_rng([seed, 7, len(rule)])

            def cache(cells, rng=rng):
                temps = rng.uniform(LO, HI, len(cells))
                return outcome(np.column_stack([CA[cells // 3], temps]), A[cells % 3])[0]

            for k in cfg['intervention_budgets']:
                e = intervention_estimates(rng, k, rule, cache)
                rows.append(dict(part='A', source='interventions', q=None, seed=seed, size=k, method=rule,
                                 empty_cells=int(np.isinf(e[:, :, 0]).sum()), regret=regret(e, mu)))
    print('Part A interventions done', flush=True)
    return rows


# ---------------------------------------------------------------- part B
def top_two_cells(count, total, squares, rng, beta):
    """Choose 15 cells by top-two Thompson sampling within round-robin contexts."""
    mean = total / count
    sse = np.maximum(squares - total ** 2 / count, 0)
    pooled = max(float(sse.sum() / np.maximum(count - 1, 1).sum()), 1e-8)
    se = np.sqrt((sse + 2 * pooled) / (count + 1) / count)
    cells = []
    for context in np.repeat(np.arange(5), 3):
        m, s = mean[context * 3:context * 3 + 3], se[context * 3:context * 3 + 3]
        leader = rng.normal(m, s).argmin()
        pick = leader
        if rng.random() >= beta:
            for _ in range(100):
                challenger = rng.normal(m, s).argmin()
                if challenger != leader:
                    pick = challenger
                    break
            else:
                pick = rng.choice([a for a in range(3) if a != leader])
        cells.append(context * 3 + pick)
    return np.array(cells)


def part_b(mu):
    cfg = CONFIG['part_b']
    methods = cfg['methods']
    seeds = cfg['seeds']
    rows = []
    S, M = len(seeds), len(methods)
    counts = np.zeros((S, M, 15), dtype=int)
    sums = np.zeros((S, M, 15, 3))
    sq = np.zeros_like(sums)
    environments = [np.random.default_rng(s) for s in seeds]
    selectors = [[np.random.default_rng(s * 100 + m) for m in range(M)] for s in seeds]
    # Shared balanced initialization: two outcomes per cell, charged to every arm.
    init_cells = np.repeat(np.arange(15), 2)
    for si in range(S):
        temps = environments[si].uniform(LO, HI, 30)
        y, _ = outcome(np.column_stack([CA[init_cells // 3], temps]), A[init_cells % 3])
        for mi in range(M):
            np.add.at(counts[si, mi], init_cells, 1)
            np.add.at(sums[si, mi], init_cells, y)
            np.add.at(sq[si, mi], init_cells, y * y)
    budget = 30
    while True:
        if budget in cfg['budgets']:
            for si, seed in enumerate(seeds):
                for mi, method in enumerate(methods):
                    est = (sums[si, mi] / counts[si, mi][:, None]).reshape(5, 3, 3)
                    rows.append(dict(part='B', seed=seed, method=method, size=budget, regret=regret(est, mu)))
        if budget == max(cfg['budgets']):
            break
        states, actions, owners = [], [], []
        for si in range(S):
            temps = environments[si].uniform(LO, HI, 15)  # same hidden draws across arms
            for mi, method in enumerate(methods):
                rng = selectors[si][mi]
                if method == 'top_two_ts':
                    cells = top_two_cells(counts[si, mi], sums[si, mi, :, 0], sq[si, mi, :, 0], rng, cfg['top_two_beta'])
                elif method == 'stratified':
                    cells = np.arange(15)
                else:
                    p = probabilities(counts[si, mi], sums[si, mi, :, 0], sq[si, mi, :, 0], method, rng)
                    cells = rng.choice(15, 15, p=p)
                    if method == 'passive':
                        act, _ = operator_commands(temps, rng.random(15), 0.9)
                        cells = (cells // 3) * 3 + act
                for k, cell in enumerate(cells):
                    states.append([CA[cell // 3], temps[k]])
                    actions.append(A[cell % 3])
                    owners.append((si, mi, int(cell)))
        y, _ = outcome(np.array(states), np.array(actions))
        for value, (si, mi, cell) in zip(y, owners):
            counts[si, mi, cell] += 1
            sums[si, mi, cell] += value
            sq[si, mi, cell] += value ** 2
        budget += 15
        if budget % 60 == 0:
            print(f'Part B: {budget} outcomes per arm/seed', flush=True)
    return rows


# ---------------------------------------------------------------- summaries
def describe(values):
    v = np.asarray(values)
    return dict(mean=float(v.mean()), median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                frac_over_5=float((v > 5).mean()), frac_zero=float((v <= 1e-9).mean()), n=int(len(v)))


def paired(a, b, rng):
    """Bootstrap CI for mean(a - b) over seeds, plus win/tie/loss of a vs b (lower is better)."""
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), (CONFIG['bootstrap']['resamples'], len(d)))
    boots = d[idx].mean(1)
    return dict(mean_diff=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                wins=int((d < -1e-9).sum()), ties=int((np.abs(d) <= 1e-9).sum()), losses=int((d > 1e-9).sum()))


def summarize(rows):
    rng = np.random.default_rng(CONFIG['bootstrap']['seed'])
    summary = []
    groups = {}
    for r in rows:
        key = (r['part'], r.get('source'), r.get('q'), r['method'], r['size'])
        groups.setdefault(key, []).append(r)
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: r['seed'])
        entry = dict(part=key[0], source=key[1], q=key[2], method=key[3], size=key[4])
        for j, cost in enumerate(COSTS):
            entry[cost] = describe([r['regret'][j] for r in rs])
        if 'empty_cells' in rs[0]:
            entry['mean_empty_cells'] = float(np.mean([r['empty_cells'] for r in rs]))
        # Paired comparison against stratified at the same size (Part B) or K=240 stratified (Part A).
        if key[0] == 'B' and key[3] != 'stratified':
            ref = sorted(groups[('B', None, None, 'stratified', key[4])], key=lambda r: r['seed'])
            entry['vs_stratified_squared'] = paired([r['regret'][0] for r in rs], [r['regret'][0] for r in ref], rng)
        summary.append(entry)
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'config.json').exists():
        raise FileExistsError('Preserve frozen run; choose a new output version')
    sources = [Path(__file__), ROOT / 'research/process/run_information_audit.py', ROOT / 'research/process/run_acquisition.py',
               ROOT / 'research/process/cstr.py', ROOT / 'research/EXCHANGE_RATE_PROTOCOL.md']
    config = dict(CONFIG, source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    raw = json.dumps(config, indent=2) + '\n'
    (OUT / 'config.json').write_text(raw)
    (OUT / 'config.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest() + '\n')
    mu, delta, max_t = reference_means()
    (OUT / 'reference.json').write_text(json.dumps(dict(means=mu.tolist(), quadrature_delta=delta.tolist(),
                                                        max_temperature_K=max_t,
                                                        best_actions_squared=A[mu[:, :, 0].argmin(1)].tolist()), indent=2) + '\n')
    rows = part_a(mu) + part_b(mu)
    (OUT / 'per_seed.json').write_text(json.dumps(rows) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summarize(rows), indent=2) + '\n')
    print('Done', flush=True)


if __name__ == '__main__':
    main()
