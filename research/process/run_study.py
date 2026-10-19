"""Frozen, equal-label-budget intervention experiment; CPU reproducible."""
import hashlib
import json
from pathlib import Path
import numpy as np
from research.models import fit_predictor
from research.process.cstr import simulate_commands

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/process_v1'
CA = np.array([.72, .78, .84, .90, .96])
A = np.array([290., 296., 302.])


def labels(states, actions, dt=.0025):
    steps = round(1 / dt)
    trajectory = simulate_commands(states, np.repeat(actions[:, None], steps, axis=1), dt=dt, max_step=.000125)
    cost = np.trapezoid((trajectory[:, :, 1] - 330) ** 2, dx=dt, axis=1)
    future = trajectory[:, np.arange(1, 11) * (steps // 10), :].reshape(len(states), -1)
    return np.column_stack([future, cost])


def sample(rng, n):
    return np.column_stack([rng.choice(CA, n), rng.uniform(320, 340, n)])


def commands(states, uniforms, q=.9, reverse=False):
    preferred = 2 - np.minimum(((states[:, 1] - 320) / (20 / 3)).astype(int), 2)
    if reverse:
        preferred = 2 - preferred
    offsets = np.where(uniforms < q, 0, np.where(uniforms < (1 + q) / 2, 1, 2))
    return A[(preferred + offsets) % 3]


def features(states, actions, full=False):
    return np.column_stack([states if full else states[:, :1], actions])


class CellMean:
    def fit(self, x, y):
        self.global_mean = y.mean(0)
        self.values = {tuple(row): y[np.all(x == row, axis=1)].mean(0) for row in np.unique(x, axis=0)}
        return self
    def predict(self, x):
        return np.array([self.values.get(tuple(row), self.global_mean) for row in x])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = dict(seeds=[11, 23, 47], base_episodes=512, budgets=[0, 16, 32, 64, 128], models=['cell', 'ridge', 'mlp'], epochs=160, width=64,
                  observation='snapshot concentration; temperature hidden', initial_temperature=[320, 340], concentrations=CA.tolist(), actions=A.tolist(),
                  target='10 future states and direct integrated squared temperature error', horizon_source_units=1, dt=.0025, max_step=.000125,
                  arms=['passive', 'random'], fits=['pooled', 'new_only'], quadrature_nodes=[61, 122, 244, 488],
                  convergence_rule='double uniform midpoint nodes until all action means change < 1 K^2; fail if not converged by 488',
                  controls=['random-source snapshot', 'full-state source'], inference='three seeds descriptive only; simulator-specific, no factory transfer')
    config['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'research/models.py', ROOT/'research/process/cstr.py']}
    raw = json.dumps(config, indent=2) + '\n'
    if (OUT/'config.json').exists() and (OUT/'config.json').read_text() != raw:
        raise RuntimeError('Refusing to overwrite a differently configured experiment')
    (OUT/'config.json').write_text(raw)
    (OUT/'config.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest()+'\n')
    previous = None
    convergence = []
    for n in config['quadrature_nodes']:
        states = np.array([[c, t] for c in CA for t in 320 + (np.arange(n)+.5)*20/n])
        test_states = np.repeat(states, 3, axis=0)
        test_actions = np.tile(A, len(states))
        truth = labels(test_states, test_actions)
        costs = truth[:, -1].reshape(5, n, 3)
        means = costs.mean(1)
        delta = None if previous is None else float(np.abs(means-previous).max())
        convergence.append(dict(nodes=n, maximum_mean_change_K2=delta))
        if delta is not None and delta < 1:
            break
        previous = means
    else:
        (OUT/'quadrature.json').write_text(json.dumps(convergence, indent=2))
        raise RuntimeError('Quadrature failed convergence; no model evaluation performed')
    # Independently check temporal quadrature; same solver step, finer output integration.
    finer = labels(test_states, test_actions, dt=.00125)
    time_delta = float(np.abs(finer[:, -1].reshape(5, n, 3).mean(1)-means).max())
    if time_delta > 1:
        raise RuntimeError(f'Cost time quadrature error {time_delta} > 1 K^2')
    (OUT/'quadrature.json').write_text(json.dumps(dict(spatial=convergence, temporal_mean_delta_K2=time_delta), indent=2))
    np.savez_compressed(OUT/'reference.npz', states=test_states, actions=test_actions, targets=truth, mean_costs=means)
    baseline_regret = float(np.mean(means.mean(1)-means.min(1)))
    records = []
    predictions = {}
    def evaluate(model, seed, kind, arm, budget, fit, full=False):
        prediction = model.predict(features(test_states, test_actions, full))
        pcost = prediction[:, -1].reshape(5, n, 3)
        if full:
            choice = pcost.argmin(2)
            selected = np.take_along_axis(costs, choice[..., None], axis=2)[..., 0]
            regret = float((selected-costs.min(2)).mean())
            chosen = None
        else:
            chosen = pcost.mean(1).argmin(1)
            regret = float(np.mean(means[np.arange(5), chosen]-means.min(1)))
        error = (prediction[:, -1]-truth[:, -1]).reshape(5, n, 3)**2
        preferred = 2-np.minimum(((states[:,1]-320)/(20/3)).astype(int),2).reshape(5,n)
        def rmse(reverse):
            weights = np.full((5,n,3), .05)
            np.put_along_axis(weights, (2-preferred if reverse else preferred)[...,None], .9, axis=2)
            return float(np.sqrt(np.mean(np.sum(error*weights, axis=2))))
        key = f'{seed}_{kind}_{arm}_{budget}_{fit}'
        predictions[key] = prediction
        records.append(dict(seed=seed, model=kind, arm=arm, budget=budget, fit=fit, full_state=full, regret_K2=regret,
                            normalized_regret=None if full else regret/baseline_regret, selected_actions=None if chosen is None else A[chosen].tolist(),
                            source_cost_rmse_K2=rmse(False), reversed_cost_rmse_K2=rmse(True),
                            trajectory_temperature_rmse_K=float(np.sqrt(np.mean((prediction[:,1:20:2]-truth[:,1:20:2])**2)))))
    for seed in config['seeds']:
        rng = np.random.default_rng(seed)
        base, added = sample(rng,512), sample(rng,128)
        ba = commands(base, rng.random(512))
        pa = commands(added, rng.random(128))
        ra = rng.choice(A,128)
        by, py, ry = labels(base,ba), labels(added,pa), labels(added,ra)
        random_ba = rng.choice(A,512)
        random_by = labels(base,random_ba)
        np.savez_compressed(OUT/f'ledger_{seed}.npz', base_states=base, base_actions=ba, base_targets=by, added_states=added, passive_actions=pa, passive_targets=py, random_actions=ra, random_targets=ry, random_source_actions=random_ba, random_source_targets=random_by)
        for kind in config['models']:
            def train(x,y):
                return CellMean().fit(x,y) if kind=='cell' else fit_predictor(kind,x,y,seed=seed,epochs=160)
            evaluate(train(features(base,ba),by),seed,kind,'base',0,'pooled')
            evaluate(train(features(base,random_ba),random_by),seed,kind,'random_source',0,'pooled')
            if kind!='cell':
                evaluate(train(features(base,ba,True),by),seed,kind,'full_state',0,'pooled',True)
            for budget in config['budgets'][1:]:
                for arm, aa, ay in [('passive',pa,py),('random',ra,ry)]:
                    ax = features(added[:budget],aa[:budget])
                    for fit in config['fits']:
                        x,y = (np.concatenate([features(base,ba),ax]),np.concatenate([by,ay[:budget]])) if fit=='pooled' else (ax,ay[:budget])
                        evaluate(train(x,y),seed,kind,arm,budget,fit)
        (OUT/'metrics.json').write_text(json.dumps(records, indent=2)+'\n')
        print(f'Completed seed {seed}: {len(records)} evaluations',flush=True)
    np.savez_compressed(OUT/'predictions.npz',**predictions)
    (OUT/'reference_summary.json').write_text(json.dumps(dict(mean_costs=means.tolist(), optimal_actions=A[means.argmin(1)].tolist(), uniform_action_regret_K2=baseline_regret, max_sampled_temperature_K=float(truth[:,1:20:2].max()), note='High-temperature excursions are intentionally retained; not safe operating advice'),indent=2))


if __name__ == '__main__':
    main()
