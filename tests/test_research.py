from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp
from research.process.cstr import cstr_rhs, simulate_commands
from research.process.run_study import commands, features, CellMean
from research.real_data.run_baselines import make_windows, HORIZONS


def test_windows_use_current_inputs_and_stay_in_rollout():
    t = np.arange(400)[None, :, None]
    q = np.broadcast_to(t + np.array([0,10000])[:,None,None], (2,400,3)).copy().astype(float)
    arrays = {k:q.copy() for k in ['measured_angles','measured_velocities','desired_torques','constrained_torques','measured_torques']}
    data = SimpleNamespace(arrays=arrays, shape=q.shape, sample_interval_seconds=.001)
    w = make_windows(data)
    for i,(r,a) in enumerate(zip(w['rollout'],w['anchor'])):
        np.testing.assert_array_equal(w['target'][i],q[r,a+HORIZONS])
        np.testing.assert_array_equal(w['features']['execution'][i],np.tile(q[r,a],5))
    arrays['desired_torques'][:,101,:] = -999999
    np.testing.assert_array_equal(make_windows(data)['features']['desired'],w['features']['desired'])


def test_stiff_trajectory_matches_independent_solver():
    initial = np.array([[.96,340.]])
    reference = solve_ivp(lambda t,x:cstr_rhs(x,302.),(0,1),initial[0],method='Radau',rtol=1e-10,atol=1e-12,t_eval=np.linspace(0,1,21))
    actual = simulate_commands(initial,np.full((1,20),302.),dt=.05,max_step=.000125)[0]
    np.testing.assert_allclose(actual[:,0],reference.y[0],atol=1e-6,rtol=0)
    np.testing.assert_allclose(actual[:,1],reference.y[1],atol=.001,rtol=0)


def test_private_information_removed_and_policy_reversed():
    states = np.array([[.84,321],[.84,330],[.84,339]])
    np.testing.assert_array_equal(commands(states,np.zeros(3)),[302,296,290])
    np.testing.assert_array_equal(commands(states,np.zeros(3),reverse=True),[290,296,302])
    np.testing.assert_array_equal(features(states,np.full(3,296)),np.tile([.84,296],(3,1)))
    assert features(states,np.full(3,296),True).shape == (3,3)


def test_cell_mean_has_explicit_missing_cell_fallback():
    model = CellMean().fit(np.array([[1,0],[1,0],[2,1]]),np.array([[2.],[4.],[9.]]))
    np.testing.assert_allclose(model.predict(np.array([[1,0],[7,1]])),[[3.],[5.]])


def test_acquisition_selector_uses_only_observed_sufficient_statistics():
    import inspect
    from research.process.run_acquisition import probabilities
    assert list(inspect.signature(probabilities).parameters)==['count','total','squares','method','rng']
    n=np.full(15,2);total=np.arange(15,dtype=float)*2;squares=total**2/2+1
    for method in ['random','uncertainty','decision']:
        p=probabilities(n,total,squares,method,np.random.default_rng(4))
        assert p.shape==(15,)
        np.testing.assert_allclose(p.sum(),1)
        assert (p>=.2/15).all()


def test_history_ends_at_decision_before_any_candidate_action():
    from research.process.run_information_audit import prepare,features
    state,hist=prepare(np.array([[.84,325.],[.84,335.]]))
    np.testing.assert_allclose(hist[:,-1],state[:,0])
    a=features(state,hist,np.array([290,302]),'history')
    b=features(state,hist,np.array([302,290]),'history')
    np.testing.assert_array_equal(a[:,:-1],b[:,:-1])
    assert a.shape==(2,4)


def test_saved_acquisition_ledgers_account_for_every_outcome():
    import json
    from pathlib import Path
    from collections import defaultdict
    p=Path('results/acquisition_v1/ledger.json')
    if not p.exists():
        import pytest
        pytest.skip('Run acquisition experiment to check its ledger')
    groups=defaultdict(list)
    for row in json.loads(p.read_text()):
        groups[(row['lo'],row['hi'],row['seed'],row['method'])].append(row)
    assert len(groups)==100
    for key,rows in groups.items():
        assert [r['budget_index'] for r in rows]==list(range(1,241))
        assert np.isfinite([r['costs'] for r in rows]).all()
        assert all(key[0]<=r['temperature']<=key[1] for r in rows)
        initial=rows[:30]
        assert np.bincount([r['cell'] for r in initial],minlength=15).tolist()==[2]*15
        reference=groups[(*key[:3],'random')][:30]
        assert [(r['cell'],r['temperature'],r['costs']) for r in initial]==[(r['cell'],r['temperature'],r['costs']) for r in reference]


def test_physics_inversion_recovers_temperature_from_exact_local_derivative():
    from research.process.run_state_estimation import finite_difference
    state=np.array([[.78,322.],[.90,336.]])
    dc=cstr_rhs(state,296.)[:,0]
    # Linear local history makes the backward-difference derivative exact.
    history=state[:,0,None]+dc[:,None]*np.array([-.02,-.01,0])
    estimate,valid,_=finite_difference(history)
    assert valid.all()
    np.testing.assert_allclose(estimate,state,atol=1e-10)


def test_trajectory_observer_recovers_known_warmup_without_temperature_input():
    from research.process.run_information_audit import prepare
    from research.process.run_state_estimation import trajectory_fit
    state,history=prepare(np.array([[.78,322.],[.90,336.]]))
    estimate,valid,boundary=trajectory_fit(history)
    assert valid.all() and not boundary.any()
    np.testing.assert_allclose(estimate,state,atol=1e-4)


def test_nonphysical_inverse_is_an_abstention():
    from research.process.run_state_estimation import finite_difference
    estimate,valid,_=finite_difference(np.array([[.7,.8,.9]]))
    assert not valid[0]
    assert np.isnan(estimate[0,1])


def test_record_checker_preserves_failures_and_distinguishes_execution():
    import json
    from pathlib import Path
    from research.records.qualify import qualify
    records=[json.loads(line) for line in Path('examples/synthetic_action_records.jsonl').read_text().splitlines()]
    report=qualify(records)
    assert report['total_records']==7 and report['eligible_records']==2
    assert len(report['records'])==7
    assert report['issue_counts']['duplicate_episode_id']==2
    assert report['issue_counts']['observation_available_after_decision']==1
    assert report['issue_counts']['execution_not_verified_as_executed']==1
    assert report['requested_applied_mismatches']==1
    assert any(r['applied_action']=='fictional-option-B' for r in report['observed_cell_counts'])


def test_record_checker_rejects_naive_timestamps_and_reversed_outcomes():
    import json
    from pathlib import Path
    from research.records.qualify import qualify
    record=json.loads(Path('examples/synthetic_action_records.jsonl').read_text().splitlines()[0])
    record['decision_at']='2026-10-16T10:00:00'
    record['outcome']['observed_at']='2026-10-16T09:00:00Z'
    row=qualify([record])['records'][0]
    assert 'invalid_decision_time' in row['issues']
    assert 'outcome_precedes_application' in row['issues']
    assert not row['eligible_for_action_outcome_evaluation']
