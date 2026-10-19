"""Known-physics observer controls, including noise and parameter mismatch."""
import hashlib,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from research.process.cstr import DEFAULT_PARAMETERS,simulate_commands
from research.process.run_study import A
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/state_estimation_v1'

def finite_difference(history,params=DEFAULT_PARAMETERS):
    c=history[:,-1]
    dc=(3*c-4*history[:,1]+history[:,0])/.02
    k=(params.q/params.V*(params.Caf-c)-dc)/c
    valid=(c>0)&(k>0)&(k<params.k0)
    temp=np.full(len(c),np.nan)
    temp[valid]=params.EA_over_R/np.log(params.k0/k[valid])
    return np.column_stack([c,temp]),valid,np.zeros(len(c),bool)

def trajectory_fit(history,params=DEFAULT_PARAMETERS):
    def trajectory(t):
        return simulate_commands(np.column_stack([history[:,0],t]),np.full((len(t),2),296.),dt=.01,max_step=.000125,params=params)
    def loss(t):
        return np.sum((trajectory(t)[:,1:,0]-history[:,1:])**2,axis=1)
    lo=np.full(len(history),300.);hi=np.full(len(history),360.)
    ratio=(np.sqrt(5)-1)/2
    left=hi-ratio*(hi-lo);right=lo+ratio*(hi-lo)
    for _ in range(36):
        choose_left=loss(left)<loss(right)
        hi=np.where(choose_left,right,hi);lo=np.where(choose_left,lo,left)
        left=hi-ratio*(hi-lo);right=lo+ratio*(hi-lo)
    t=(lo+hi)/2;estimate=trajectory(t)[:,-1,:]
    return estimate,np.isfinite(estimate).all(1),(t<300.01)|(t>359.99)

def costs(states,params):
    x=np.repeat(states,3,axis=0);a=np.tile(A,len(states))
    tr=simulate_commands(x,np.repeat(a[:,None],400,axis=1),dt=.0025,max_step=.000125,params=params)
    err=np.abs(tr[:,:,1]-330)
    return np.stack([np.trapezoid(v,dx=.0025,axis=1) for v in [err**2,err,np.minimum(err**2,400)]],axis=1).reshape(len(states),3,3)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'config.json').exists():raise FileExistsError('Preserve frozen run')
    config=dict(ranges=[330,340],noise=[0,.0001,.001],noise_seeds=[101,102,103],kinetic_factors=[.9,1,1.1],methods=['finite_difference','trajectory_fit'],fallback_command=296,scope='Known-physics control on previously examined evaluator; not new held-out confirmation')
    config['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'research/process/cstr.py',ROOT/'research/STATE_ESTIMATION_PROTOCOL.md']}
    raw=json.dumps(config,indent=2)+'\n';(OUT/'config.json').write_text(raw);(OUT/'config.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest())
    records=[];saved={}
    conditions=[(sigma,seed,1.) for sigma in config['noise'] for seed in ([0] if sigma==0 else config['noise_seeds'])]+[(0,0,f) for f in [.9,1.1]]
    for hi in config['ranges']:
        data=np.load(ROOT/f'results/information_audit_v1/evaluator_{hi}.npz')
        true=data['states'];truth=data['truth']
        for sigma,seed,factor in conditions:
            observed=data['history']+np.random.default_rng(seed).normal(0,sigma,data['history'].shape)
            params=replace(DEFAULT_PARAMETERS,k0=DEFAULT_PARAMETERS.k0*factor)
            for name,fn in [('finite_difference',finite_difference),('trajectory_fit',trajectory_fit)]:
                estimate,valid,boundary=fn(observed,params)
                # Numerical invalidity is explicitly recorded, not replaced by hidden state.
                pred=np.full_like(truth,np.nan)
                try:
                    pred[valid]=costs(estimate[valid],params)
                except FloatingPointError:
                    # Preserve individual usable estimates if a batch contains an unstable case.
                    for i in np.flatnonzero(valid):
                        try:pred[i]=costs(estimate[i:i+1],params)[0]
                        except FloatingPointError:valid[i]=False
                chosen=np.full((len(true),3),1,dtype=int)
                chosen[valid]=pred[valid].argmin(1)
                key=f'{hi}_{sigma}_{seed}_{factor}_{name}'
                saved[key]=np.column_stack([estimate,valid,boundary,chosen])
                for j,cost in enumerate(['squared','absolute','capped_squared']):
                    excess=truth[np.arange(len(true)),chosen[:,j],j]-truth[:,:,j].min(1)
                    records.append(dict(range_high=hi,noise_sd=sigma,noise_seed=seed,k0_factor=factor,method=name,cost=cost,valid_fraction=float(valid.mean()),boundary_fraction=float(boundary.mean()),temperature_rmse_K=None if not valid.any() else float(np.sqrt(np.mean((estimate[valid,1]-true[valid,1])**2))),conditional_excess=None if not valid.any() else float(excess[valid].mean()),fallback_population_excess=float(excess.mean())))
            (OUT/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')
            print(f'Finished interval {hi}, noise {sigma}, seed {seed}, k0 factor {factor}',flush=True)
    np.savez_compressed(OUT/'estimates.npz',**saved)
if __name__=='__main__':main()
