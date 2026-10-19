"""Prospective information/thermal audit; v1 remains unchanged."""
import hashlib,json
from pathlib import Path
import numpy as np
from research.models import fit_predictor
from research.process.cstr import simulate_commands
from research.process.run_study import CA,A
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/information_audit_v1'

def prepare(reset):
    past=simulate_commands(reset,np.full((len(reset),2),296.),dt=.01,max_step=.000125)
    return past[:,-1,:],past[:,:,0]

def outcome(states,actions):
    tr=simulate_commands(states,np.repeat(actions[:,None],400,axis=1),dt=.0025,max_step=.000125)
    err=np.abs(tr[:,:,1]-330)
    costs=np.stack([np.trapezoid(v,dx=.0025,axis=1) for v in [err**2,err,np.minimum(err**2,400)]],axis=1)
    return costs,float(tr[:,:,1].max())

def features(state,history,action,mode):
    observed={'snapshot':state[:,:1],'history':history,'sensor':state}[mode]
    return np.column_stack([observed,action])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'config.json').exists():raise FileExistsError('Preserve frozen run')
    config=dict(seeds=[11,23,47],ranges=[[320,340],[320,330]],q=[1/3,.6,.9],modes=['snapshot','history','sensor'],models=['ridge','mlp'],epochs=160,train_n=512,test_n=1000,evaluator_seed=20261016,costs=['squared','absolute','capped_squared'],oracle='Full-state privileged per-episode oracle for ALL modes; not matched information',warmup=.02)
    config['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'research/models.py',ROOT/'research/process/cstr.py',ROOT/'research/FOLLOWUP_PROTOCOL.md']}
    raw=json.dumps(config,indent=2)+'\n';(OUT/'config.json').write_text(raw);(OUT/'config.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest())
    rows=[];preds={}
    for lo,hi in config['ranges']:
        rng=np.random.default_rng(config['evaluator_seed'])
        reset=np.column_stack([rng.choice(CA,1000),rng.uniform(lo,hi,1000)])
        state,hist=prepare(reset)
        es,eh,ea=np.repeat(state,3,axis=0),np.repeat(hist,3,axis=0),np.tile(A,len(state))
        truth,max_t=outcome(es,ea);truth=truth.reshape(1000,3,3)
        np.savez_compressed(OUT/f'evaluator_{hi}.npz',states=state,history=hist,truth=truth)
        for seed in config['seeds']:
            rng=np.random.default_rng(seed)
            reset=np.column_stack([rng.choice(CA,512),rng.uniform(lo,hi,512)])
            state,hist=prepare(reset);uniforms=rng.random(512)
            preferred=2-np.clip(((state[:,1]-lo)/((hi-lo)/3)).astype(int),0,2)
            for q in config['q']:
                offset=np.where(uniforms<q,0,np.where(uniforms<(1+q)/2,1,2))
                act=A[(preferred+offset)%3];y,train_max=outcome(state,act)
                for kind in config['models']:
                    for mode in config['modes']:
                        m=fit_predictor(kind,features(state,hist,act,mode),y,seed=seed,epochs=160)
                        pred=m.predict(features(es,eh,ea,mode)).reshape(1000,3,3)
                        key=f'{hi}_{seed}_{q}_{kind}_{mode}';preds[key]=pred
                        for j,cost in enumerate(config['costs']):
                            chosen=pred[:,:,j].argmin(1)
                            excess=truth[np.arange(1000),chosen,j]-truth[:,:,j].min(1)
                            rows.append(dict(range_high=hi,seed=seed,q=q,model=kind,mode=mode,cost=cost,excess_full_state_oracle=float(excess.mean()),all_action_rmse=float(np.sqrt(np.mean((pred[:,:,j]-truth[:,:,j])**2))),test_max_temperature_K=max_t,train_max_temperature_K=train_max))
            (OUT/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(f'Finished range [{lo},{hi}], seed {seed}: {len(rows)} metric rows',flush=True)
    np.savez_compressed(OUT/'predictions.npz',**preds)

if __name__=='__main__':main()
