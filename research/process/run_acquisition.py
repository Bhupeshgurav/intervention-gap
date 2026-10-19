"""Budget-accounted acquisition feasibility test, with isolated selector inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
from research.process.run_study import CA,A
from research.process.run_information_audit import outcome
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/acquisition_v1'
METHODS=['passive','random','stratified','uncertainty','decision']

def probabilities(count,total,squares,method,rng):
    if method in ['random','passive','stratified']:return np.full(15,1/15)
    mean=total/count
    sse=np.maximum(squares-total**2/count,0)
    pooled=max(float(sse.sum()/np.maximum(count-1,1).sum()),1e-8)
    se2=(sse+2*pooled)/(count+1)/count
    scores=se2/(count+1)
    if method=='decision':
        draws=rng.normal(mean,np.sqrt(se2),size=(128,15)).reshape(128,5,3)
        choices=draws.argmin(2)
        p=np.stack([(choices==a).mean(0) for a in range(3)],axis=1).reshape(-1)
        scores*=p*(1-p)+.01
    if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite selector score')
    return .2/15+.8*scores/scores.sum()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'config.json').exists():raise FileExistsError('Preserve frozen run')
    config=dict(seeds=list(range(101,111)),ranges=[[320,340],[322,338]],methods=METHODS,budgets=[30,60,120,240],batch=15,initial_per_cell=2,costs=['squared','absolute','capped_squared'])
    config['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'research/process/run_information_audit.py',ROOT/'research/process/cstr.py',ROOT/'research/ACQUISITION_PROTOCOL.md']}
    raw=json.dumps(config,indent=2)+'\n';(OUT/'config.json').write_text(raw);(OUT/'config.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest())
    metrics=[];ledger=[];references=[]
    for lo,hi in config['ranges']:
        prior=None
        for nodes in [488,976]:
            state=np.array([[c,t] for c in CA for t in lo+(np.arange(nodes)+.5)*(hi-lo)/nodes])
            truth,max_t=outcome(np.repeat(state,3,axis=0),np.tile(A,len(state)))
            mu=truth.reshape(5,nodes,3,3).mean(1)
            if prior is not None:
                delta=np.abs(mu-prior).max(axis=(0,1))
                if np.any(delta>np.array([1,.1,.5])):raise RuntimeError(f'Unconverged reference {delta}')
            prior=mu
        references.append(dict(range=[lo,hi],means=mu.tolist(),quadrature_delta=delta.tolist(),max_temperature_K=max_t))
        # Every method starts from identical randomized observations for each seed.
        counts=np.full((10,5,15),2,dtype=int)
        sums=np.zeros((10,5,15,3));sq=np.zeros_like(sums)
        selectors=[[np.random.default_rng(seed*100+m) for m in range(5)] for seed in config['seeds']]
        environments=[np.random.default_rng(seed) for seed in config['seeds']]
        initial_states=[];initial_actions=[]
        for si,seed in enumerate(config['seeds']):
            cells=np.repeat(np.arange(15),2)
            temp=environments[si].uniform(lo,hi,30)
            initial_states.extend(np.column_stack([CA[cells//3],temp]));initial_actions.extend(A[cells%3])
        initial_y,_=outcome(np.array(initial_states),np.array(initial_actions))
        for si,seed in enumerate(config['seeds']):
            y=initial_y[si*30:(si+1)*30].reshape(15,2,3)
            sums[si]=y.sum(1);sq[si]=(y*y).sum(1)
            for m in range(5):
                for k in range(30):
                    ledger.append(dict(lo=lo,hi=hi,seed=seed,method=METHODS[m],budget_index=k+1,cell=k//2,selection_probability=None,selection='balanced initialization',temperature=float(initial_states[si*30+k][1]),costs=initial_y[si*30+k].tolist()))
        for budget in range(30,241,15):
            if budget in config['budgets']:
                for si,seed in enumerate(config['seeds']):
                    for mi,method in enumerate(METHODS):
                        pred=(sums[si,mi]/counts[si,mi,:,None]).reshape(5,3,3)
                        for j,cost in enumerate(config['costs']):
                            choice=pred[:,:,j].argmin(1)
                            regret=np.mean(mu[np.arange(5),choice,j]-mu[:,:,j].min(1))
                            metrics.append(dict(lo=lo,hi=hi,seed=seed,method=method,budget=budget,cost=cost,regret=float(regret),actions=A[choice].tolist()))
            if budget==240:break
            todo=[];states=[];actions=[]
            for si,seed in enumerate(config['seeds']):
                # Same fresh temperature vector across arms, never visible to selectors.
                temp=environments[si].uniform(lo,hi,15)
                for mi,method in enumerate(METHODS):
                    rng=selectors[si][mi]
                    p=probabilities(counts[si,mi],sums[si,mi,:,0],sq[si,mi,:,0],method,rng)
                    cells=np.arange(15) if method=='stratified' else rng.choice(15,15,p=p)
                    if method=='passive':
                        preferred=2-np.clip(((temp-lo)/((hi-lo)/3)).astype(int),0,2)
                        u=rng.random(15);offset=np.where(u<.9,0,np.where(u<.95,1,2))
                        cells=(cells//3)*3+(preferred+offset)%3
                    for k,cell in enumerate(cells):
                        states.append([CA[cell//3],temp[k]]);actions.append(A[cell%3])
                        todo.append((si,mi,k,int(cell),float(p[cell]),float(temp[k])))
            y,_=outcome(np.array(states),np.array(actions))
            for value,(si,mi,k,cell,prob,temp) in zip(y,todo):
                counts[si,mi,cell]+=1;sums[si,mi,cell]+=value;sq[si,mi,cell]+=value**2
                ledger.append(dict(lo=lo,hi=hi,seed=config['seeds'][si],method=METHODS[mi],budget_index=budget+k+1,cell=cell,selection_probability=None if METHODS[mi] in ['passive','stratified'] else prob,selection='operator q=.9' if METHODS[mi]=='passive' else METHODS[mi],temperature=temp,costs=value.tolist()))
            if (budget+15)%60==0:print(f'Interval {lo}-{hi}: {budget+15} paid outcomes per arm/seed',flush=True)
        (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        (OUT/'ledger.json').write_text(json.dumps(ledger)+'\n')
        (OUT/'reference.json').write_text(json.dumps(references,indent=2)+'\n')
if __name__=='__main__':main()
