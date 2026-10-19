"""Exploratory uncertainty/error association; no causal or calibrated risk claim."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/robot_uncertainty_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'config.json').exists():raise FileExistsError('Preserve run')
    config=dict(model='mlp_desired',seeds=[11,23,47],horizon_samples=100,selection_fraction=.2,score='Mean across joint dimensions of population variance across three trained model predictions',metrics=['Within-rollout Spearman correlation with ensemble-mean squared error','MSE of top-20%-uncertainty windows / all-window MSE'],scope='Exploratory analysis of existing test predictions; no fitting or threshold selection; not an unseen confirmatory dataset',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw=json.dumps(config,indent=2)+'\n';(OUT/'config.json').write_text(raw);(OUT/'config.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest())
    rows=[]
    for p in sorted((ROOT/'results/real_robot_v1').glob('*_predictions.npz')):
        d=np.load(p);pred=np.stack([d[f'mlp_desired_seed{s}'][:,2,:] for s in config['seeds']])
        uncertainty=pred.var(0).mean(1);error=((pred.mean(0)-d['target'][:,2,:])**2).mean(1)
        for rollout in np.unique(d['rollout']):
            mask=d['rollout']==rollout;u,e=uncertainty[mask],error[mask]
            order=np.argsort(u);k=int(np.ceil(len(u)*.2))
            rows.append(dict(dataset=p.stem.replace('_predictions',''),rollout=int(rollout),spearman=float(spearmanr(u,e).statistic),top20_mse_ratio=float(e[order[-k:]].mean()/e.mean()),windows=len(u)))
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    for dataset in sorted({r['dataset'] for r in rows}):
        z=[r for r in rows if r['dataset']==dataset]
        print(dataset,'median rho',round(float(np.median([r['spearman'] for r in z])),3),'median top20 ratio',round(float(np.median([r['top20_mse_ratio'] for r in z])),3))
if __name__=='__main__':main()
