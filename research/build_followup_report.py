"""Descriptive summaries and plots; preserves all outcomes and seed variation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
r=json.loads((ROOT/'results/acquisition_v1/metrics.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
colors=['#888888','#3465a4','#138568','#c47c20','#9a458b']
summary=[]
for ax,hi in zip(axs,[340,338]):
 for m,color in zip(['passive','random','stratified','uncertainty','decision'],colors):
  means=[]
  for b in [30,60,120,240]:
   z=[x['regret'] for x in r if x['hi']==hi and x['method']==m and x['budget']==b and x['cost']=='squared']
   means.append(np.mean(z));summary.append(dict(range_high=hi,method=m,budget=b,mean=float(np.mean(z)),median=float(np.median(z)),min=float(min(z)),max=float(max(z)),seeds=10))
  ax.plot([30,60,120,240],means,'-o',label=m,color=color)
 ax.set(title='Original interval [320,340] K' if hi==340 else 'Held-out interval [322,338] K',xlabel='Total paid outcomes, including initialization',ylim=(0,None))
 ax.grid(alpha=.15)
axs[0].set_ylabel('Mean matched-information regret (K²)');axs[1].legend(frameon=False)
fig.suptitle('A simple coverage baseline beats the targeted heuristic at 240 outcomes\nMeans across 10 seeds; full per-seed results retained',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'acquisition_followup.png');plt.close(fig)
(ROOT/'results/acquisition_v1/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
r=json.loads((ROOT/'results/information_audit_v1/metrics.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(12,4.3))
for ax,cost,unit in zip(axs,['squared','absolute','capped_squared'],['K²','K','K²']):
 for mode,color in zip(['snapshot','history','sensor'],['#777777','#3465a4','#138568']):
  means=[np.mean([x['excess_full_state_oracle'] for x in r if x['range_high']==340 and x['q']==q and x['model']=='mlp' and x['mode']==mode and x['cost']==cost]) for q in [1/3,.6,.9]]
  ax.plot([1/3,.6,.9],means,'-o',label=mode,color=color)
 ax.set(title=cost.replace('_',' '),xlabel='Operator preference probability',ylabel=f'Excess over full-state oracle ({unit})',ylim=(0,None));ax.grid(alpha=.15)
axs[-1].legend(frameon=False)
fig.suptitle('Wider-domain MLP: measuring missing state helps more than short history\nThree-seed means; privileged oracle, not v1 matched-information regret',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'information_followup.png');plt.close(fig)
print('Saved follow-up figures and acquisition summary')
