"""Build descriptive plots from saved results; never retrains models."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
r=json.loads((ROOT/'results/process_v1/metrics.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
for ax,model in zip(axs,['cell','ridge','mlp']):
 for arm,color in [('passive','#a05a2c'),('random','#007d87')]:
  budgets=[0,16,32,64,128]; values=[]
  for b in budgets:
   z=[x['regret_K2'] for x in r if x['model']==model and x['arm']==('base' if b==0 else arm) and x['budget']==b and x['fit']=='pooled']
   values.append(z)
  v=np.array(values)
  ax.plot(budgets,v.mean(1),'-o',color=color,label=f'{arm}: mean')
  for i in range(3): ax.plot(budgets,v[:,i],color=color,alpha=.22,lw=.8)
 ax.set(title={'cell':'Cell means','ridge':'Quadratic ridge','mlp':'Tiny MLP'}[model],xlabel='Additional labeled episodes')
 ax.grid(alpha=.15);ax.set_ylim(0, 590)
axs[0].set_ylabel('Action regret (K²; lower is better)')
axs[-1].legend(frameon=False)
fig.suptitle('Equal-budget acquisition: pooled fitting, three seeds\nFaint lines show individual seeds; this is not a confidence interval',fontsize=12)
fig.tight_layout()
fig.savefig(OUT/'process_acquisition.png');fig.savefig(OUT/'process_acquisition.svg');plt.close(fig)
r=json.loads((ROOT/'results/real_robot_v1/summary.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(11,4.3))
for ax,family in zip(axs,['Sines','GPs']):
 for model,label,color in [('mlp_state','State','#707070'),('mlp_desired','State + command','#007d87'),('mlp_execution','State + all torque fields','#8a489a')]:
  v=[next(x['rmse_rad'] for x in r if x['family']==family and x['split']==split and x['model']==model and x['horizon_ms']==100) for split in ['test_iid','test_transfer_1','test_transfer_2','test_transfer_3']]
  ax.plot(range(4),v,'-o',label=label,color=color)
 ax.set_xticks(range(4),['IID','Transfer 1','Transfer 2','Transfer 3'])
 ax.set_title(family);ax.set_ylabel('100 ms position RMSE (radians)');ax.grid(alpha=.15)
axs[0].legend(frameon=False,fontsize=8)
fig.suptitle('Real robot: command information does not improve every shifted forecast\nPooled MSE across nine rollouts and three training seeds; no causal interpretation',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'real_robot_transfer.png');fig.savefig(OUT/'real_robot_transfer.svg')
print(OUT)
