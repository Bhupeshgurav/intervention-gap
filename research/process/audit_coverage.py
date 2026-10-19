"""Post-run descriptive audit of saved data; does not modify experiments."""
import json
from pathlib import Path
import numpy as np
from research.process.run_study import CA,A
ROOT=Path(__file__).resolve().parents[2]
rows=[]
for seed in [11,23,47]:
 d=np.load(ROOT/f'results/process_v1/ledger_{seed}.npz')
 for arm,states,actions in [('base',d['base_states'],d['base_actions']),('passive_128',d['added_states'],d['passive_actions']),('random_128',d['added_states'],d['random_actions']),('random_source',d['base_states'],d['random_source_actions'])]:
  for c in CA:
   for a in A:
    t=states[(states[:,0]==c)&(actions==a),1]
    rows.append(dict(seed=seed,arm=arm,concentration=float(c),command=float(a),n=len(t),temperature_mean_K=None if not len(t) else float(t.mean()),temperature_min_K=None if not len(t) else float(t.min()),temperature_max_K=None if not len(t) else float(t.max())))
report={'scope':'Post-run descriptive coverage audit, not a new preregistered hypothesis test','cells':rows}
(ROOT/'results/process_v1/coverage_audit.json').write_text(json.dumps(report,indent=2)+'\n')
for arm in ['base','passive_128','random_128','random_source']:
 selected=[r for r in rows if r['arm']==arm]
 print(arm,'minimum cell count',min(r['n'] for r in selected),'empty cells',sum(r['n']==0 for r in selected))
 for a in A:
  v=[r for r in selected if r['command']==a and r['n']]
  print(a,round(sum(r['n']*r['temperature_mean_K'] for r in v)/sum(r['n'] for r in v),2))
