import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'src'))
import causal_run as run
run.verify_contract()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selection=json.loads((P/'selection.json').read_text(encoding='utf-8'))
days,_=run.make_days()
episodes=pd.concat([pd.read_csv(P/f'test_seed_{s}.csv') for s in run.SEEDS],ignore_index=True)
steps=pd.concat([pd.read_csv(P/f'trajectory_seed_{s}.csv') for s in run.SEEDS],ignore_index=True)
assert len(episodes)==650 and len(steps)==62400
assert set(zip(episodes.controller,episodes.seed,episodes.split_day))=={(c,s,d) for c in run.VARIANTS for s in run.SEEDS for d in range(300,365)}
assert not episodes.duplicated(['controller','seed','split_day']).any()
for seed in run.SEEDS:
    receipt=json.loads((P/f'test_receipt_{seed}.json').read_text(encoding='utf-8'))
    assert receipt['selection_sha256']==sha(P/'selection.json')
for (controller,seed,day),g in steps.groupby(['controller','seed','day']):
    g=g.sort_values('step'); e=episodes[(episodes.controller==controller)&(episodes.seed==seed)&(episodes.split_day==day)].iloc[0]
    assert list(g.step)==list(range(96))
    assert e.budget==selection['selected_budgets'][controller]
    assert (g.p>=g.power_lower_kw-1e-7).all() and (g.p<=g.power_upper_kw+1e-7).all()
    previous=np.r_[125.0,g.soc.to_numpy()[:-1]]
    expected=previous+(np.maximum(0,-g.p)*0.95-np.maximum(0,g.p)/0.95)*0.25
    assert np.max(np.abs(g.soc-expected))<1e-7
    grid=g.grid_import-g.grid_export
    violation=np.maximum(0,grid-260-0.01)+np.maximum(0,-40-grid-0.01)
    assert np.max(np.abs(g.grid_violation-violation))<1e-7
    ev_power=grid-g.base_load_minus_pv_kw-g.lighting_kw+g.p
    assert (ev_power>=g.ev_lower_fraction*60-1e-6).all() and (ev_power<=g.ev_upper_fraction*60+1e-6).all()
    assert abs(g.grid_violation.sum()*0.25-e.grid_excess_energy_kwh)<1e-7
    assert abs(np.sum(g.grid_import*days[int(day)]['price'])*0.25-e.cost)<1e-7
    assert abs(np.sum(g.grid_import*days[int(day)]['carbon'])*0.25-e.carbon)<1e-7
    assert abs(np.max(np.abs(np.diff(np.r_[0,g.p])))-e.max_ramp_kw)<1e-7
    for path in ['hard_qp_solution','phase2_solution','phase1_fallback','heuristic_fallback']:
        assert int((g.qp_execution_path==path).sum())==e[path]
    assert g.terminal_roundoff_repair.sum()==e.terminal_roundoff_repair_steps
    assert g.terminal_ramp_relaxation.sum()==e.terminal_ramp_relaxation_steps
assert (episodes.terminal_soc_error_kwh.abs()<=1e-8).all()
assert (episodes.soc_violation_rate==0).all() and (episodes.ev_completion==1).all()
assert (episodes.lighting_energy_kwh.sub(472.5).abs()<1e-8).all()
assert (episodes.terminal_target_conflict_steps==0).all() and (steps.terminal_target_conflict_step==0).all()
episodes.groupby('controller')[['cost','carbon','task_score','peak','grid_excess_energy_kwh','terminal_soc_error_kwh','terminal_roundoff_repair_steps','terminal_ramp_relaxation_steps']].mean().to_csv(P/'policy_test_means.csv')
result={'status':'PASS','episodes':len(episodes),'steps':len(steps),'selection_sha256':sha(P/'selection.json'),'selected_budgets':selection['selected_budgets'],'maximum_absolute_terminal_error_kwh':float(episodes.terminal_soc_error_kwh.abs().max()),'service_failures':0,'terminal_target_conflict_steps':0,'roundoff_repair_steps':int(steps.terminal_roundoff_repair.sum()),'terminal_ramp_relaxation_steps':int(steps.terminal_ramp_relaxation.sum()),'execution_paths':steps.qp_execution_path.value_counts().to_dict(),'audit_source_sha256':sha(Path(__file__))}
(P/'policy_test_qa.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
