import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
SEEDS=range(20260805,20260810)
VARIANTS=['PPO-QP','PPO-QP-no-penalty']
BUDGETS=[15,30,45,90]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]; validation=[]; models=0
assert not list(P.glob('training_failure_*.json'))
for name in VARIANTS:
    for seed in SEEDS:
        suffix=f'{name}_{seed}'
        summary=json.loads((P/f'train_{suffix}.json').read_text(encoding='utf-8'))
        ep=pd.read_csv(P/f'training_episodes_{suffix}.csv')
        curve=pd.read_csv(P/f'learning_{suffix}.csv')
        event=pd.read_csv(P/f'training_bound_events_{suffix}.csv')
        assert len(ep)==480 and len(curve)==90
        assert (ep.steps==96).all() and list(ep.ending_transition)==list(range(96,46081,96))
        assert (curve.transitions==curve['update']*512).all()
        assert (curve.cumulative_completed_episodes*96+curve.partial_episode_steps==curve.transitions).all()
        assert ep.steps.sum()+summary['partial_episode_metrics']['steps']==summary['transitions']==46080
        assert (ep.service_failure==0).all() and (ep.terminal_soc_error_kwh.abs()<=1e-8).all()
        assert (ep.ev_completion==1).all() and (ep.soc_violation_rate==0).all()
        assert (ep.lighting_energy_kwh.sub(472.5).abs()<1e-8).all()
        assert (event.episode*96+event.step+1==event.transition).all()
        assert ((event.transition-1)//512+1==event['update']).all()
        assert event.terminal_roundoff_repair.sum()==ep.terminal_roundoff_repair_steps.sum()==curve.terminal_roundoff_repair.sum()==summary['training_audit']['terminal_roundoff_repair']
        assert event.terminal_ramp_relaxation.sum()==ep.terminal_ramp_relaxation_steps.sum()==curve.terminal_ramp_relaxation.sum()==summary['training_audit']['terminal_ramp_relaxation']
        assert curve.terminal_target_conflict_step.sum()==ep.terminal_target_conflict_steps.sum()==summary['training_audit']['terminal_target_conflict_step']==0
        for budget in BUDGETS:
            path=P/f'{suffix}_u{budget}.npz'
            assert sha(path)==summary['checkpoints'][str(budget)]
            with np.load(path,allow_pickle=False) as model:
                assert all(np.isfinite(model[k]).all() for k in model)
            models+=1
        v=pd.read_csv(P/f'validation_{suffix}.csv')
        assert len(v)==240 and set(zip(v.budget,v.split_day))=={(b,d) for b in BUDGETS for d in range(240,300)}
        assert (v.terminal_soc_error_kwh.abs()<=1e-8).all() and (v.ev_completion==1).all()
        assert (v.soc_violation_rate==0).all() and (v.lighting_energy_kwh.sub(472.5).abs()<1e-8).all()
        assert (v.terminal_target_conflict_steps==0).all()
        validation.append(v)
        rows.append({'controller':name,'seed':seed,'transitions':summary['transitions'],'episodes':len(ep),'partial_episode_steps':summary['partial_episode_metrics']['steps'],'service_failure_episodes':int(ep.service_failure.sum()),'roundoff_repair_steps':int(event.terminal_roundoff_repair.sum()),'terminal_ramp_relaxation_steps':int(event.terminal_ramp_relaxation.sum()),'terminal_target_conflict_steps':0,'terminal_error_abs_max_kwh':float(ep.terminal_soc_error_kwh.abs().max()),'elapsed_seconds':summary['elapsed_seconds']})
pd.DataFrame(rows).to_csv(P/'training_diagnostics_summary.csv',index=False)
v=pd.concat(validation,ignore_index=True)
v.groupby(['controller','budget'])[['terminal_soc_error_kwh','terminal_roundoff_repair_steps','terminal_target_conflict_steps']].agg(['count','sum','max']).to_csv(P/'validation_service_summary.csv')
result={'status':'PASS','training_runs':len(rows),'transitions':sum(r['transitions'] for r in rows),'completed_episodes':sum(r['episodes'] for r in rows),'partial_episode_steps':sum(r['partial_episode_steps'] for r in rows),'service_failures':sum(r['service_failure_episodes'] for r in rows),'roundoff_repair_steps':sum(r['roundoff_repair_steps'] for r in rows),'terminal_conflict_steps':0,'finite_checkpoints':models,'validation_records':len(v),'validation_max_abs_terminal_error_kwh':float(v.terminal_soc_error_kwh.abs().max()),'validation_roundoff_repair_steps':float(v.terminal_roundoff_repair_steps.sum()),'run_contract_sha256':sha(P/'run_contract.json'),'audit_source_sha256':sha(Path(__file__))}
(P/'training_diagnostics_qa.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
