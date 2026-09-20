from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import sys
import time

for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SNAP = ROOT / 'source_snapshot'
sys.path.insert(0, str(SNAP / 'src'))
import numpy as np
import pandas as pd
import causal_run as cr

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

extension = json.loads((ROOT / 'analysis_extension_protocol.json').read_text())
assert all(sha(SNAP / name) == digest for name, digest in extension['source_hashes'].items())
protocol = json.loads((ROOT / 'protocol.json').read_text())
name = protocol['controller']
d = pd.read_csv(ROOT / 'daily.csv').assign(seed=0)
t = pd.read_csv(ROOT / 'trajectory.csv').assign(seed=0)
days, _ = cr.make_days()
scope = {'np': np, 'pd': pd, 'DT': .25, 'OUT': ROOT, 'REPLICATES': 20000}
for path, function in [('src/aggregate_causal.py', 'bootstrap_indices'), ('revision_src/analyze_revision.py', 'terminal_recovery')]:
    source = ast.parse((SNAP / path).read_text(encoding='utf-8-sig'))
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == function)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SNAP / path), 'exec'), scope)
indices = scope['bootstrap_indices']()
recovery, by_step = scope['terminal_recovery'](t, days)
ppo = pd.concat([pd.read_csv(p) for p in (SNAP/'artifacts').glob('test_seed_*.csv')], ignore_index=True)
ppo['export_excess_energy_kwh'] = ppo.grid_export_violation_kwh
ppo['import_excess_energy_kwh'] = ppo.grid_import_violation_kwh
metrics = ['cost', 'carbon', 'task_score', 'peak', 'grid_excess_energy_kwh', 'export_excess_energy_kwh', 'import_excess_energy_kwh', 'ramp_slack_kw_mean']
rows = []
for controller, learned in ppo.groupby('controller'):
    learned = learned.set_index(['seed', 'split_day'])[metrics].sort_index()
    right = d.set_index('split_day')[metrics].reindex(learned.index.get_level_values('split_day')).to_numpy()
    differences = pd.DataFrame(learned.to_numpy()-right, index=learned.index, columns=metrics)
    by_day = differences.groupby('split_day').mean().reindex(np.arange(300,365)).to_numpy()
    seed_effects = differences.groupby('seed').mean()
    for method, index in indices.items():
        lower, upper = np.quantile(by_day[index].mean(axis=1), [.025,.975], axis=0)
        for j, metric in enumerate(metrics):
            rows.append({'controller':controller,'comparator':name,'metric':metric,'method':method,'mean_difference':float(by_day[:,j].mean()),'ci_lower':float(lower[j]),'ci_upper':float(upper[j]),'seed_difference_min':float(seed_effects[metric].min()),'seed_difference_max':float(seed_effects[metric].max()),'day_clusters':65,'fixed_trained_seeds':5,'bootstrap_replicates':20000,'confidence_level':.95,'inference_scope':'conditional_on_five_fixed_trained_seeds_and_inspected_scenario_holdout'})
intervals = pd.DataFrame(rows)
assert len(intervals) == 48
intervals.to_csv(ROOT/'paired_intervals_all_methods.csv', index=False)
paths = t.qp_execution_path.value_counts()
table4 = {'controller':name,'steps':len(t),'endpoint_infeasible':int(t.endpoint_infeasible.sum()),'endpoint_infeasible_percent':float(100*t.endpoint_infeasible.mean()),'feasible_state_violation':int(t.feasible_state_violation.sum()),'hard_qp_solution':int(paths.get('hard_qp_solution',0)),'phase2_solution':int(paths.get('phase2_solution',0)),'phase1_fallback':int(paths.get('phase1_fallback',0)),'heuristic_fallback':int(paths.get('heuristic_fallback',0))}
pd.DataFrame([table4]).to_csv(ROOT/'table_IV_row.csv',index=False)
peaks = t.loc[t.groupby('day').grid_import.idxmax()]
table10 = {'controller':name,'episodes':65,'peak_last_hour_count':int((peaks.step>=92).sum()),'peak_forced_charge_count':int(peaks.forced_full_charge.sum()),'lower_bound_hours_per_day':float((((t.soc-25).abs()<=1e-5)&((t.soc_before-25).abs()<=1e-5)).sum()*.25/65),'upper_bound_hours_per_day':float((((t.soc-225).abs()<=1e-5)&((t.soc_before-225).abs()<=1e-5)).sum()*.25/65),'ramp_exceed_steps':int((t.ramp_slack_kw>1e-5).sum()),'mean_ramp_slack_kw':float(t.ramp_slack_kw.mean())}
pd.DataFrame([table10]).to_csv(ROOT/'table_S10_row.csv',index=False)
assert len(recovery)==1 and len(by_step)==16
states = json.loads((SNAP/'artifacts/timing_state_protocol.json').read_text())['states']
assert len(states)==25
warmups, timed = [], []
for state in states:
    for repetition in range(6):
        env = cr.CausalEnv(days, cr.CFG, mode='qp', projection_penalty=0)
        env.reset(state['day'], 20260805)
        env.soc, env.prev_p, env.ev_remaining, env.t = state['soc_kwh'], state['previous_battery_power_kw'], state['ev_remaining_kwh'], state['step']
        assert hashlib.sha256(env.observation().tobytes()).hexdigest()==state['observation_sha256']
        before = time.perf_counter_ns()
        obs = env.observation()
        ev = min(1.0, env.ev_remaining/(env.ev_max*env.dt)) if days[env.day]['ev_active'][env.t] else 0.0
        action = cr.core.raw_from_physical(env.P, ev, 1.0, env.P)
        _,_,_,info = env.step(action)
        elapsed = (time.perf_counter_ns()-before)/1e6
        row = {'controller':name,'state_id':state['state_id'],'day':state['day'],'step':state['step'],'repetition':repetition,'elapsed_ms':elapsed,'qp_execution_path':info['qp_execution_path'],'executed_battery_power_kw':info['p'],'soc_after_kwh':info['soc'],'grid_violation_kw':info['grid_violation']}
        (warmups if repetition==0 else timed).append(row)
f, w = pd.DataFrame(timed), pd.DataFrame(warmups)
assert len(f)==125 and len(w)==25 and f.groupby('state_id').size().eq(5).all()
f.to_csv(ROOT/'timing_rows.csv',index=False)
w.to_csv(ROOT/'timing_warmup_rows.csv',index=False)
table5 = {'controller':name,'calls':len(f),'unique_states':25,'mean_ms':float(f.elapsed_ms.mean()),'p95_ms':float(f.elapsed_ms.quantile(.95)),'maximum_ms':float(f.elapsed_ms.max())}
pd.DataFrame([table5]).to_csv(ROOT/'table_V_row.csv',index=False)
save('timing_protocol.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'archived_states_sha256':sha(SNAP/'artifacts/timing_state_protocol.json'),'states':states,'timed_scope':'Same observation/proposal/QP/full model step as the original common-state protocol. Environment construction, reset and restoration, identity checks and serialization excluded.','warmups':25,'timed_calls':125,'execution':'separate serial follow-up; original seven controllers were not retimed','thread_variables':{k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},'qp_paths':f.qp_execution_path.value_counts().to_dict(),'maximum_within_state_power_spread_kw':float(f.groupby('state_id').executed_battery_power_kw.agg(lambda x:x.max()-x.min()).max())})
save('reporting_verification.json',{'status':'PASS','interval_rows':48,'bootstrap_seed':20260908,'bootstrap_function':'Unchanged AST of src/aggregate_causal.py bootstrap_indices','terminal_recovery_function':'Unchanged AST of revision_src/analyze_revision.py terminal_recovery','controller_unchanged':True,'extension_protocol_sha256':sha(ROOT/'analysis_extension_protocol.json'),'runner_sha256':sha(Path(__file__)),'daily_sha256':sha(ROOT/'daily.csv'),'trajectory_sha256':sha(ROOT/'trajectory.csv'),'outputs':{p.name:sha(p) for p in ROOT.glob('*.csv')},'timing':table5,'Table_IV':table4,'Table_S10':table10})
print(json.dumps({'Table_IV':table4,'Table_V':table5,'Table_S10':table10,'Table_S15':recovery.iloc[0].to_dict()},indent=2),flush=True)
