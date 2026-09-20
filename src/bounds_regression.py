import copy
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import causal_run as run
import run_experiment as core

ROOT=run.ROOT
OLD=ROOT.parents[1]/'submission_reanalysis_20260908'
REVIEW=ROOT.parents[1]/'peer_review_20260908'

def load_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

oldcore=load_source('historical_core',OLD/'src/run_experiment.py')
sys.modules['run_experiment']=oldcore
oldrun=load_source('historical_causal',OLD/'src/causal_run.py')
sys.modules['run_experiment']=core

def replay():
    cases=pd.read_csv(REVIEW/'validation_terminal_failures.csv')
    days,_=run.make_days()
    rows=[]; traces=[]
    for case in cases.to_dict('records'):
        name=case['controller']; seed=int(case['seed']); budget=int(case['budget']); day=int(case['split_day'])
        checkpoint=OLD/'artifacts'/f'{name}_{seed}_u{budget}.npz'
        results={}
        for version,module in [('old',oldrun),('fixed',run)]:
            agent=module.core.PPOAgent.load(checkpoint,106,module.CFG['ppo']['hidden'],seed,module.CFG['ppo']['learning_rate'])
            env=module.CausalEnv(days,module.CFG,mode='qp',projection_penalty=module.VARIANTS[name]); env.reset(day,seed)
            while True:
                if version=='fixed':
                    diagnostic={}; before=copy.deepcopy(env.metrics)
                    bounds=env._bounds(diagnostic)
                    assert env.metrics==before
                    assert env._bounds()==bounds and env.metrics==before
                else:
                    before=copy.deepcopy(env.metrics); bounds=env._bounds(); env.metrics=before
                    remaining=env.n-env.t-1; target=env.cfg['bess']['soc_initial']*env.E
                    low=max(env.soc_min,target-remaining*env.P*env.eta_c*env.dt)
                    high=min(env.soc_max,target+remaining*env.P/env.eta_d*env.dt)
                    power=lambda e:(env.soc-e)*env.eta_d/env.dt if env.soc>=e else -(e-env.soc)/(env.eta_c*env.dt)
                    physical_lo=max(-env.P,-(env.soc_max-env.soc)/(env.eta_c*env.dt))
                    physical_hi=min(env.P,(env.soc-env.soc_min)*env.eta_d/env.dt)
                    diagnostic={'terminal_lower_kw':power(high),'terminal_upper_kw':power(low),'terminal_physical_gap_kw':max(physical_lo,power(high))-min(physical_hi,power(low))}
                step=env.t; soc=env.soc
                action=agent.act(env.observation(),deterministic=True)[0]
                _,_,done,info=env.step(action)
                traces.append({'version':version,'controller':name,'seed':seed,'budget':budget,'day':day,'step':step,'soc_before_kwh':soc,'power_lower_kw':bounds[0],'power_upper_kw':bounds[1],'raw_power_kw':action[0]*env.P,'executed_power_kw':info['p'],'soc_after_kwh':env.soc,'qp_path':info['qp_execution_path'],**diagnostic})
                if done:
                    results[version]=info
                    break
        assert abs(results['old']['terminal_soc_error_kwh']-case['terminal_soc_error_kwh'])<1e-7
        assert abs(results['fixed']['terminal_soc_error_kwh'])<1e-8
        assert results['fixed']['terminal_target_conflict_steps']==0
        assert results['fixed']['ev_completion']==1 and results['fixed']['soc_violation_rate']==0
        fixed_steps=[r for r in traces if r['version']=='fixed' and r['controller']==name and r['budget']==budget and r['day']==day]
        assert sum(r['terminal_roundoff_repair'] for r in fixed_steps)==results['fixed']['terminal_roundoff_repair_steps']
        rows.append({'controller':name,'seed':seed,'budget':budget,'day':day,'old_terminal_error_kwh':results['old']['terminal_soc_error_kwh'],'fixed_terminal_error_kwh':results['fixed']['terminal_soc_error_kwh'],'fixed_roundoff_repair_steps':results['fixed']['terminal_roundoff_repair_steps'],'fixed_terminal_ramp_relaxation_steps':results['fixed']['terminal_ramp_relaxation_steps'],'checkpoint_sha256':core.sha256(checkpoint)})
    pd.DataFrame(rows).to_csv(run.OUT/'regression_historical_cases.csv',index=False)
    pd.DataFrame(traces).to_csv(run.OUT/'regression_historical_steps.csv',index=False)
    return rows

def synthetic():
    days,_=run.make_days(); rows=[]
    for sign in [-1,1]:
        cfg=copy.deepcopy(run.CFG); cfg['bess']['soc_initial']=0.6 if sign<0 else 0.4
        for remaining in range(1,5):
            for offset in ['inside','exact','roundoff','infeasible']:
                for reverse_ramp in [False,True]:
                    env=run.CausalEnv(days,cfg,mode='qp'); env.reset(240); env.t=96-remaining
                    target=cfg['bess']['soc_initial']*env.E
                    edge=target+remaining*env.P/env.eta_d*env.dt if sign>0 else target-remaining*env.P*env.eta_c*env.dt
                    env.soc=edge-sign*1e-7 if offset=='inside' else np.nextafter(edge,np.inf*sign) if offset=='roundoff' else edge+sign*1e-3 if offset=='infeasible' else edge
                    env.prev_p=sign*env.P*(-1 if reverse_ramp else 1)
                    before=copy.deepcopy(env.metrics); initial_soc=env.soc; diagnosis={}
                    try:
                        bounds=env._bounds(diagnosis)
                        assert offset!='infeasible'
                        assert env.metrics==before
                        assert env._bounds()==bounds and env.metrics==before
                        assert -env.P<=bounds[0]<=bounds[1]<=env.P
                        assert abs(bounds[0]-sign*env.P)<1e-5 or abs(bounds[1]-sign*env.P)<1e-5
                        for _ in range(remaining):
                            _,_,done,info=env.step(np.array([-sign,1,1]))
                        assert done and abs(env.soc-target)<core.TERMINAL_ENERGY_TOL_KWH
                        rows.append({'sign':sign,'remaining_steps':remaining,'offset':offset,'reverse_ramp':reverse_ramp,'initial_soc_kwh':initial_soc,'result':'feasible','terminal_error_kwh':env.soc-target,**diagnosis})
                    except core.TerminalTargetInfeasible as exc:
                        assert offset=='infeasible'
                        assert env.metrics==before and env.soc==initial_soc
                        assert exc.diagnostics['terminal_physical_gap_kw']>exc.diagnostics['tolerance_kw']
                        try:
                            env.step(np.array([sign,1,1]))
                            raise AssertionError('infeasible step did not fail')
                        except core.TerminalTargetInfeasible:
                            assert env.metrics==before and env.soc==initial_soc
                        rows.append({'sign':sign,'remaining_steps':remaining,'offset':offset,'reverse_ramp':reverse_ramp,'initial_soc_kwh':initial_soc,'result':'explicit_infeasible',**exc.diagnostics})
    pd.DataFrame(rows).to_csv(run.OUT/'regression_synthetic_bounds.csv',index=False)
    return rows

if __name__=='__main__':
    run.OUT.mkdir(exist_ok=True)
    synthetic_rows=synthetic(); historical_rows=replay()
    run.save_json(run.OUT/'bounds_regression.json',{'status':'PASS','energy_tolerance_kwh':core.TERMINAL_ENERGY_TOL_KWH,'power_tolerance_kw':core.TERMINAL_ENERGY_TOL_KWH*0.95/0.25,'synthetic_cases':len(synthetic_rows),'synthetic_infeasible_explicit_failures':sum(r['result']=='explicit_infeasible' for r in synthetic_rows),'historical_cases':len(historical_rows),'maximum_fixed_terminal_error_kwh':max(abs(r['fixed_terminal_error_kwh']) for r in historical_rows),'historical_checkpoint_sources':'Historical checkpoints replayed in the fixed environment; these are regression evidence, not newly trained candidates.','code_hashes':{name:core.sha256(ROOT/'src'/name) for name in ['run_experiment.py','causal_run.py','bounds_regression.py']}})
    print(json.dumps({'status':'PASS','synthetic_cases':len(synthetic_rows),'historical_cases':len(historical_rows)}),flush=True)
