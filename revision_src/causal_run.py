from pathlib import Path
from collections import Counter
import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
import numpy as np
import pandas as pd
import run_experiment as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts'
SEEDS = [20260805+i for i in range(5)]
BUDGETS = [15,30,45,90]
VARIANTS = {'PPO-QP':0.1,'PPO-QP-no-penalty':0.0}
CFG = json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
METRICS = ['cost','carbon','task_score','peak','grid_excess_energy_kwh','grid_violation_step_rate','grid_max_exceedance_kw','soc_violation_rate','terminal_soc_error_kwh','ev_completion','lighting_energy_kwh','mean_action_correction','bess_throughput_kwh','ramp_slack_kw_mean','max_ramp_kw','endpoint_infeasible_steps','feasible_state_violation_steps','extra_excess_energy_kwh','hard_qp_solution','phase2_solution','phase1_fallback','heuristic_fallback','mpc_fallbacks','mpc_time_limit_incumbents','runtime_mean_ms','runtime_p95_ms']

def save_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def raw_data():
    return {'load':pd.read_csv(core.DATA/'annual_load_pattern_CAMX_baseline.csv').load_data.to_numpy(float),
            'price':pd.read_csv(core.DATA/'cambium_grid_data_California_cambium_grid_value.csv').value.to_numpy(float)/1000,
            'carbon':pd.read_csv(core.DATA/'cambium_grid_data_California_cambium_co2_rate_lrmer.csv').value.to_numpy(float)/1000,
            'pv':np.asarray(json.loads((core.DATA/'pvwatts_pasadena_1kw.json').read_text(encoding='utf-8'))['outputs']['ac'],float)/1000}

def make_days(raw=None):
    raw = raw_data() if raw is None else raw
    assert all(len(x)==8760 and np.isfinite(x).all() for x in raw.values())
    fit = {'load_mean':float(raw['load'][:5760].mean()),'pv_mean':float(raw['pv'][:5760].mean()),'price_mean':float(raw['price'][:5760].mean()),'price_sd':float(raw['price'][:5760].std()),'fit_hours':5760}
    values = {'load':raw['load']/fit['load_mean']*145,'pv':raw['pv']*(0.55*145/fit['pv_mean']),'price':np.clip(0.08+0.025*(raw['price']-fit['price_mean'])/fit['price_sd'],0.015,0.28),'carbon':np.maximum(raw['carbon'],0.03)}
    values = {k:np.repeat(v,4) for k,v in values.items()}
    rng = np.random.default_rng(7331)
    days = []
    for day in range(365):
        local = np.random.default_rng(7331+day)
        d = {'day':day,'ev_required':float(rng.uniform(30,52))}
        for key,sd in [('load',0.035),('pv',0.07),('price',0.05),('carbon',0.04)]:
            x=values[key][day*96:(day+1)*96]*(1+sd*local.normal(size=96))
            d[key]=np.clip(x,0.01,0.35) if key=='price' else np.maximum(x,{'load':25,'pv':0,'carbon':0.02}[key])
        hours=np.arange(96)/4
        d['ev_active']=((hours<7)|(hours>=18)).astype(float)
        d['light_profile']=np.where(d['ev_active']>0,1.0,0.25)
        days.append(d)
    return days,fit

class CausalEnv(core.MicrogridEnv):
    def reset(self,day,seed=0,scenario=None):
        self.path_counts=Counter(); self.audit=Counter(); self.latencies=[]
        self.reward_sum=0.0; self.correction_sum=0.0
        return super().reset(day,seed,{'terminal_soc_target':True,'fixed_lighting':True,**(scenario or {})})

    def _forecast(self,key):
        current=self.day*96+self.t
        result=[]
        for h in range(self.h):
            source=current if h==0 or current+h<96 else current+h-96
            d,t=divmod(source,96)
            result.append(float(self.days[d][key][t]))
        return np.asarray(result)

    def step(self,raw):
        before=self.prev_p
        terminal={}
        lo,hi,emin,emax,_=self._bounds(terminal)
        light=self.cfg['loads']['lighting_base_kw']*self.days[self.day]['light_profile'][self.t]
        base=self._actual('load')-self._actual('pv')
        minimum=max(0,base+emin*self.ev_max+light-hi-self.grid_max-0.01,self.grid_min-(base+emax*self.ev_max+light-lo)-0.01)
        obs,reward,done,info=super().step(raw)
        violation=info['grid_violation']
        self.step_audit={'endpoint_minimum_violation_kw':minimum,'endpoint_infeasible':int(minimum>1e-9),'feasible_state_violation':int(minimum<=1e-9 and violation>1e-9),'extra_excess_energy_kwh_step':max(0,violation-minimum)*self.dt,'power_lower_kw':lo,'power_upper_kw':hi,'ev_lower_fraction':emin,'ev_upper_fraction':emax,'base_load_minus_pv_kw':base,'lighting_kw':light}
        self.step_audit.update(terminal)
        self.step_audit['terminal_target_conflict_step']=0
        info.update({k:terminal[k] for k in ['terminal_roundoff_repair','terminal_ramp_relaxation','terminal_physical_gap_kw','terminal_ramp_gap_kw']})
        info['terminal_target_conflict_step']=0
        self.path_counts[info['qp_execution_path']]+=1
        self.audit['endpoint_infeasible_steps']+=int(minimum>1e-9)
        self.audit['feasible_state_violation_steps']+=int(minimum<=1e-9 and violation>1e-9)
        self.audit['grid_steps']+=int(violation>1e-12)
        self.audit['grid_energy']+=violation*self.dt
        self.audit['extra_excess_energy_kwh']+=max(0,violation-minimum)*self.dt
        self.audit['max_actual_ramp']=max(self.audit['max_actual_ramp'],abs(info['p']-before))
        self.reward_sum+=reward; self.correction_sum+=info['correction']
        if done:
            assert abs(self.audit['grid_energy']-info['grid_excess_energy_kwh'])<1e-7
            assert self.audit['grid_steps']==self.metrics['grid_violation_steps']
            info.update(self.audit)
            info.update({k:self.path_counts[k] for k in ['hard_qp_solution','phase2_solution','phase1_fallback','heuristic_fallback']})
            info['task_score']=-self.reward_sum-self.projection_penalty*self.correction_sum
        return obs,reward,done,info

def model_path(name,seed,budget):
    return OUT/f'{name}_{seed}_u{budget}.npz'

def load_model(name,seed,budget):
    return core.PPOAgent.load(model_path(name,seed,budget),106,CFG['ppo']['hidden'],seed,CFG['ppo']['learning_rate'])

def evaluate(days,controller,agent,day,seed=0,keep_steps=False):
    env=CausalEnv(days,CFG,mode='qp',projection_penalty=VARIANTS.get(controller,0))
    env.reset(day,seed)
    mpc_fallbacks=0; limited=0; timings=[]; steps=[]
    while True:
        start=time.perf_counter()
        if controller=='Rule+QP': action=core.rule_action(env)
        elif controller=='MPC-H24+QP':
            from matched_mpc import matched_mpc_action
            action=matched_mpc_action(env)
            m=env.mpc_last_info
            mpc_fallbacks+=int(m.get('fallback',False))
            limited+=int(m.get('status')==1 and not m.get('fallback',False))
        else: action=agent.act(env.observation(),deterministic=True)[0]
        _,_,done,info=env.step(action)
        timings.append((time.perf_counter()-start)*1000)
        if keep_steps:
            steps.append({'controller':controller,'seed':seed,'day':day,'step':len(timings)-1,**{k:info[k] for k in ['p','soc','grid_import','grid_export','grid_violation','qp_execution_path']},**env.step_audit,'mpc_status':env.mpc_last_info['status'] if controller=='MPC-H24+QP' else -1,'mpc_fallback':int(env.mpc_last_info['fallback']) if controller=='MPC-H24+QP' else 0,'runtime_ms':timings[-1]})
            if controller=='MPC-H24+QP':
                steps[-1].update({'mpc_'+k:env.mpc_last_info[k] for k in ['mip_gap','seconds','constraint_violation_max','message']})
        if done: break
    result={k:float(v) for k,v in info.items() if isinstance(v,(int,float,np.number))}
    result.update({'controller':controller,'seed':seed,'split_day':day,'split':'validation' if day<300 else 'test','mpc_fallbacks':mpc_fallbacks,'mpc_time_limit_incumbents':limited,'runtime_mean_ms':float(np.mean(timings)),'runtime_p95_ms':float(np.quantile(timings,0.95))})
    assert sum(result[k] for k in ['hard_qp_solution','phase2_solution','phase1_fallback','heuristic_fallback'])==96
    return result,steps

def train(seed):
    verify_contract()
    days,_=make_days(); training=days[:240]
    for name,penalty in VARIANTS.items():
        agent=core.PPOAgent(106,CFG['ppo']['hidden'],seed,CFG['ppo']['learning_rate'])
        initial=agent.actor.p['w1'].copy()
        env=CausalEnv(training,CFG,mode='qp',projection_penalty=penalty)
        rng=np.random.default_rng(seed+17)
        obs=env.reset(int(rng.integers(240)),seed+100)
        start=time.perf_counter(); curve=[]; episodes=[]; events=[]; cumulative=Counter()
        episode_seed=seed+100
        episode_fields=['steps','cost','carbon','task_score','terminal_soc_kwh','terminal_soc_error_kwh','terminal_target_conflict_steps','terminal_roundoff_repair_steps','terminal_ramp_relaxation_steps','soc_violation_rate','soc_violation_steps','ev_completion','final_ev_remaining','lighting_energy_kwh','grid_excess_energy_kwh']
        for update in range(CFG['ppo']['updates']):
            n=CFG['ppo']['rollout_steps']; ob=np.zeros((n,106)); latent=np.zeros((n,3)); lp=np.zeros(n); val=np.zeros(n); rew=np.zeros(n); done=np.zeros(n,dtype=bool)
            update_audit=Counter()
            for i in range(n):
                ob[i]=obs
                action,lp[i],val[i],_,latent[i]=agent.act(obs,return_latent=True)
                try:
                    obs,rew[i],done[i],info=env.step(action)
                except core.TerminalTargetInfeasible as exc:
                    pd.DataFrame(episodes).to_csv(OUT/f'training_episodes_{name}_{seed}.csv',index=False)
                    pd.DataFrame(events).to_csv(OUT/f'training_bound_events_{name}_{seed}.csv',index=False)
                    save_json(OUT/f'training_failure_{name}_{seed}.json',{'controller':name,'seed':seed,'update':update+1,'completed_transitions':update*n+i,'episode':len(episodes),'terminal_target_conflict_steps':1,'diagnostics':exc.diagnostics,'cumulative':dict(cumulative),'partial_episode_metrics':env.metrics})
                    raise
                cumulative['steps']+=1
                for event in ['terminal_roundoff_repair','terminal_ramp_relaxation','terminal_target_conflict_step']:
                    update_audit[event]+=int(info[event]); cumulative[event]+=int(info[event])
                if info['terminal_roundoff_repair'] or info['terminal_ramp_relaxation']:
                    events.append({'controller':name,'seed':seed,'episode':len(episodes),'update':update+1,'transition':update*n+i+1,**env.step_audit})
                if done[i]:
                    service_failure=int(info['soc_violation_rate']>0 or abs(info['terminal_soc_error_kwh'])>1e-5 or info['ev_completion']<1 or abs(info['lighting_energy_kwh']-472.5)>1e-5)
                    episodes.append({'controller':name,'seed':seed,'episode':len(episodes),'episode_seed':episode_seed,'training_day':env.day,'ending_update':update+1,'ending_transition':update*n+i+1,'service_failure':service_failure,**{k:info[k] for k in episode_fields}})
                    cumulative['completed_episodes']+=1; cumulative['service_failure_episodes']+=service_failure
                    episode_seed=seed+1000+update*100+i
                    obs=env.reset(int(rng.integers(240)),episode_seed)
            next_value=0.0 if done[-1] else agent.critic.forward(obs)[0][0,0]
            advantages=np.zeros(n); gae=0.0
            for i in range(n-1,-1,-1):
                nonterminal=1.0-float(done[i])
                future_value=next_value if i==n-1 else val[i+1]
                delta=rew[i]+CFG['ppo']['gamma']*future_value*nonterminal-val[i]
                gae=delta+CFG['ppo']['gamma']*CFG['ppo']['gae_lambda']*nonterminal*gae
                advantages[i]=gae
            agent.update(ob,latent,lp,advantages+val,advantages,CFG['ppo']['clip'],CFG['ppo']['epochs'],CFG['ppo']['minibatch'])
            curve.append({'seed':seed,'controller':name,'update':update+1,'transitions':(update+1)*n,'mean_training_reward':float(rew.mean()),**dict(update_audit),'cumulative_completed_episodes':cumulative['completed_episodes'],'cumulative_service_failure_episodes':cumulative['service_failure_episodes'],'partial_episode_steps':env.metrics['steps'],'partial_episode_soc_kwh':env.soc,'partial_episode_ev_remaining_kwh':env.ev_remaining,'partial_episode_terminal_conflict_steps':env.metrics['terminal_target_conflict_steps'],'partial_episode_roundoff_repair_steps':env.metrics.get('terminal_roundoff_repair_steps',0)})
            pd.DataFrame(curve).to_csv(OUT/f'learning_{name}_{seed}.csv',index=False)
            pd.DataFrame(episodes).to_csv(OUT/f'training_episodes_{name}_{seed}.csv',index=False)
            pd.DataFrame(events).to_csv(OUT/f'training_bound_events_{name}_{seed}.csv',index=False)
            if update+1 in BUDGETS:
                agent.save(model_path(name,seed,update+1))
                print(f'trained {name} seed={seed} updates={update+1} seconds={time.perf_counter()-start:.1f}',flush=True)
        assert not np.array_equal(initial,agent.actor.p['w1'])
        assert all(np.isfinite(x).all() for x in [*agent.actor.p.values(),*agent.critic.p.values()])
        pd.DataFrame(curve).to_csv(OUT/f'learning_{name}_{seed}.csv',index=False)
        assert sum(r['steps'] for r in episodes)+env.metrics['steps']==cumulative['steps']==CFG['ppo']['updates']*n
        save_json(OUT/f'train_{name}_{seed}.json',{'seed':seed,'controller':name,'transitions':cumulative['steps'],'elapsed_seconds':time.perf_counter()-start,'checkpoints':{str(b):core.sha256(model_path(name,seed,b)) for b in BUDGETS},'finite':all(np.isfinite(x).all() for x in [*agent.actor.p.values(),*agent.critic.p.values()]),'training_audit':dict(cumulative),'partial_episode_metrics':env.metrics,'completed_episode_steps':sum(r['steps'] for r in episodes),'run_contract_sha256':core.sha256(OUT/'run_contract.json')})
        rows=[]
        for budget in BUDGETS:
            model=load_model(name,seed,budget)
            for day in range(240,300):
                result,_=evaluate(days,name,model,day,seed)
                result['budget']=budget; rows.append(result)
        pd.DataFrame(rows).to_csv(OUT/f'validation_{name}_{seed}.csv',index=False)
        print(f'validation complete {name} seed={seed}',flush=True)

def verify_contract():
    contract=json.loads((OUT/'run_contract.json').read_text(encoding='utf-8'))
    assert contract['protocol_sha256']==core.sha256(ROOT/'PLAN.md')
    assert contract['config_sha256']==core.sha256(ROOT/'configs/experiment.json')
    assert contract['config']==CFG
    assert all(core.sha256(ROOT/'src'/name)==digest for name,digest in contract['code_hashes'].items())

def prepare():
    days,fit=make_days(); raw=raw_data(); changed={k:v.copy() for k,v in raw.items()}
    for x in changed.values(): x[5760:]*=7
    changed_days,changed_fit=make_days(changed)
    assert fit==changed_fit
    assert all(np.array_equal(days[d][k],changed_days[d][k]) for d in range(240) for k in raw)
    checks=[]
    for day,t in [(0,0),(239,95),(240,0),(299,95),(300,0),(364,95)]:
        env=CausalEnv(days,CFG,mode='qp'); env.reset(day); env.t=t
        changed=copy.deepcopy(days)
        for d in range(day,365):
            for k in raw: changed[d][k][t+1 if d==day else 0:]*=11
        probe=CausalEnv(changed,CFG,mode='qp'); probe.reset(day); probe.t=t
        assert np.array_equal(env.observation(),probe.observation())
        checks.append({'day':day,'step':t,'future_perturbation_invariant':True})
    agent=core.PPOAgent(106,48,SEEDS[0],0.0003); obs=env.observation()
    _,oldlp,_,_,z=agent.act(obs,return_latent=True)
    mean=agent.actor.forward(obs)[0][0]
    newlp=np.sum(-0.5*((z-mean)/agent.std)**2-np.log(agent.std*np.sqrt(2*np.pi)))
    assert abs(np.exp(newlp-oldlp)-1)<1e-12
    for day in [0,239,240,299,300,364]:
        result,_=evaluate(days,'Rule+QP',None,day)
        assert abs(result['terminal_soc_error_kwh'])<1e-5
        assert abs(result['lighting_energy_kwh']-472.5)<1e-5
        assert result['ev_completion']==1
    save_json(OUT/'preprocessing.json',{'fit':fit,'chronological_partition':[240,60,65],'source_hashes':{p.name:core.sha256(p) for p in core.DATA.iterdir() if p.suffix in ['.csv','.json']},'split_hashes':{label:hashlib.sha256(b''.join(days[d][k].tobytes() for d in range(start,end) for k in raw)).hexdigest() for label,start,end in [('train',0,240),('validation',240,300),('test',300,365)]}})
    save_json(OUT/'causality_checks.json',{'status':'passed','train_preprocessing_holdout_perturbation_invariant':True,'training_values_holdout_perturbation_invariant':True,'forecast_checks':checks,'latent_ratio':float(np.exp(newlp-oldlp)),'rule_service_boundary_checks':6})
    save_json(OUT/'run_contract.json',{'protocol_sha256':core.sha256(ROOT/'PLAN.md'),'code_hashes':{name:core.sha256(ROOT/'src'/name) for name in ['run_experiment.py','causal_run.py']},'config_sha256':core.sha256(ROOT/'configs/experiment.json'),'config':CFG,'seeds':SEEDS,'budgets':BUDGETS,'variants':VARIANTS,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'test_history':'Previously inspected holdout; causal chronological reanalysis, not untouched external validation'})
    print('preprocessing, causal isolation, PPO ratio and shared-service smoke checks passed',flush=True)

def baselines():
    days,_=make_days()
    for controller in ['Rule+QP','MPC-H24+QP']:
        rows=[]; traces=[]; started=time.perf_counter()
        for day in range(300,365):
            result,steps=evaluate(days,controller,None,day,keep_steps=True)
            rows.append(result); traces.extend(steps)
            if (day-299)%5==0:
                pd.DataFrame(rows).to_csv(OUT/f'baseline_{controller}.csv',index=False)
                print(f'baseline {controller} days={day-299}/65 seconds={time.perf_counter()-started:.1f}',flush=True)
        pd.DataFrame(rows).to_csv(OUT/f'baseline_{controller}.csv',index=False)
        pd.DataFrame(traces).to_csv(OUT/f'trajectory_{controller}.csv',index=False)

def select():
    verify_contract()
    assert not any(OUT.glob('test_seed*.csv'))
    selected={}; all_rows=[]
    for name in VARIANTS:
        frame=pd.concat([pd.read_csv(OUT/f'validation_{name}_{seed}.csv') for seed in SEEDS],ignore_index=True)
        assert len(frame)==1200
        assert set(zip(frame.seed,frame.budget,frame.split_day))=={(s,b,d) for s in SEEDS for b in BUDGETS for d in range(240,300)}
        frame['service_failure']=((frame.soc_violation_rate>0)|(frame.terminal_soc_error_kwh.abs()>1e-5)|(frame.ev_completion<1)|(abs(frame.lighting_energy_kwh-472.5)>1e-5)).astype(int)
        grouped=frame.groupby('budget')[METRICS+['service_failure']].mean().reset_index()
        winner=grouped.sort_values(['service_failure','grid_excess_energy_kwh','cost','mean_action_correction','budget']).iloc[0]
        selected[name]=int(winner.budget); all_rows.append(grouped.assign(controller=name))
    pd.concat(all_rows).to_csv(OUT/'validation_budget_means.csv',index=False)
    save_json(OUT/'selection.json',{'selected_budgets':selected,'selection_inputs':{p.name:core.sha256(p) for p in OUT.glob('validation_PPO*.csv')},'selected_model_hashes':{model_path(name,s,b).name:core.sha256(model_path(name,s,b)) for name,b in selected.items() for s in SEEDS},'test_evaluation_not_started':not any(OUT.glob('test_seed*.csv')),'run_contract_sha256':core.sha256(OUT/'run_contract.json')})
    print(json.dumps(selected),flush=True)

def test(seed):
    verify_contract()
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    assert selection['test_evaluation_not_started']
    assert selection['run_contract_sha256']==core.sha256(OUT/'run_contract.json')
    assert all(core.sha256(OUT/name)==digest for name,digest in selection['selection_inputs'].items())
    assert all(core.sha256(OUT/name)==digest for name,digest in selection['selected_model_hashes'].items())
    save_json(OUT/f'test_receipt_{seed}.json',{'seed':seed,'selection_sha256':core.sha256(OUT/'selection.json'),'selected_budgets':selection['selected_budgets'],'started_unix_seconds':time.time(),'run_contract_sha256':core.sha256(OUT/'run_contract.json')})
    days,_=make_days(); rows=[]; traces=[]
    for name,budget in selection['selected_budgets'].items():
        agent=load_model(name,seed,budget)
        for day in range(300,365):
            result,steps=evaluate(days,name,agent,day,seed,keep_steps=True)
            result['budget']=budget; rows.append(result); traces.extend(steps)
    pd.DataFrame(rows).to_csv(OUT/f'test_seed_{seed}.csv',index=False)
    pd.DataFrame(traces).to_csv(OUT/f'trajectory_seed_{seed}.csv',index=False)
    print(f'test complete seed={seed}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','train','baselines','select','test']); parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args(); OUT.mkdir(exist_ok=True)
    {'prepare':prepare,'train':lambda:train(args.seed),'baselines':baselines,'select':select,'test':lambda:test(args.seed)}[args.stage]()
