from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import sys

for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE']:
    os.environ[name] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SNAP = ROOT / 'source_snapshot'
sys.path.insert(0, str(SNAP / 'src'))
import numpy as np
import pandas as pd
import scipy
import osqp
import causal_run as cr

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

protocol = json.loads((ROOT / 'protocol.json').read_text())
assert all(sha(SNAP / name) == value for name, value in protocol['source_files'].items())
days, fit = cr.make_days()
assert fit == json.loads((SNAP / 'artifacts/preprocessing.json').read_text())['fit']
model = cr.load_model('PPO-QP', 20260805, 90)
replayed, replay_steps = cr.evaluate(days, 'PPO-QP', model, 300, 20260805, keep_steps=True)
reference = pd.read_csv(SNAP / 'artifacts/test_seed_20260805.csv')
reference = reference[(reference.controller == 'PPO-QP') & (reference.split_day == 300)].iloc[0]
metrics = ['cost', 'carbon', 'task_score', 'peak', 'grid_excess_energy_kwh', 'mean_action_correction']
reference_delta = max(abs(replayed[k] - reference[k]) for k in metrics)
assert reference_delta < 1e-7
old_trace = pd.read_csv(SNAP / 'artifacts/trajectory_seed_20260805.csv')
old_trace = old_trace[(old_trace.controller == 'PPO-QP') & (old_trace.day == 300)].reset_index(drop=True)
new_trace = pd.DataFrame(replay_steps)
step_delta = float(np.max(np.abs(new_trace[['p', 'soc', 'grid_import', 'grid_export']].to_numpy() - old_trace[['p', 'soc', 'grid_import', 'grid_export']].to_numpy())))
assert step_delta < 1e-7 and new_trace.qp_execution_path.equals(old_trace.qp_execution_path)

daily, steps = [], []
for day in protocol['days']:
    env = cr.CausalEnv(days, cr.CFG, mode='qp', projection_penalty=0)
    env.reset(day, 0)
    for t in range(96):
        ev = min(1.0, env.ev_remaining / (env.ev_max * env.dt)) if days[day]['ev_active'][t] else 0.0
        raw = cr.core.raw_from_physical(env.P, ev, 1.0, env.P)
        before, ev_before = env.soc, env.ev_remaining
        _, reward, done, info = env.step(raw)
        actual = info['projected_action']
        row = {'controller': protocol['controller'], 'day': day, 'step': t, 'soc_before': before, 'ev_before': ev_before,
               'price': days[day]['price'][t], 'carbon_signal': days[day]['carbon'][t], 'reward': reward,
               'ev_power_kw': (actual[1] + 1) / 2 * env.ev_max * days[day]['ev_active'][t],
               **{k: info[k] for k in ['p', 'soc', 'grid_import', 'grid_export', 'grid_violation', 'correction', 'qp_execution_path']},
               **env.step_audit}
        for i, name in enumerate(['battery', 'ev', 'lighting']):
            row[name + '_proposal'] = raw[i]
            row[name + '_correction_squared'] = float((actual[i] - raw[i]) ** 2)
        steps.append(row)
    assert done
    daily.append({'controller': protocol['controller'], 'split_day': day,
                  **{k: float(v) for k, v in info.items() if isinstance(v, (int, float, np.number))},
                  'export_excess_energy_kwh': info['grid_export_violation_kwh'], 'import_excess_energy_kwh': info['grid_import_violation_kwh']})
    if (day - 299) % 10 == 0:
        print(f'Completed {day - 299}/65 diagnostic days', flush=True)

d, t = pd.DataFrame(daily), pd.DataFrame(steps)
t['ramp_kw'] = abs(t.p - t.groupby('day').p.shift().fillna(0))
t['ramp_slack_kw'] = (t.ramp_kw - 45).clip(lower=0)
t['forced_full_charge'] = ((t.power_lower_kw + 100).abs() <= 1e-5) & ((t.power_upper_kw + 100).abs() <= 1e-5)
t['charge_kwh'] = (-t.p).clip(lower=0) * 0.25
for day, block in t.groupby('day'):
    r = d[d.split_day == day].iloc[0]
    cost = float((block.price * block.grid_import * 0.25).sum())
    carbon = float((block.carbon_signal * block.grid_import * 0.25).sum())
    excess = float(block.grid_violation.sum() * 0.25)
    task = cost / 10 + 1.8 * carbon + float((0.0015 * abs(block.p) * 0.25 + 0.00035 * (block.grid_import - 220).clip(lower=0)**2 * 0.25).sum()) + excess
    assert max(abs(cost-r.cost), abs(carbon-r.carbon), abs(excess-r.grid_excess_energy_kwh), abs(task-r.task_score), abs(block.grid_import.max()-r.peak), abs(block.correction.mean()-r.mean_action_correction)) < 1e-7
    next_soc = block.soc_before + ((-block.p).clip(lower=0)*.95 - block.p.clip(lower=0)/.95)*.25
    assert np.max(abs(next_soc-block.soc)) < 1e-7
    assert abs(block.ev_power_kw.sum()*.25-days[day]['ev_required']) < 1e-8
    assert abs(r.terminal_soc_error_kwh) < 1e-8 and r.soc_violation_rate == 0 and r.ev_completion == 1 and abs(r.lighting_energy_kwh-472.5) < 1e-8
assert len(d) == 65 and len(t) == 6240 and t.groupby('day').size().eq(96).all()
window = t[t.step >= 80].groupby('day')
charges = window.charge_kwh.sum()
forced = window.forced_full_charge.any()
summary = {k: float(d[k].mean()) for k in metrics + ['export_excess_energy_kwh', 'import_excess_energy_kwh', 'ramp_slack_kw_mean', 'max_ramp_kw', 'bess_throughput_kwh']}
summary.update({'controller': protocol['controller'], 'episodes': 65, 'steps': 6240, 'forced_final_window_days': int(forced.sum()),
                'final_window_charge_median_kwh': float(charges.median()), 'final_window_charge_q25_kwh': float(charges.quantile(.25)), 'final_window_charge_q75_kwh': float(charges.quantile(.75)),
                'ramp_events': int((t.ramp_slack_kw > 1e-5).sum()), 'ramp_slack_max_kw': float(t.ramp_slack_kw.max()),
                'lower_bound_hours_per_day': float((((t.soc-25).abs() <= 1e-5) & ((t.soc_before-25).abs() <= 1e-5)).sum()*.25/65),
                'endpoint_infeasible_steps': int(t.endpoint_infeasible.sum()), 'feasible_state_violation_steps': int(t.feasible_state_violation.sum()),
                'execution_paths': t.qp_execution_path.value_counts().to_dict()})
means = t[[name+'_correction_squared' for name in ['battery', 'ev', 'lighting']]].mean()
summary['correction_squared_shares'] = {name: float(value / means.sum()) for name, value in means.items()}
summary['correction_rms_norm'] = float(np.sqrt(means.sum()))
pd.concat([pd.read_csv(SNAP / 'artifacts/summary_revision.csv'), pd.DataFrame([summary])], ignore_index=True).to_csv(ROOT/'table_III_comparison.csv', index=False)
ppo = pd.concat([pd.read_csv(p) for p in (SNAP/'artifacts').glob('test_seed_*.csv')], ignore_index=True)
ppo['export_excess_energy_kwh'] = ppo.grid_export_violation_kwh
ppo['import_excess_energy_kwh'] = ppo.grid_import_violation_kwh
rng = np.random.default_rng(protocol['bootstrap']['seed'])
indices = rng.integers(0, 65, (20000, 65))
contrasts = []
for controller, rows in ppo.groupby('controller'):
    for metric in metrics + ['export_excess_energy_kwh', 'import_excess_energy_kwh']:
        diff = rows.groupby('split_day')[metric].mean().sort_index().to_numpy() - d.sort_values('split_day')[metric].to_numpy()
        boots = diff[indices].mean(axis=1)
        contrasts.append({'controller': controller, 'comparator': protocol['controller'], 'metric': metric, 'mean_difference': float(diff.mean()), 'ci_lower': float(np.quantile(boots,.025)), 'ci_upper': float(np.quantile(boots,.975)), 'day_clusters': 65, 'fixed_ppo_seeds': 5})
d.to_csv(ROOT/'daily.csv', index=False)
t.to_csv(ROOT/'trajectory.csv', index=False)
pd.DataFrame(contrasts).to_csv(ROOT/'paired_PPO_minus_baseline.csv', index=False)
save('summary.json', summary)
save('verification.json', {'status': 'PASS', 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'reference_day': 300, 'reference_ppo_seed': 20260805, 'reference_daily_max_difference': reference_delta, 'reference_step_max_difference': step_delta, 'reference_qp_paths_identical': True, 'independently_recomputed_metrics': ['cost','carbon','task_score','grid_excess_energy_kwh','peak','mean_action_correction','battery_energy_balance','EV_delivery'], 'episodes': 65, 'steps': 6240, 'service_failures': 0, 'source_hashes_unchanged': all(sha(SNAP/name)==value for name,value in protocol['source_files'].items()), 'protocol_sha256': sha(ROOT/'protocol.json'), 'runner_sha256': sha(Path(__file__)), 'python': sys.version, 'platform': platform.platform(), 'versions': {'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'osqp':osqp.__version__}, 'outputs': {p.name:sha(p) for p in ROOT.glob('*.csv')}})
save('metric_contract.json', {'status':'comparison_ready', 'controller':protocol['controller'], 'split':'fixed days 300-364, one deterministic trajectory per day', 'metrics': metrics + ['export_excess_energy_kwh','import_excess_energy_kwh'], 'direction':'lower is favorable for all operating metrics; correction is a diagnostic, not a safety ranking', 'protocol_sha256':sha(ROOT/'protocol.json'), 'summary_sha256':sha(ROOT/'summary.json'), 'validation':'Independent trajectory aggregation, energy/service checks, one exact archived PPO-day replay, source hash verification', 'limitations':['Post-hoc diagnostic on an already inspected scenario','No cap search or training','EV proposal differs from each PPO policy; this is a complete simple controller comparator, not a pure battery-action ablation']})
print(json.dumps(summary, indent=2), flush=True)
