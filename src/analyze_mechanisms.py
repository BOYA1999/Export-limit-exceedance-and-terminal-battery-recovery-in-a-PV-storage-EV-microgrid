from pathlib import Path
import json
import numpy as np
import pandas as pd
import causal_run as cr

OUT = cr.OUT
BASELINES = ['Rule+QP', 'NetLoadRule+QP', 'MPC-H24+QP', 'MPC-H24-FRR+QP']
KEYS = ['controller', 'seed', 'day']
TOL = 1e-5
UNITS = {'cost': 'benchmark cost score/day', 'carbon': 'benchmark carbon score/day',
         'task_score': 'benchmark task score/day', 'peak': 'kW', 'grid_excess_energy_kwh': 'kWh/day',
         'import_excess_energy_kwh': 'kWh/day', 'export_excess_energy_kwh': 'kWh/day',
         'mean_action_correction': 'dimensionless action norm', 'ramp_slack_kw_mean': 'kW (per-step mean)'}


def main():
    cr.verify_contract()
    verification = json.loads((OUT / 'aggregation_verification.json').read_text())
    assert verification['status'] == 'passed'
    assert all(cr.core.sha256(OUT / name) == digest for name, digest in verification['source_sha256'].items())
    selection = json.loads((OUT / 'selection.json').read_text())
    assert all(cr.core.sha256(OUT / name) == digest for name, digest in selection['selected_model_hashes'].items())
    sources = [OUT / f'trajectory_seed_{seed}.csv' for seed in cr.SEEDS] + [OUT / f'trajectory_{name}.csv' for name in BASELINES]
    trace = pd.concat([pd.read_csv(p) for p in sources], ignore_index=True).sort_values(KEYS + ['step']).reset_index(drop=True)
    seeds = pd.read_csv(OUT / 'seed_means.csv').set_index(['controller', 'seed']).sort_index()
    summary = pd.read_csv(OUT / 'summary.csv').set_index('controller')
    required = ['seed', 'day', 'step', 'p', 'soc', 'grid_import', 'grid_export', 'grid_violation',
                'power_lower_kw', 'power_upper_kw', 'physical_lower_kw', 'physical_upper_kw',
                'terminal_lower_kw', 'terminal_upper_kw', 'base_load_minus_pv_kw', 'lighting_kw']
    assert np.isfinite(trace[required].to_numpy()).all()
    assert len(trace) == 87360 and not trace.duplicated(KEYS + ['step']).any()
    assert trace.groupby(KEYS).step.apply(lambda x: list(x) == list(range(96))).all()
    cfg = cr.CFG
    dt, power, ramp = cfg['dt_hours'], cfg['bess']['power_kw'], cfg['bess']['ramp_kw']
    emin, emax = cfg['bess']['energy_kwh'] * cfg['bess']['soc_min'], cfg['bess']['energy_kwh'] * cfg['bess']['soc_max']
    initial = cfg['bess']['energy_kwh'] * cfg['bess']['soc_initial']
    trace['soc_before_kwh'] = trace.groupby(KEYS).soc.shift().fillna(initial)
    trace['ramp_kw'] = (trace.p - trace.groupby(KEYS).p.shift().fillna(0)).abs()
    trace['ramp_excess_kw'] = (trace.ramp_kw - ramp).clip(lower=0)
    trace['ramp_violation'] = trace.ramp_excess_kw > 1e-9
    trace['ramp_material_violation'] = trace.ramp_excess_kw > 1e-5
    trace['import_excess_kw'] = (trace.grid_import - cfg['loads']['grid_max_kw'] - .01).clip(lower=0)
    trace['export_excess_kw'] = (trace.grid_export + cfg['loads']['grid_min_kw'] - .01).clip(lower=0)
    assert np.allclose(trace.import_excess_kw + trace.export_excess_kw, trace.grid_violation, atol=1e-7, rtol=0)
    trace['battery_box_width_kw'] = trace.power_upper_kw - trace.power_lower_kw
    trace['collapsed_box'] = trace.battery_box_width_kw <= TOL
    trace['narrowed_box'] = trace.battery_box_width_kw < 2 * power - TOL
    trace['forced_full_charge_box'] = (trace.power_lower_kw + power).abs().le(TOL) & (trace.power_upper_kw + power).abs().le(TOL)
    trace['executed_full_charge'] = (trace.p + power).abs().le(TOL)
    trace['terminal_envelope_narrows_physical_box'] = (trace.terminal_lower_kw > trace.physical_lower_kw + TOL) | (trace.terminal_upper_kw < trace.physical_upper_kw - TOL)
    for label, bound in [('lower', emin), ('upper', emax)]:
        trace[f'soc_{label}_step_end'] = (trace.soc - bound).abs() <= TOL
        trace[f'soc_{label}_whole_interval'] = trace[f'soc_{label}_step_end'] & ((trace.soc_before_kwh - bound).abs() <= TOL)
    peaks = []
    for key, group in trace.groupby(KEYS):
        peak = group.loc[group.grid_import.idxmax()]
        near = group[(group.grid_import - peak.grid_import).abs() <= TOL]
        peaks.append({**dict(zip(KEYS, key)), 'peak_kw': peak.grid_import, 'peak_step': int(peak.step),
                      'peak_time_hours': peak.step * dt, 'first_maximum_in_last_hour': int(peak.step >= 92),
                      'near_maximum_in_last_hour': int((near.step >= 92).any()), 'near_maximum_steps': len(near),
                      **{k: peak[k] for k in ['p', 'soc', 'soc_before_kwh', 'power_lower_kw', 'power_upper_kw',
                                               'battery_box_width_kw', 'collapsed_box', 'narrowed_box', 'forced_full_charge_box',
                                               'executed_full_charge', 'terminal_envelope_narrows_physical_box']}})
    peaks = pd.DataFrame(peaks)
    by_controller = []
    for name, group in trace.groupby('controller'):
        peak = peaks[peaks.controller == name]
        episodes = len(peak)
        row = {'controller': name, 'episodes': episodes, 'steps': len(group), 'scenario_days': 65,
               'trained_seeds': 5 if name in cr.VARIANTS else 0,
               'peak_in_last_hour_count': int(peak.first_maximum_in_last_hour.sum()),
               'peak_in_last_hour_fraction': float(peak.first_maximum_in_last_hour.mean()),
               'near_peak_in_last_hour_fraction': float(peak.near_maximum_in_last_hour.mean()),
               'ramp_violation_steps': int(group.ramp_violation.sum()), 'ramp_violation_fraction': float(group.ramp_violation.mean()),
               'ramp_violation_threshold_kw': 1e-9,
               'ramp_material_violation_steps': int(group.ramp_material_violation.sum()),
               'ramp_material_violation_fraction': float(group.ramp_material_violation.mean()), 'ramp_material_violation_threshold_kw': 1e-5,
               'ramp_mean_kw': float(group.ramp_kw.mean()), 'ramp_max_kw': float(group.ramp_kw.max()),
               'ramp_excess_mean_kw_per_step': float(group.ramp_excess_kw.mean())}
        for flag in ['collapsed_box', 'narrowed_box', 'forced_full_charge_box', 'executed_full_charge', 'terminal_envelope_narrows_physical_box']:
            row['peak_' + flag + '_count'] = int(peak[flag].sum())
            row['peak_' + flag + '_fraction'] = float(peak[flag].mean())
            row['all_steps_' + flag + '_fraction'] = float(group[flag].mean())
        for direction in ['import', 'export']:
            x = group[direction + '_excess_kw']
            row[direction + '_exceedance_steps'] = int((x > 1e-12).sum())
            row[direction + '_excess_total_kwh'] = float(x.sum() * dt)
            row[direction + '_excess_kwh_per_day'] = float(x.sum() * dt / episodes)
            assert abs(row[direction + '_excess_kwh_per_day'] - summary.loc[name, direction + '_excess_energy_kwh']) < 1e-7
        for label in ['lower', 'upper']:
            row[f'soc_{label}_step_end_fraction'] = float(group[f'soc_{label}_step_end'].mean())
            row[f'soc_{label}_step_end_equivalent_hours_per_day'] = float(group[f'soc_{label}_step_end'].sum() * dt / episodes)
            row[f'soc_{label}_whole_interval_hours_per_day'] = float(group[f'soc_{label}_whole_interval'].sum() * dt / episodes)
        by_controller.append(row)
    hourly = trace.groupby(['controller', 'step']).agg(samples=('p', 'size'), battery_power_mean_kw=('p', 'mean'),
              soc_step_end_mean_kwh=('soc', 'mean'), ramp_mean_kw=('ramp_kw', 'mean'), ramp_max_kw=('ramp_kw', 'max'),
              ramp_excess_mean_kw=('ramp_excess_kw', 'mean'), ramp_violation_fraction=('ramp_violation', 'mean'),
              ramp_material_violation_fraction=('ramp_material_violation', 'mean'),
              import_excess_mean_kw=('import_excess_kw', 'mean'), export_excess_mean_kw=('export_excess_kw', 'mean')).reset_index()
    hourly['time_hours'] = hourly.step * dt
    hourly['ramp_violation_threshold_kw'] = 1e-9
    hourly['ramp_material_violation_threshold_kw'] = 1e-5
    differences = seeds.loc['PPO-QP', list(UNITS)] - seeds.loc['PPO-QP-no-penalty', list(UNITS)]
    assert list(differences.index) == cr.SEEDS
    penalty, leave_one_out = [], []
    for metric, unit in UNITS.items():
        x = differences[metric]
        penalty.append({'metric': metric, 'unit': unit, 'mean_difference': float(x.mean()), 'median_difference': float(x.median()),
                        'minimum_seed_difference': float(x.min()), 'maximum_seed_difference': float(x.max()),
                        'negative_seed_count': int((x < -1e-9).sum()), 'positive_seed_count': int((x > 1e-9).sum())})
        for seed in cr.SEEDS:
            y = x.drop(seed)
            leave_one_out.append({'metric': metric, 'unit': unit, 'omitted_seed': seed, 'remaining_seeds': 4,
                                  'mean_difference': float(y.mean()), 'median_difference': float(y.median())})
    days, _ = cr.make_days()
    actor_rows = []
    for (name, seed), policy in trace[trace.controller.isin(cr.VARIANTS)].groupby(['controller', 'seed']):
        budget = selection['selected_budgets'][name]
        model = cr.load_model(name, int(seed), budget)
        observations, actuals, ev_residuals = [], [], []
        for day, group in policy.groupby('day'):
            env = cr.CausalEnv(days, cfg, mode='qp', projection_penalty=cr.VARIANTS[name]); env.reset(int(day), int(seed))
            for r in group.itertuples(index=False):
                env.t = int(r.step); observations.append(env.observation())
                ev_kw = r.grid_import - r.grid_export - r.base_load_minus_pv_kw - r.lighting_kw + r.p
                assert -1e-7 <= ev_kw <= env.ev_max + 1e-7
                assert ev_kw * dt <= env.ev_remaining + 1e-7
                actuals.append([r.p / power, 2 * ev_kw / env.ev_max - 1, 1])
                env.ev_remaining = max(0, env.ev_remaining - ev_kw * dt)
                env.soc, env.prev_p = r.soc, r.p
            ev_residuals.append(env.ev_remaining)
        latent = model.actor.forward(np.asarray(observations))[0]
        raw = np.clip(latent, -1, 1); correction = np.asarray(actuals) - raw
        norms = np.linalg.norm(correction, axis=1)
        reference = float(seeds.loc[(name, seed), 'mean_action_correction'])
        assert abs(norms.mean() - reference) < 1e-6 and max(ev_residuals) < 1e-7
        row = {'controller': name, 'seed': int(seed), 'selected_budget': budget, 'states': len(raw),
               'mean_correction_reconstructed': float(norms.mean()), 'mean_correction_seed_means': reference,
               'correction_absolute_difference': float(abs(norms.mean() - reference)),
               'max_ev_reconstruction_residual_kwh': float(max(ev_residuals)),
               'battery_raw_positive_fraction': float((raw[:, 0] > 0).mean()),
               'battery_raw_plus_one_fraction': float((raw[:, 0] == 1).mean()),
               'battery_raw_minus_one_fraction': float((raw[:, 0] == -1).mean()),
               'battery_latent_min': float(latent[:, 0].min()), 'battery_latent_max': float(latent[:, 0].max())}
        for i, label in enumerate(['battery', 'ev', 'lighting']):
            row[label + '_correction_mse_dimensionless'] = float(np.square(correction[:, i]).mean())
        actor_rows.append(row)
    tables = {'mechanism_controller_summary': pd.DataFrame(by_controller), 'mechanism_peak_events': peaks,
              'mechanism_by_step': hourly, 'mechanism_penalty_seed_differences': differences.reset_index(),
              'mechanism_penalty_summary': pd.DataFrame(penalty), 'mechanism_penalty_leave_one_seed_out': pd.DataFrame(leave_one_out),
              'mechanism_actor_diagnostics': pd.DataFrame(actor_rows)}
    for name, table in tables.items():
        assert np.isfinite(table.select_dtypes(include='number').to_numpy()).all(), name
        table.to_csv(OUT / (name + '.csv'), index=False)
    report = {'status': 'passed', 'definitions': {'peak_step': 'earliest exact daily import maximum; zero-based 15-minute interval',
              'near_peak': 'within 1e-5 kW of daily maximum, reported separately to expose ties', 'last_hour': 'steps 92 through 95',
              'box_binding_tolerance_kw': TOL, 'narrowed_box': 'width below physical 200 kW range; includes ordinary ramp restrictions',
              'forced_full_charge_box': 'both effective battery endpoints within 1e-5 kW of -100 kW',
              'soc_binding_tolerance_kwh': TOL, 'soc_residence': 'step-end equivalent hours are an occupancy proxy; whole-interval hours require both endpoints at the bound within tolerance',
              'ramp': 'absolute executed power change from preceding step; day starts with previous power zero; excess above 45 kW',
              'ramp_material_events': 'primary event reporting uses ramp_material_violation: excess > 1e-5 kW, matching baseline_solver_audit',
              'ramp_numerical_sensitivity': 'retained ramp_violation columns use excess > 1e-9 kW as a sensitivity; do not compare event counts across thresholds without labeling',
              'ramp_mean_slack': 'all positive excess magnitudes contribute to the unchanged average; event-reporting thresholds do not censor slack magnitudes',
              'grid_excess': 'positive power beyond import/export limit after subtracting 0.01 kW; multiply by 0.25 h for energy',
              'penalty_direction': 'PPO-QP minus PPO-QP-no-penalty; negative favors the penalty variant for listed minimized metrics',
              'uncertainty_scope': 'descriptive fixed five-seed comparisons and all five leave-one-seed-out sensitivities, not new-seed confidence intervals',
              'actor': 'selected-budget deterministic clipped means reconstructed at saved executed states; no QP or training rerun',
              'actor_dimensions': 'battery, EV, lighting; squared corrections are in actor coordinates, not physical power or causal attribution'},
              'controllers': tables['mechanism_controller_summary'].to_dict('records'), 'penalty': penalty,
              'leave_one_seed_out': leave_one_out, 'actors': actor_rows,
              'peak_step_counts': {name: {str(int(k)): int(v) for k, v in g.peak_step.value_counts().sort_index().items()} for name, g in peaks.groupby('controller')},
              'source_sha256': {p.name: cr.core.sha256(p) for p in sources + [OUT / 'seed_means.csv', OUT / 'summary.csv', OUT / 'selection.json', OUT / 'aggregation_verification.json']},
              'script_sha256': cr.core.sha256(Path(__file__)),
              'output_sha256': {name + '.csv': cr.core.sha256(OUT / (name + '.csv')) for name in tables}}
    (OUT / 'mechanism_analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'controllers': len(by_controller), 'peak_events': len(peaks), 'time_bins': len(hourly), 'policy_seed_reconstructions': len(actor_rows)}))


if __name__ == '__main__':
    main()
