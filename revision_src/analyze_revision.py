from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import causal_run as cr

OUT = cr.OUT
DT = 0.25
CONTROLLERS = ['Rule+QP', 'NetLoadRule+QP', 'MPC-H24+QP', 'MPC-H24-FRR+QP', 'MPC-H24-XP+QP', 'PPO-QP', 'PPO-QP-no-penalty']


def load_traces():
    frames = [pd.read_csv(OUT / f'trajectory_{name}.csv') for name in CONTROLLERS[:5]]
    frames += [pd.read_csv(path) for path in sorted(OUT.glob('trajectory_seed_*.csv'))]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.controller.isin(CONTROLLERS)].copy()
    assert len(frame) == 93600
    assert (frame.groupby(['controller', 'seed', 'day']).size() == 96).all()
    return frame


def split_difficulty(days):
    rows = []
    for split, selected in [('training', range(0, 240)), ('validation', range(240, 300)), ('evaluation', range(300, 365))]:
        load = np.concatenate([days[d]['load'] for d in selected])
        pv = np.concatenate([days[d]['pv'] for d in selected])
        lighting = np.concatenate([30.0 * days[d]['light_profile'] for d in selected])
        net = load + lighting - pv
        raw_daily = [float(np.maximum(0.0, -40.0 - (days[d]['load'] + 30.0 * days[d]['light_profile'] - days[d]['pv']) - 0.01).sum() * DT) for d in selected]
        errors = {k: [] for k in ['load', 'pv', 'net']}
        for d in selected:
            for t in range(96):
                for h in range(1, min(24, 96 - t)):
                    if d == 0:
                        forecast_load = days[d]['load'][t]
                        forecast_pv = days[d]['pv'][t]
                    else:
                        forecast_load = days[d - 1]['load'][t + h]
                        forecast_pv = days[d - 1]['pv'][t + h]
                    errors['load'].append(abs(forecast_load - days[d]['load'][t + h]))
                    errors['pv'].append(abs(forecast_pv - days[d]['pv'][t + h]))
                    forecast_net = forecast_load + 30.0 * days[d]['light_profile'][t + h] - forecast_pv
                    actual_net = days[d]['load'][t + h] + 30.0 * days[d]['light_profile'][t + h] - days[d]['pv'][t + h]
                    errors['net'].append(abs(forecast_net - actual_net))
        rows.append({'split': split, 'days': len(selected), 'load_mean_kw': float(load.mean()), 'load_p95_kw': float(np.quantile(load, .95)),
                     'pv_mean_kw': float(pv.mean()), 'pv_p95_kw': float(np.quantile(pv, .95)),
                     'exogenous_net_mean_kw': float(net.mean()), 'exogenous_net_p05_kw': float(np.quantile(net, .05)),
                     'raw_export_pressure_kwh_per_day': float(np.mean(raw_daily)), 'raw_export_days_fraction': float(np.mean(np.asarray(raw_daily) > 0)),
                     'previous_day_load_mae_kw': float(np.mean(errors['load'])), 'previous_day_pv_mae_kw': float(np.mean(errors['pv'])),
                     'previous_day_net_mae_kw': float(np.mean(errors['net']))})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'split_difficulty.csv', index=False)
    return result


def terminal_recovery(trace, days):
    trace = trace.copy()
    trace['price'] = [days[int(d)]['price'][int(t)] for d, t in zip(trace.day, trace.step)]
    trace['forced_full_charge'] = ((trace.power_lower_kw + 100).abs() <= 1e-5) & ((trace.power_upper_kw + 100).abs() <= 1e-5)
    final = trace[trace.step >= 80].copy()
    step_rows = []
    for (controller, step), group in final.groupby(['controller', 'step'], sort=False):
        step_rows.append({'controller': controller, 'step': int(step), 'time_hours': (int(step) + 1) * DT, 'samples': len(group),
                          'soc_q25_kwh': float(group.soc.quantile(.25)), 'soc_median_kwh': float(group.soc.median()), 'soc_q75_kwh': float(group.soc.quantile(.75)),
                          'power_q25_kw': float(group.p.quantile(.25)), 'power_median_kw': float(group.p.median()), 'power_q75_kw': float(group.p.quantile(.75)),
                          'forced_full_charge_fraction': float(group.forced_full_charge.mean())})
    by_step = pd.DataFrame(step_rows)
    by_step.to_csv(OUT / 'terminal_recovery_by_step.csv', index=False)
    daily_rows = []
    for keys, group in final.groupby(['controller', 'seed', 'day'], sort=False):
        before = trace[(trace.controller == keys[0]) & (trace.seed == keys[1]) & (trace.day == keys[2]) & (trace.step == 79)].soc.iloc[0]
        daily_rows.append({'controller': keys[0], 'seed': int(keys[1]), 'day': int(keys[2]),
                           'charge_energy_final4_kwh': float(np.maximum(-group.p, 0).sum() * DT),
                           'soc_change_final4_kwh': float(group.iloc[-1].soc - before),
                           'gross_import_cost_final4': float((group.price * group.grid_import).sum() * DT),
                           'export_excess_final4_kwh': float(np.maximum(group.grid_export - 40.0 - 0.01, 0).sum() * DT),
                           'forced_full_charge_steps_final4': int(group.forced_full_charge.sum()),
                           'any_forced_full_charge_final4': int(group.forced_full_charge.any())})
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT / 'terminal_recovery_daily.csv', index=False)
    summary = []
    for controller, group in daily.groupby('controller', sort=False):
        controller_trace = trace[trace.controller == controller].copy()
        controller_trace['previous_p'] = controller_trace.groupby(['seed', 'day']).p.shift().fillna(0.0)
        ramp_slack = np.maximum(np.abs(controller_trace.p - controller_trace.previous_p) - 45.0, 0.0)
        row = {'controller': controller, 'episodes': len(group), 'any_forced_full_charge_fraction': float(group.any_forced_full_charge_final4.mean()),
               'ramp_slack_kw_p95': float(np.quantile(ramp_slack, .95)), 'ramp_slack_kw_max': float(ramp_slack.max()),
               'ramp_material_event_count': int(np.sum(ramp_slack > 1e-5))}
        for metric in ['charge_energy_final4_kwh', 'soc_change_final4_kwh', 'gross_import_cost_final4', 'export_excess_final4_kwh', 'forced_full_charge_steps_final4']:
            row[f'{metric}_mean'] = float(group[metric].mean())
            row[f'{metric}_q25'] = float(group[metric].quantile(.25))
            row[f'{metric}_median'] = float(group[metric].median())
            row[f'{metric}_q75'] = float(group[metric].quantile(.75))
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'terminal_recovery_by_controller.csv', index=False)
    return summary, by_step


def endpoint_excess(frame, p_lo, p_hi, ev_lo, ev_hi):
    base = frame.base_load_minus_pv_kw.to_numpy() + frame.lighting_kw.to_numpy()
    grid_lo = base + 60.0 * ev_lo - p_hi
    grid_hi = base + 60.0 * ev_hi - p_lo
    import_excess = np.maximum(0.0, grid_lo - 260.0 - 0.01)
    export_excess = np.maximum(0.0, -40.0 - grid_hi - 0.01)
    excess = np.maximum(import_excess, export_excess)
    return excess, grid_hi - grid_lo, import_excess, export_excess


def constraint_sources(trace):
    frame = trace.copy()
    frame['previous_p'] = frame.groupby(['controller', 'seed', 'day']).p.shift().fillna(0.0)
    actual, actual_width, actual_import, actual_export = endpoint_excess(frame, frame.power_lower_kw.to_numpy(), frame.power_upper_kw.to_numpy(), frame.ev_lower_fraction.to_numpy(), frame.ev_upper_fraction.to_numpy())
    assert np.max(np.abs(actual - frame.endpoint_minimum_violation_kw.to_numpy())) < 1e-7
    physical_lo, physical_hi = frame.physical_lower_kw.to_numpy(), frame.physical_upper_kw.to_numpy()
    ramp_lo, ramp_hi = frame.previous_p.to_numpy() - 45.0, frame.previous_p.to_numpy() + 45.0
    terminal_lo = np.maximum(physical_lo, frame.terminal_lower_kw.to_numpy())
    terminal_hi = np.minimum(physical_hi, frame.terminal_upper_kw.to_numpy())
    physical_ramp_lo, physical_ramp_hi = np.maximum(physical_lo, ramp_lo), np.minimum(physical_hi, ramp_hi)
    tolerance = frame.tolerance_kw.to_numpy()
    ramp_enforced = (physical_ramp_lo <= physical_ramp_hi) & frame.terminal_ramp_relaxation.eq(0).to_numpy()
    no_terminal_lo = np.where(ramp_enforced, physical_ramp_lo, physical_lo)
    no_terminal_hi = np.where(ramp_enforced, physical_ramp_hi, physical_hi)
    no_ramp_lo = np.maximum(physical_lo, frame.terminal_lower_kw.to_numpy())
    no_ramp_hi = np.minimum(physical_hi, frame.terminal_upper_kw.to_numpy())
    for lower, upper in [(no_terminal_lo, no_terminal_hi), (no_ramp_lo, no_ramp_hi)]:
        assert np.all(lower <= frame.power_lower_kw.to_numpy() + tolerance + 1e-12)
        assert np.all(upper >= frame.power_upper_kw.to_numpy() - tolerance - 1e-12)
    assert np.all(frame.ev_lower_fraction.to_numpy() >= 0)
    comparisons = {
        'remove_terminal_envelope': endpoint_excess(frame, no_terminal_lo, no_terminal_hi, frame.ev_lower_fraction.to_numpy(), frame.ev_upper_fraction.to_numpy()),
        'remove_ramp_interval': endpoint_excess(frame, no_ramp_lo, no_ramp_hi, frame.ev_lower_fraction.to_numpy(), frame.ev_upper_fraction.to_numpy()),
        'remove_ev_lower_bound': endpoint_excess(frame, frame.power_lower_kw.to_numpy(), frame.power_upper_kw.to_numpy(), np.zeros(len(frame)), frame.ev_upper_fraction.to_numpy())}
    rows = []
    for controller, indices in frame.groupby('controller', sort=False).groups.items():
        idx = np.asarray(list(indices), int)
        episodes = frame.iloc[idx].groupby(['seed', 'day']).ngroups
        for source, (counterfactual, width, counterfactual_import, counterfactual_export) in comparisons.items():
            assert np.all(counterfactual[idx] <= actual[idx] + 1e-7)
            rows.append({'controller': controller, 'one_step_counterfactual': source, 'steps': len(idx),
                         'actual_endpoint_infeasible_steps': int(np.sum(actual[idx] > 1e-9)),
                         'actual_import_infeasible_steps': int(np.sum(actual_import[idx] > 1e-9)),
                         'actual_export_infeasible_steps': int(np.sum(actual_export[idx] > 1e-9)),
                         'resolved_endpoint_infeasible_steps': int(np.sum((actual[idx] > 1e-9) & (counterfactual[idx] <= 1e-9))),
                         'resolved_import_infeasible_steps': int(np.sum((actual_import[idx] > 1e-9) & (counterfactual_import[idx] <= 1e-9))),
                         'resolved_export_infeasible_steps': int(np.sum((actual_export[idx] > 1e-9) & (counterfactual_export[idx] <= 1e-9))),
                         'new_endpoint_infeasible_steps': int(np.sum((actual[idx] <= 1e-9) & (counterfactual[idx] > 1e-9))),
                         'mean_change_minimum_excess_kw_all_steps': float(np.mean(counterfactual[idx] - actual[idx])),
                         'mean_change_attainable_grid_width_kw_all_steps': float(np.mean(width[idx] - actual_width[idx])),
                         'mean_daily_change_minimum_excess_kwh': float(np.sum(counterfactual[idx] - actual[idx]) * DT / episodes),
                         'subset_check_passed': True,
                         'monotonic_excess_check_passed': True})
    result = pd.DataFrame(rows)
    assert result.new_endpoint_infeasible_steps.eq(0).all()
    result.to_csv(OUT / 'constraint_source_decomposition.csv', index=False)

    rated_lo = np.full(len(frame), -float(cr.CFG['bess']['power_kw']))
    energy_headroom_lo = -(float(cr.CFG['bess']['soc_max']) * float(cr.CFG['bess']['energy_kwh']) - frame.soc_kwh.to_numpy()) / (float(cr.CFG['bess']['eta_charge']) * DT)
    fixed_terminal_lo = frame.terminal_lower_kw.to_numpy()
    actual_lo = frame.power_lower_kw.to_numpy()
    effective_ramp_lo = np.where(ramp_enforced, ramp_lo, -np.inf)
    terminal_additional = fixed_terminal_lo > physical_lo + tolerance
    effective_terminal_lo = np.where(terminal_additional, fixed_terminal_lo, -np.inf)
    lower_without_rating = np.maximum.reduce([energy_headroom_lo, effective_terminal_lo, effective_ramp_lo])
    lower_without_headroom = np.maximum.reduce([rated_lo, effective_terminal_lo, effective_ramp_lo])
    assert np.all(lower_without_rating <= actual_lo + tolerance + 1e-12)
    assert np.all(lower_without_headroom <= actual_lo + tolerance + 1e-12)
    no_rating = endpoint_excess(frame, lower_without_rating, frame.power_upper_kw.to_numpy(), frame.ev_lower_fraction.to_numpy(), frame.ev_upper_fraction.to_numpy())
    no_headroom = endpoint_excess(frame, lower_without_headroom, frame.power_upper_kw.to_numpy(), frame.ev_lower_fraction.to_numpy(), frame.ev_upper_fraction.to_numpy())
    assert np.all(no_rating[0] <= actual + 1e-7) and np.all(no_headroom[0] <= actual + 1e-7)
    active = {
        'rated_charge_power': np.abs(actual_lo - rated_lo) <= 1e-5,
        'energy_headroom': np.abs(actual_lo - energy_headroom_lo) <= 1e-5,
        'terminal_lower_envelope': terminal_additional & (np.abs(actual_lo - fixed_terminal_lo) <= 1e-5),
        'ramp_lower_bound': ramp_enforced & (np.abs(actual_lo - ramp_lo) <= 1e-5)}
    charge_rows = []
    for controller, indices in frame.groupby('controller', sort=False).groups.items():
        idx = np.asarray(list(indices), int)
        export_idx = idx[actual_export[idx] > 1e-9]
        episodes = frame.iloc[idx].groupby(['seed', 'day']).ngroups
        charge_rows.append({'controller': controller, 'actual_export_infeasible_steps': len(export_idx),
                            **{f'active_{name}_steps': int(mask[export_idx].sum()) for name, mask in active.items()},
                            'resolved_export_steps_remove_charge_rating': int(np.sum(no_rating[3][export_idx] <= 1e-9)),
                            'resolved_export_steps_remove_energy_headroom': int(np.sum(no_headroom[3][export_idx] <= 1e-9)),
                            'mean_daily_export_minimum_change_remove_charge_rating_kwh': float(np.sum(no_rating[3][idx] - actual_export[idx]) * DT / episodes),
                            'mean_daily_export_minimum_change_remove_energy_headroom_kwh': float(np.sum(no_headroom[3][idx] - actual_export[idx]) * DT / episodes),
                            'active_counts_nonexclusive': True})
    charge = pd.DataFrame(charge_rows)
    charge.to_csv(OUT / 'charge_bound_diagnostics.csv', index=False)

    timing_rows = []
    for controller, indices in frame.groupby('controller', sort=False).groups.items():
        idx = np.asarray(list(indices), int)
        events = frame.iloc[idx][actual_export[idx] > 1e-9]
        steps = events.step.to_numpy(int)
        timing_rows.append({'controller': controller, 'export_infeasible_steps': len(events),
                            'earliest_step': int(steps.min()) if len(steps) else None, 'latest_step': int(steps.max()) if len(steps) else None,
                            'events_steps_0_23': int(np.sum(steps < 24)), 'events_steps_24_47': int(np.sum((steps >= 24) & (steps < 48))),
                            'events_steps_48_71': int(np.sum((steps >= 48) & (steps < 72))), 'events_steps_72_95': int(np.sum(steps >= 72)),
                            'events_last_five_actions_91_95': int(np.sum(steps >= 91)), 'events_last_four_actions_92_95': int(np.sum(steps >= 92))})
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(OUT / 'export_event_timing.csv', index=False)
    return result, charge, timing


def export_priority_comparison():
    original = pd.read_csv(OUT / 'baseline_MPC-H24+QP.csv').set_index('split_day')
    priority = pd.read_csv(OUT / 'baseline_MPC-H24-XP+QP.csv').set_index('split_day')
    metrics = ['cost', 'carbon', 'task_score', 'peak', 'grid_excess_energy_kwh', 'grid_export_violation_kwh', 'ramp_slack_kw_mean', 'max_ramp_kw']
    rng = np.random.default_rng(20260918)
    rows = []
    for metric in metrics:
        diff = priority[metric].to_numpy() - original[metric].to_numpy()
        boots = diff[rng.integers(0, len(diff), size=(20000, len(diff)))].mean(axis=1)
        rows.append({'metric': metric, 'original_mean': float(original[metric].mean()), 'export_priority_mean': float(priority[metric].mean()),
                     'mean_difference_export_priority_minus_original': float(diff.mean()),
                     'day_cluster_ci_lower': float(np.quantile(boots, .025)), 'day_cluster_ci_upper': float(np.quantile(boots, .975))})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'export_priority_comparison.csv', index=False)
    return result


def figure2_export_priority_intervals():
    policy = pd.concat([pd.read_csv(path) for path in sorted(OUT.glob('test_seed_*.csv'))], ignore_index=True)
    policy = policy[policy.controller == 'PPO-QP'].groupby('split_day')[['cost', 'peak', 'grid_export_violation_kwh']].mean()
    priority = pd.read_csv(OUT / 'baseline_MPC-H24-XP+QP.csv').set_index('split_day')
    rng = np.random.default_rng(20260918)
    rows = []
    for metric, column in [('cost', 'cost'), ('peak', 'peak'), ('export_excess_energy_kwh', 'grid_export_violation_kwh')]:
        difference = policy[column].to_numpy() - priority[column].to_numpy()
        bootstrap = difference[rng.integers(0, len(difference), size=(20000, len(difference)))].mean(axis=1)
        rows.append({'metric': metric, 'comparator': 'MPC-H24-XP+QP', 'mean_difference_ppo_qp_minus_comparator': float(difference.mean()),
                     'day_cluster_ci_lower': float(np.quantile(bootstrap, .025)), 'day_cluster_ci_upper': float(np.quantile(bootstrap, .975)),
                     'day_clusters': 65, 'fixed_ppo_seeds': 5, 'bootstrap_replicates': 20000})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'figure2_export_priority_intervals.csv', index=False)
    return result


def solver_stage_summary():
    records = [json.loads(line) for line in (OUT / 'mpc_diagnostics_MPC-H24-XP+QP.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = []
    for phase in ['phase1', 'phase2']:
        attempted = [record for record in records if record.get(f'{phase}_status') is not None]
        categories = {
            'optimal_accepted': [record for record in attempted if record.get(f'{phase}_status') == 0 and record.get(f'{phase}_feasible')],
            'time_limit_accepted': [record for record in attempted if record.get(f'{phase}_status') == 1 and record.get(f'{phase}_feasible')],
            'no_accepted_incumbent': [record for record in attempted if not record.get(f'{phase}_feasible')]}
        assert sum(len(selected) for selected in categories.values()) == len(attempted)
        for label, selected in categories.items():
            gaps = [record.get(f'{phase}_mip_gap') for record in selected if record.get(f'{phase}_mip_gap') is not None] if label == 'time_limit_accepted' else []
            rows.append({'phase': phase, 'status': label, 'calls': len(selected),
                         'mip_gap_min': float(np.min(gaps)) if gaps else np.nan,
                         'mip_gap_median': float(np.median(gaps)) if gaps else np.nan,
                         'mip_gap_max': float(np.max(gaps)) if gaps else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'mpc_export_priority_solver_stage_summary.csv', index=False)
    return result


def export_priority_repeat_sensitivity():
    rows = []
    for label, suffix in [('initial_frozen_run', '_initial'), ('stage_logging_rerun', '')]:
        daily = pd.read_csv(OUT / f'baseline_MPC-H24-XP+QP{suffix}.csv')
        execution = json.loads((OUT / f'baseline_execution_MPC-H24-XP+QP{suffix}.json').read_text(encoding='utf-8'))
        rows.append({'run': label, 'episodes': len(daily), 'cost_mean': float(daily.cost.mean()), 'carbon_mean': float(daily.carbon.mean()),
                     'task_score_mean': float(daily.task_score.mean()), 'peak_mean_kw': float(daily.peak.mean()),
                     'export_excess_mean_kwh_day': float(daily.grid_export_violation_kwh.mean()),
                     'import_excess_mean_kwh_day': float(daily.grid_import_violation_kwh.mean()),
                     'ramp_slack_mean_kw': float(daily.ramp_slack_kw_mean.mean()), 'mean_daily_maximum_ramp_kw': float(daily.max_ramp_kw.mean()),
                     'phase1_failures': int(execution['phase1_failures']), 'phase2_fallback_to_phase1': int(execution['phase2_fallback_to_phase1']),
                     'final_time_limit_accepted_calls': int(round(execution['means']['mpc_time_limit_incumbents'] * len(daily))),
                     'elapsed_seconds': float(execution['elapsed_seconds'])})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'export_priority_repeat_sensitivity.csv', index=False)
    return result


def main():
    days, _ = cr.make_days()
    trace = load_traces()
    split = split_difficulty(days)
    terminal, by_step = terminal_recovery(trace, days)
    sources, charge, event_timing = constraint_sources(trace)
    priority = export_priority_comparison()
    figure2 = figure2_export_priority_intervals()
    solver = solver_stage_summary()
    repeat = export_priority_repeat_sensitivity()
    files = ['split_difficulty.csv', 'terminal_recovery_daily.csv', 'terminal_recovery_by_controller.csv', 'terminal_recovery_by_step.csv',
             'constraint_source_decomposition.csv', 'charge_bound_diagnostics.csv', 'export_event_timing.csv', 'export_priority_comparison.csv',
             'figure2_export_priority_intervals.csv', 'mpc_export_priority_solver_stage_summary.csv', 'export_priority_repeat_sensitivity.csv']
    result = {'status': 'passed', 'trajectory_rows': len(trace), 'controllers': CONTROLLERS,
              'split_rows': len(split), 'terminal_summary_rows': len(terminal), 'terminal_step_rows': len(by_step),
              'constraint_rows': len(sources), 'charge_bound_rows': len(charge), 'event_timing_rows': len(event_timing),
              'comparison_rows': len(priority), 'figure2_added_rows': len(figure2), 'solver_stage_rows': len(solver), 'repeat_rows': len(repeat),
              'source_sha256': {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in files}}
    (OUT / 'objective_mechanism_analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
