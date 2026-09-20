from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import causal_run as driver

OUT = driver.OUT
BASELINES = ['Rule+QP', 'NetLoadRule+QP', 'MPC-H24+QP', 'MPC-H24-FRR+QP']
CONTROLLERS = BASELINES + list(driver.VARIANTS)
COMPARE = ['cost', 'carbon', 'task_score', 'peak', 'grid_excess_energy_kwh', 'export_excess_energy_kwh', 'import_excess_energy_kwh', 'ramp_slack_kw_mean']
PATHS = ['hard_qp_solution', 'phase2_solution', 'phase1_fallback', 'heuristic_fallback']
KEYS = ['controller', 'seed', 'split_day']
DAYS = np.arange(300, 365)
REPLICATES = 20000


def bootstrap_indices():
    rng = np.random.default_rng(20260908)
    result = {'day_cluster': rng.integers(65, size=(REPLICATES, 65))}
    for length in [7, 14]:
        starts = rng.integers(65, size=(REPLICATES, int(np.ceil(65 / length))))
        result[f'circular_block_{length}'] = ((starts[:, :, None] + np.arange(length)) % 65).reshape(REPLICATES, -1)[:, :65]
    return result


def paired_intervals(frame):
    learned = frame[frame.controller == 'PPO-QP'].set_index(['seed', 'split_day'])[COMPARE].sort_index()
    indices = bootstrap_indices()
    rows = []
    for comparator in BASELINES + ['PPO-QP-no-penalty']:
        other = frame[frame.controller == comparator]
        if comparator in BASELINES:
            right = other.set_index('split_day')[COMPARE].reindex(learned.index.get_level_values('split_day')).to_numpy()
        else:
            right = other.set_index(['seed', 'split_day'])[COMPARE].reindex(learned.index).to_numpy()
        differences = pd.DataFrame(learned.to_numpy() - right, index=learned.index, columns=COMPARE)
        by_day = differences.groupby('split_day').mean().reindex(DAYS).to_numpy()
        seed_effects = differences.groupby('seed').mean()
        for method, index in indices.items():
            sampled = by_day[index].mean(axis=1)
            lower, upper = np.quantile(sampled, [0.025, 0.975], axis=0)
            for j, metric in enumerate(COMPARE):
                rows.append({'controller': 'PPO-QP', 'comparator': comparator, 'metric': metric, 'method': method, 'mean_difference': float(by_day[:, j].mean()), 'ci_lower': float(lower[j]), 'ci_upper': float(upper[j]), 'seed_difference_min': float(seed_effects[metric].min()), 'seed_difference_max': float(seed_effects[metric].max()), 'day_clusters': 65, 'fixed_trained_seeds': 5, 'bootstrap_replicates': REPLICATES, 'confidence_level': 0.95, 'inference_scope': 'conditional_on_five_fixed_trained_seeds_and_inspected_scenario_holdout'})
    return pd.DataFrame(rows)


def main():
    daily_files = [OUT / f'test_seed_{seed}.csv' for seed in driver.SEEDS] + [OUT / f'baseline_{name}.csv' for name in BASELINES]
    trace_files = [OUT / f'trajectory_seed_{seed}.csv' for seed in driver.SEEDS] + [OUT / f'trajectory_{name}.csv' for name in BASELINES]
    missing = [p.name for p in daily_files + trace_files if not p.exists()]
    if missing:
        raise SystemExit('Aggregation requires completed input files: ' + ', '.join(missing))
    checks = []

    def check(name, condition, **detail):
        checks.append({'check': name, 'passed': bool(condition), **detail})

    daily = [pd.read_csv(p) for p in daily_files]
    traces = [pd.read_csv(p) for p in trace_files]
    required_trace = ['controller', 'seed', 'day', 'step', 'p', 'soc', 'grid_import', 'grid_export', 'grid_violation', 'qp_execution_path', 'runtime_ms', 'endpoint_minimum_violation_kw', 'endpoint_infeasible', 'feasible_state_violation', 'extra_excess_energy_kwh_step', 'power_lower_kw', 'power_upper_kw', 'ev_lower_fraction', 'ev_upper_fraction', 'base_load_minus_pv_kw', 'lighting_kw', 'mpc_status', 'mpc_fallback']
    for path, data in zip(daily_files, daily):
        absent = sorted(set(driver.METRICS + KEYS + ['steps', 'grid_violation_steps', 'grid_steps', 'ev_violation', 'terminal_soc_kwh', 'lighting_below_full_steps', 'degradation', 'peak_cost', 'comfort_cost']) - set(data.columns))
        check('daily_schema', not absent, file=path.name, missing=absent)
        check('daily_all_numeric_finite', np.isfinite(data.select_dtypes(include='number')).all().all(), file=path.name)
    for path, trace in zip(trace_files, traces):
        absent = sorted(set(required_trace) - set(trace.columns))
        check('trajectory_schema', not absent, file=path.name, missing=absent)
        numeric = trace.select_dtypes(include='number')
        optional = ['mpc_mip_gap', 'mpc_constraint_violation_max']
        check('trajectory_required_numeric_finite', np.isfinite(numeric.drop(columns=optional, errors='ignore')).all().all(), file=path.name)
        for column in optional:
            if column in numeric:
                values = numeric[column]
                allowed = values.isna() & trace.mpc_fallback.eq(1)
                check('solver_diagnostic_finite_or_explicitly_unavailable_on_fallback', (np.isfinite(values) | allowed).all(), file=path.name, column=column, unavailable_count=int(values.isna().sum()))
    if not all(c['passed'] for c in checks):
        (OUT / 'aggregation_verification.json').write_text(json.dumps({'status': 'failed', 'checks': checks}, indent=2), encoding='utf-8')
        raise SystemExit('Input schema or finite-value verification failed; see aggregation_verification.json')
    frame = pd.concat(daily, ignore_index=True)
    trace = pd.concat(traces, ignore_index=True).rename(columns={'day': 'split_day'})
    trace['import_excess_energy_kwh']=(trace.grid_import-260-.01).clip(lower=0)*.25
    trace['export_excess_energy_kwh']=(trace.grid_export-40-.01).clip(lower=0)*.25
    components=trace.groupby(KEYS)[['import_excess_energy_kwh','export_excess_energy_kwh']].sum()
    frame=frame.set_index(KEYS).join(components).reset_index()
    check('exchange_components_reconcile',np.allclose(frame.import_excess_energy_kwh+frame.export_excess_energy_kwh,frame.grid_excess_energy_kwh,atol=1e-7,rtol=0))
    expected_keys = {(c, seed, day) for c in driver.VARIANTS for seed in driver.SEEDS for day in DAYS} | {(c, 0, day) for c in BASELINES for day in DAYS}
    check('exact_daily_keys_without_baseline_replication', len(frame) == 910 and not frame.duplicated(KEYS).any() and set(frame[KEYS].itertuples(index=False, name=None)) == expected_keys)
    check('daily_source_file_scope', all(len(x) == 130 and set(x.seed) == {seed} and set(x.controller) == set(driver.VARIANTS) for x, seed in zip(daily[:5], driver.SEEDS)) and all(len(x) == 65 and set(x.controller) == {c} and set(x.seed) == {0} for x, c in zip(daily[5:], BASELINES)))
    check('test_only_and_96_steps_per_day', frame['split'].eq('test').all() and frame.steps.eq(96).all())
    check('trajectory_complete_87360_steps', len(trace) == 87360 and not trace.duplicated(KEYS + ['step']).any() and trace.groupby(KEYS).step.apply(lambda s: sorted(s) == list(range(96))).all() and set(trace[KEYS].itertuples(index=False, name=None)) == expected_keys)
    check('separate_policy_baseline_denominators', len(frame[frame.controller.isin(driver.VARIANTS)]) == 650 and len(trace[trace.controller.isin(driver.VARIANTS)]) == 62400 and len(frame[frame.controller.isin(BASELINES)]) == 260 and len(trace[trace.controller.isin(BASELINES)]) == 24960)
    check('full_common_service', frame.ev_completion.eq(1).all() and frame.ev_violation.abs().le(1e-8).all() and np.allclose(frame.lighting_energy_kwh, 472.5, atol=1e-5, rtol=0) and frame.lighting_below_full_steps.eq(0).all() and np.allclose(frame.terminal_soc_error_kwh, 0, atol=1e-5, rtol=0), ev_tolerance_kwh=1e-8, maximum_absolute_ev_residual_kwh=float(frame.ev_violation.abs().max()))
    cfg, dt = driver.CFG, driver.CFG['dt_hours']
    minimum = np.maximum.reduce([np.zeros(len(trace)), trace.base_load_minus_pv_kw + trace.ev_lower_fraction * cfg['loads']['ev_max_kw'] + trace.lighting_kw - trace.power_upper_kw - cfg['loads']['grid_max_kw'] - .01, cfg['loads']['grid_min_kw'] - (trace.base_load_minus_pv_kw + trace.ev_upper_fraction * cfg['loads']['ev_max_kw'] + trace.lighting_kw - trace.power_lower_kw) - .01])
    infeasible = minimum > 1e-9
    feasible_violation = (~infeasible) & (trace.grid_violation.to_numpy() > 1e-9)
    check('independent_endpoint_formula', np.allclose(minimum, trace.endpoint_minimum_violation_kw, atol=1e-7, rtol=0))
    check('independent_mutually_exclusive_endpoint_flags', np.array_equal(infeasible.astype(int), trace.endpoint_infeasible) and np.array_equal(feasible_violation.astype(int), trace.feasible_state_violation) and not np.any(infeasible & feasible_violation))
    check('endpoint_categories_reconcile_actual_violations', np.array_equal(infeasible | feasible_violation, trace.grid_violation.to_numpy() > 1e-12), numeric_positive_below_audit_threshold=int(((trace.grid_violation > 1e-12) & (trace.grid_violation <= 1e-9)).sum()))
    check('extra_excess_energy_independent', np.allclose(np.maximum(0, trace.grid_violation - minimum) * dt, trace.extra_excess_energy_kwh_step, atol=1e-7, rtol=0))
    lo, hi = cfg['bess']['energy_kwh'] * cfg['bess']['soc_min'], cfg['bess']['energy_kwh'] * cfg['bess']['soc_max']
    check('soc_all_steps_and_days', trace.soc.between(lo - 1e-8, hi + 1e-8).all() and frame.soc_violation_rate.eq(0).all())
    check('qp_paths_exhaustive_daily', trace.qp_execution_path.isin(PATHS).all() and np.allclose(frame[PATHS].sum(axis=1), 96, atol=0, rtol=0))
    indexed = frame.set_index(KEYS)
    mismatches = []
    for key, group in trace.groupby(KEYS):
        group = group.sort_values('step'); row = indexed.loc[key]
        comparisons = {'steps': len(group), 'grid_violation_steps': (group.grid_violation > 1e-12).sum(), 'grid_steps': (group.grid_violation > 1e-12).sum(), 'grid_excess_energy_kwh': group.grid_violation.sum() * dt, 'grid_max_exceedance_kw': group.grid_violation.max(), 'peak': group.grid_import.max(), 'terminal_soc_kwh': group.soc.iloc[-1], 'endpoint_infeasible_steps': group.endpoint_infeasible.sum(), 'feasible_state_violation_steps': group.feasible_state_violation.sum(), 'extra_excess_energy_kwh': group.extra_excess_energy_kwh_step.sum(), 'runtime_mean_ms': group.runtime_ms.mean(), 'runtime_p95_ms': group.runtime_ms.quantile(.95), 'mpc_fallbacks': group.mpc_fallback.sum(), 'mpc_time_limit_incumbents': ((group.mpc_status == 1) & (group.mpc_fallback == 0)).sum()}
        comparisons.update({path: (group.qp_execution_path == path).sum() for path in PATHS})
        for metric, actual in comparisons.items():
            if not np.isclose(actual, row[metric], atol=1e-7, rtol=1e-10):
                mismatches.append({'key': [str(x) for x in key], 'metric': metric, 'from_steps': float(actual), 'reported_day': float(row[metric])})
    check('trajectory_to_daily_exact_aggregation', not mismatches, mismatches=mismatches)
    task_expected = frame.cost / 10 + 1.8 * frame.carbon + frame.degradation + frame.peak_cost + frame.comfort_cost + frame.grid_excess_energy_kwh
    check('task_score_excludes_training_correction_penalty_given_zero_soc_ev_errors', np.allclose(task_expected, frame.task_score, atol=1e-6, rtol=1e-10))
    verified = {'status': 'passed' if all(c['passed'] for c in checks) else 'failed', 'checks': checks, 'denominators': {'policy_days': 650, 'baseline_days': 260, 'policy_steps': 62400, 'baseline_steps': 24960, 'unique_scenario_days': 65, 'fixed_trained_seeds': 5}, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in daily_files + trace_files}, 'inference_scope': 'Conditional intervals for five fixed trained seeds on a previously inspected 65-day scenario holdout. Deterministic baselines are evaluated once per day.'}
    (OUT / 'aggregation_verification.json').write_text(json.dumps(verified, indent=2) + '\n', encoding='utf-8')
    if verified['status'] != 'passed':
        raise SystemExit('Statistical/trajectory verification failed; see aggregation_verification.json')
    metrics = list(dict.fromkeys(driver.METRICS + ['export_excess_energy_kwh', 'import_excess_energy_kwh', 'terminal_target_conflict_rate', 'soc_violation_steps', 'grid_violation_steps', 'steps']))
    summary = frame.groupby('controller')[metrics].mean().reindex(CONTROLLERS)
    summary['evaluation_rows'] = frame.groupby('controller').size()
    summary['unique_scenario_days'] = 65
    summary['trained_seed_count'] = [0]*len(BASELINES)+[5]*len(driver.VARIANTS)
    summary['runtime_pooled_p95_ms'] = trace.groupby('controller').runtime_ms.quantile(.95)
    for metric in PATHS + ['grid_violation_steps', 'endpoint_infeasible_steps', 'feasible_state_violation_steps', 'mpc_fallbacks', 'mpc_time_limit_incumbents']:
        summary[metric + '_total'] = frame.groupby('controller')[metric].sum()
    summary = summary.reset_index()
    seeds = frame.groupby(['controller', 'seed'])[metrics].mean().reset_index()
    seeds['record_type'] = np.where(seeds.controller.isin(BASELINES), 'deterministic_baseline_computed_once', 'trained_policy_seed')
    seeds['scenario_days'] = 65
    intervals = paired_intervals(frame)
    for name, table in [('summary.csv', summary), ('seed_means.csv', seeds), ('paired_intervals.csv', intervals)]:
        assert np.isfinite(table.select_dtypes(include='number')).all().all()
        table.to_csv(OUT / name, index=False)
    assert len(intervals) == 120 and intervals.day_clusters.eq(65).all()
    print(json.dumps({'status': 'passed', 'daily_rows': len(frame), 'trajectory_rows': len(trace), 'interval_rows': len(intervals), 'checks': len(checks)}))


if __name__ == '__main__':
    main()
