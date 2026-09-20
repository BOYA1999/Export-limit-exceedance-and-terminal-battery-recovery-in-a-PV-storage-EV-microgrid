from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

execution = json.loads((ART / "baseline_execution_MPC-H24-XP+QP.json").read_text(encoding="utf-8"))
assert execution["protocol_sha256"] == sha(ROOT / "OBJECTIVE_SENSITIVITY_PROTOCOL.md")
assert execution["source_sha256"] == sha(ROOT / "revision_src" / "run_export_priority_executed.py")
assert execution["mpc_source_sha256"] == sha(ROOT / "revision_src" / "matched_mpc.py")
assert execution["command"] == ["python", "revision_src/run_export_priority_executed.py"]
initial_execution = json.loads((ART / "baseline_execution_MPC-H24-XP+QP_initial.json").read_text(encoding="utf-8"))
assert initial_execution["source_sha256"] == sha(ROOT / "revision_src" / "run_export_priority_executed.py")
assert initial_execution["mpc_source_sha256"] == sha(ROOT / "revision_src" / "matched_mpc_initial.py")
assert initial_execution["command"] == ["python", "revision_src/run_export_priority_executed.py"]

daily = pd.read_csv(ART / "baseline_MPC-H24-XP+QP.csv")
trace = pd.read_csv(ART / "trajectory_MPC-H24-XP+QP.csv")
assert len(daily) == 65 and len(trace) == 6240
assert trace.groupby("day").size().eq(96).all()
service_failure = ((daily.soc_violation_rate > 0) | (daily.terminal_soc_error_kwh.abs() > 1e-5) |
                   (daily.ev_completion < 1) | (daily.lighting_energy_kwh.sub(472.5).abs() > 1e-5))
assert not service_failure.any()
for _, row in daily.iterrows():
    part = trace[trace.day == row.split_day]
    assert abs(part.grid_violation.sum() * 0.25 - row.grid_excess_energy_kwh) < 1e-7
    assert abs(part.grid_import.max() - row.peak) < 1e-7
assert sum(1 for _ in (ART / "mpc_diagnostics_MPC-H24-XP+QP.jsonl").open(encoding="utf-8")) == 6240

analysis = json.loads((ART / "objective_mechanism_analysis.json").read_text(encoding="utf-8"))
assert analysis["status"] == "passed" and analysis["trajectory_rows"] == 93600
assert len(analysis["controllers"]) == 7
for name, digest in analysis["source_sha256"].items():
    assert sha(ART / name) == digest
assert len(pd.read_csv(ART / "split_difficulty.csv")) == 3
assert len(pd.read_csv(ART / "terminal_recovery_by_controller.csv")) == 7
assert len(pd.read_csv(ART / "terminal_recovery_by_step.csv")) == 112
constraint = pd.read_csv(ART / "constraint_source_decomposition.csv")
assert len(constraint) == 21
assert constraint.subset_check_passed.all() and constraint.monotonic_excess_check_passed.all()
assert constraint.new_endpoint_infeasible_steps.eq(0).all()
ppo_ramp = constraint[(constraint.controller == "PPO-QP") & (constraint.one_step_counterfactual == "remove_ramp_interval")].iloc[0]
assert ppo_ramp.resolved_export_infeasible_steps == 107 and abs(ppo_ramp.mean_daily_change_minimum_excess_kwh + 0.7161254) < 1e-6
assert len(pd.read_csv(ART / "charge_bound_diagnostics.csv")) == 7
assert len(pd.read_csv(ART / "export_event_timing.csv")) == 7
assert len(pd.read_csv(ART / "export_priority_comparison.csv")) == 8
assert len(pd.read_csv(ART / "figure2_export_priority_intervals.csv")) == 3
assert len(pd.read_csv(ART / "mpc_export_priority_solver_stage_summary.csv")) == 6
assert len(pd.read_csv(ART / "export_priority_repeat_sensitivity.csv")) == 2

timing = json.loads((ART / "timing_xp_summary.json").read_text(encoding="utf-8"))
assert timing["status"] == "passed" and timing["summary"]["calls"] == 125
assert timing["summary"]["mpc_fallback_calls"] == 0
assert len(pd.read_csv(ART / "timing_xp_rows.csv")) == 125
assert len(pd.read_csv(ART / "timing_xp_warmup_rows.csv")) == 25

REPAIR = ROOT / "diagnostic_repair"
POSTHOC = ROOT / "posthoc_discharge_bias"

repair_verification = json.loads((REPAIR / "diagnostic_repair_verification.json").read_text(encoding="utf-8"))
assert repair_verification["verified_steps"] == 93600
assert repair_verification["legacy_branch_mismatch_steps"] == 48
assert repair_verification["mismatch_export_events"] == 0
assert repair_verification["changed_values_over_1e-10"] == 3
assert repair_verification["all_fixed_relaxed_intervals_are_supersets_with_recorded_tolerance"]
assert sha(REPAIR / "corrected_diagnostic_src" / "analyze_revision.py") == sha(ROOT / "revision_src" / "analyze_revision.py")
assert analysis["analysis_source_sha256"] == sha(ROOT / "revision_src" / "analyze_revision.py")
for name in ["charge_bound_diagnostics.csv", "constraint_source_decomposition.csv", "export_event_timing.csv"]:
    assert sha(REPAIR / "corrected_diagnostic_artifacts" / name) == sha(ART / name)
assert sha(REPAIR / "original_diagnostic_snapshot" / "analyze_revision.py") != sha(ROOT / "revision_src" / "analyze_revision.py")

repair_expected = {}
for line in (REPAIR / "manifest.sha256").read_text(encoding="utf-8").splitlines():
    digest, name = line.split("  ", 1)
    assert name not in repair_expected and sha(REPAIR / name) == digest
    repair_expected[name] = digest
repair_actual = {p.relative_to(REPAIR).as_posix() for p in REPAIR.rglob("*") if p.is_file()}
assert repair_actual == set(repair_expected) | {"manifest.sha256"}

eight_verification = json.loads((REPAIR / "eight_controller_verification.json").read_text(encoding="utf-8"))
assert eight_verification["controllers"] == 8 and eight_verification["steps"] == 99840
assert eight_verification["episodes"] == 1040 and eight_verification["superset_checks_all_steps"]
eight_dir = REPAIR / "corrected_eight_controller_diagnostics"
assert len(pd.read_csv(eight_dir / "charge_bound_diagnostics.csv")) == 8
assert len(pd.read_csv(eight_dir / "constraint_source_decomposition.csv")) == 24
assert len(pd.read_csv(eight_dir / "export_event_timing.csv")) == 8

posthoc_verification = json.loads((POSTHOC / "verification.json").read_text(encoding="utf-8"))
assert posthoc_verification["status"] == "PASS"
assert posthoc_verification["episodes"] == 65 and posthoc_verification["steps"] == 6240
assert posthoc_verification["service_failures"] == 0
assert posthoc_verification["protocol_sha256"] == sha(POSTHOC / "protocol.json")
assert posthoc_verification["runner_sha256"] == sha(POSTHOC / "run_baseline.py")
for name, digest in posthoc_verification["outputs"].items():
    assert sha(POSTHOC / name) == digest
posthoc_protocol = json.loads((POSTHOC / "protocol.json").read_text(encoding="utf-8"))
for name, digest in posthoc_protocol["source_files"].items():
    assert sha(POSTHOC / "source_snapshot" / name) == digest
posthoc_daily = pd.read_csv(POSTHOC / "daily.csv")
posthoc_trace = pd.read_csv(POSTHOC / "trajectory.csv")
assert len(posthoc_daily) == 65 and len(posthoc_trace) == 6240
assert posthoc_trace.groupby("day").size().eq(96).all()
posthoc_failures = ((posthoc_daily.soc_violation_rate > 0) |
                    (posthoc_daily.terminal_soc_error_kwh.abs() > 1e-5) |
                    (posthoc_daily.ev_completion < 1) |
                    (posthoc_daily.lighting_energy_kwh.sub(472.5).abs() > 1e-5))
assert not posthoc_failures.any()
posthoc_summary = json.loads((POSTHOC / "summary.json").read_text(encoding="utf-8"))
assert posthoc_summary["controller"] == "DischargeBias+QP"
assert posthoc_summary["forced_final_window_days"] == 65
assert abs(posthoc_summary["grid_excess_energy_kwh"] - 40.80846371781254) < 1e-10
assert posthoc_summary["execution_paths"] == {"hard_qp_solution": 6028, "phase2_solution": 212}
reporting = json.loads((POSTHOC / "reporting_verification.json").read_text(encoding="utf-8"))
assert reporting["status"] == "PASS" and reporting["interval_rows"] == 48
for name, digest in reporting["outputs"].items():
    assert sha(POSTHOC / name) == digest
assert len(pd.read_csv(POSTHOC / "paired_intervals_all_methods.csv")) == 48
assert len(pd.read_csv(POSTHOC / "timing_rows.csv")) == 125

result = {"status": "passed", "episodes": 65, "steps": 6240, "combined_controller_days": 1040,
          "combined_evaluation_steps": 99840, "service_failures": int(service_failure.sum()),
          "diagnostic_records": 6240, "timing_calls": 125, "figure_files": 0,
          "monotonic_counterfactual_rows": len(constraint), "retained_export_priority_runs": 2,
          "diagnostic_repair_mismatch_steps": 48, "diagnostic_repair_export_events": 0,
          "posthoc_controller": "DischargeBias+QP", "posthoc_service_failures": int(posthoc_failures.sum()),
          "posthoc_export_excess_kwh_per_day": posthoc_summary["grid_excess_energy_kwh"],
          "repair_manifest_files": len(repair_actual)}
print(json.dumps(result, indent=2))
