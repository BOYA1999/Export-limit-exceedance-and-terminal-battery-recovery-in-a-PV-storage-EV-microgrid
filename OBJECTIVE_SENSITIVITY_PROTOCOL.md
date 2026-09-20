# Frozen objective-alignment sensitivity

The new deterministic controller is `MPC-H24-XP+QP`. It is a post-hoc sensitivity on the already inspected 65-day evaluation segment, not a prospective validation.

- Start from `MPC-H24+QP`: identical 24-step causal forecast, battery-feasibility tail, hard 45 kW planned ramps, terminal target, EV and lighting constraints, MILP tolerances, and common QP execution layer.
- Phase 1 minimizes predicted export-exceedance energy, `0.25 * sum(over_export)`, in kWh over the objective horizon.
- Phase 2 constrains predicted export-exceedance energy to the accepted Phase-1 value plus `1e-6 kWh` and minimizes the unchanged operating objective.
- Each MILP phase uses the existing 1 s limit and relative-gap target 0.001. Feasibility acceptance keeps the existing `1e-5` maximum residual rule.
- If Phase 2 has no accepted incumbent, the accepted Phase-1 solution is executed. If Phase 1 has no accepted incumbent, `LegacyRule+QP` supplies the proposal.
- No coefficient, tolerance, controller branch, or day is adjusted after evaluation begins. All solver outcomes and adverse results are retained.

The primary comparison is paired by day against `MPC-H24+QP`. Report export exceedance, task score, cost, carbon, peak import, ramp slack, terminal/service checks, and solver-path counts. A reduction in export exceedance supports objective-preference sensitivity; persistence of the gap supports an additional scheduling effect. Neither outcome establishes external generality.
