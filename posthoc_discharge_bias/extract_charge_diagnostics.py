from pathlib import Path
import hashlib
import json
import pandas as pd

root = Path(__file__).resolve().parent
source = root.parent / "statistical_updates"
files = {
    "corrected_eight_controller_diagnostics/charge_bound_diagnostics.csv": "charge_bound_diagnostic_row.csv",
    "corrected_eight_controller_diagnostics/constraint_source_decomposition.csv": "constraint_source_rows.csv",
    "corrected_eight_controller_diagnostics/export_event_timing.csv": "export_event_timing_row.csv",
    "charge_bound_normalized.csv": "charge_bound_normalized_row.csv",
}
manifest = {"status": "PASS", "controller": "DischargeBias+QP", "source": "Parallel statistical audit; corrected constraint-source algorithm, not an independent rerun", "files": {}}
for relative, output in files.items():
    path = source / relative
    frame = pd.read_csv(path)
    subset = frame.loc[frame.controller.eq("DischargeBias+QP")]
    assert len(subset) == (3 if output == "constraint_source_rows.csv" else 1)
    subset.to_csv(root / output, index=False)
    manifest["files"][output] = {"source": str(Path("../statistical_updates") / relative), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "output_sha256": hashlib.sha256((root / output).read_bytes()).hexdigest(), "rows": len(subset)}
manifest["source_algorithm_sha256"] = hashlib.sha256((source / "corrected_diagnostic_src/analyze_revision.py").read_bytes()).hexdigest()
(root / "charge_diagnostic_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
