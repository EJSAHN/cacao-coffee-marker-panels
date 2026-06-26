
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

STEPS = [
    "01_inventory_raw_data.py",
    "02_clean_and_harmonize.py",
    "03_identity_structure_qc.py",
    "04_prediction_benchmarks.py",
    "04b_summarize_prediction_results.py",
    "05_marker_panel_simulation.py",
    "05b_panel_vs_full_ridge_digest.py",
    "06_export_analysis_workbook.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cacao and coffee deployment-aware genomic prediction pipeline.")
    parser.add_argument("--project-root", required=True, help="Project folder containing data/raw and receiving outputs.")
    parser.add_argument("--skip", nargs="*", default=[], help="Script names to skip.")
    parser.add_argument("--quick", action="store_true", help="Use quick settings for the most time-consuming modeling scripts.")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    for step in STEPS:
        if step in set(args.skip):
            continue
        cmd = [sys.executable, str(script_dir / step), "--project-root", args.project_root]
        if args.quick and step in {"04_prediction_benchmarks.py", "05_marker_panel_simulation.py"}:
            cmd.append("--quick")
        print("Running:", " ".join(cmd))
        env = os.environ.copy()
        repo_root = script_dir.parent
        env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
