import json
import sys
from pathlib import Path

import numpy as np

from mcap_extractor.metrics_calculations import (
    compute_ate_rmse,
    compute_rpe_stats,
    load_tum,
    sync_pair,
)
from mcap_extractor.metrics_definitions import REFERENCE_ALIGNED_STEM, RPE_DELTAS_METERS


def evaluate_pair(ref_name, ref_ts, ref_xyz, est_name, est_ts, est_xyz):
    _, ref_sync, est_sync = sync_pair(ref_ts, ref_xyz, est_ts, est_xyz)

    if len(ref_sync) == 0:
        return {
            "reference": ref_name,
            "estimate": est_name,
            "status": "no_time_overlap",
        }

    ate_rmse_m, _ = compute_ate_rmse(ref_sync, est_sync)

    rpe = {}
    for d in RPE_DELTAS_METERS:
        stats = compute_rpe_stats(ref_sync, est_sync, d)
        if stats is not None:
            rpe[f"{d}m"] = stats

    agg_rpe_rmse = float("nan")
    if rpe:
        rmse_values = [v["rmse_m"] for v in rpe.values()]
        agg_rpe_rmse = float(np.sqrt(np.mean(np.square(rmse_values))))

    return {
        "reference": ref_name,
        "estimate": est_name,
        "synced_samples": int(len(ref_sync)),
        "ate_rmse_m": float(ate_rmse_m),
        "rpe_aggregate_rmse_m": agg_rpe_rmse,
        "rpe": rpe,
        "status": "ok",
    }


def evaluate_run_folder(run_folder):
    run_dir = Path(run_folder)
    aligned_dir = run_dir / "aligned"
    metrics_dir = run_dir / "metrics"

    if not aligned_dir.is_dir():
        print(f"Error: missing aligned directory at {aligned_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(aligned_dir.glob("*_aligned.txt"))
    if not files:
        print(f"Error: no aligned trajectories found in {aligned_dir}", file=sys.stderr)
        sys.exit(1)

    trajectories = {}
    for p in files:
        ts, xyz = load_tum(p)
        trajectories[p.stem] = (ts, xyz)

    ref_name = REFERENCE_ALIGNED_STEM if REFERENCE_ALIGNED_STEM in trajectories else next(iter(trajectories))
    ref_ts, ref_xyz = trajectories[ref_name]

    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "reference": ref_name,
        "deltas_m": RPE_DELTAS_METERS,
        "results": {},
    }

    for name, (ts, xyz) in trajectories.items():
        if name == ref_name:
            continue

        report = evaluate_pair(ref_name, ref_ts, ref_xyz, name, ts, xyz)
        summary["results"][name] = report

        out_file = metrics_dir / f"{name}_vs_{ref_name}.json"
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {out_file}")

    summary_file = metrics_dir / "metrics_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Metrics summary: {summary_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: poetry run eval_metrics <output_run_folder>")
        print("Example: poetry run eval_metrics output/my_bag")
        sys.exit(1)

    evaluate_run_folder(sys.argv[1])


if __name__ == "__main__":
    main()
