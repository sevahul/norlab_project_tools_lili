import json
import math
import sys
from pathlib import Path

import numpy as np

# Relative distance windows (meters) used for RPE.
RPE_DELTAS_METERS = [1, 2, 5, 10, 20, 50, 100]

# Prefer this aligned trajectory as ground truth.
REFERENCE_ALIGNED_STEM = "theodolite_trajectory_aligned"


def load_tum(filepath):
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ts = data[:, 0]
    xyz = data[:, 1:4]

    order = np.argsort(ts)
    return ts[order], xyz[order]


def interpolate_xyz(target_ts, source_ts, source_xyz):
    x = np.interp(target_ts, source_ts, source_xyz[:, 0], left=np.nan, right=np.nan)
    y = np.interp(target_ts, source_ts, source_xyz[:, 1], left=np.nan, right=np.nan)
    z = np.interp(target_ts, source_ts, source_xyz[:, 2], left=np.nan, right=np.nan)
    return np.column_stack((x, y, z))


def sync_pair(ref_ts, ref_xyz, est_ts, est_xyz):
    est_interp = interpolate_xyz(ref_ts, est_ts, est_xyz)
    valid = ~np.isnan(est_interp[:, 0])
    return ref_ts[valid], ref_xyz[valid], est_interp[valid]


def rmse(values):
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def compute_ate_rmse(ref_xyz, est_xyz):
    errors = np.linalg.norm(est_xyz - ref_xyz, axis=1)
    return rmse(errors), errors


def cumulative_distance(positions):
    if len(positions) < 2:
        return np.zeros((len(positions),), dtype=float)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))


def compute_rpe_stats(ref_xyz, est_xyz, delta_m):
    if len(ref_xyz) < 2:
        return None

    s_ref = cumulative_distance(ref_xyz)
    errors = []

    for i in range(len(ref_xyz) - 1):
        target_s = s_ref[i] + delta_m
        j = int(np.searchsorted(s_ref, target_s, side="left"))
        if j >= len(ref_xyz):
            continue

        d_ref = ref_xyz[j] - ref_xyz[i]
        d_est = est_xyz[j] - est_xyz[i]
        errors.append(float(np.linalg.norm(d_est - d_ref)))

    if not errors:
        return None

    e = np.array(errors)
    return {
        "samples": int(e.size),
        "rmse_m": float(np.sqrt(np.mean(np.square(e)))),
        "std_m": float(np.std(e)),
        "min_m": float(np.min(e)),
        "max_m": float(np.max(e)),
        "rmse_percent": float((np.sqrt(np.mean(np.square(e))) / delta_m) * 100.0),
    }


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
