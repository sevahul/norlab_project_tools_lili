import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REFERENCE_ALIGNED_STEM = "theodolite_trajectory_aligned"
HIGHLIGHT_DELTA_METERS = 10.0

TRAJECTORY_COLORS = {
    "t265_trajectory_aligned": "tab:blue",
    "legged_trajectory_aligned": "tab:orange",
}

FALLBACK_COLORS = ["tab:green", "tab:red", "tab:purple", "tab:brown"]

# Heuristics to skip reference segments that likely cross total-station loss/reacquisition.
REF_GAP_FACTOR = 5.0
REF_MIN_GAP_SECONDS = 0.5
REF_STEP_JUMP_FACTOR = 8.0
REF_MIN_STEP_JUMP_METERS = 1.0
ACTIVE_REF_START_DISTANCE_M = 0.5
ACTIVE_REF_END_DISTANCE_M = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize aligned trajectories and RPE metrics from saved outputs."
    )
    parser.add_argument(
        "run_folder",
        help="Path to run folder, for example: output/my_bag",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
    )
    parser.add_argument(
        "--highlight-delta",
        type=float,
        default=HIGHLIGHT_DELTA_METERS,
        help="Delta in meters used for the spatial RPE highlight plot.",
    )
    return parser.parse_args()


def load_tum_trajectory(file_path):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ts = data[:, 0]
    xyz = data[:, 1:4]
    order = np.argsort(ts)
    return ts[order], xyz[order]


def load_aligned_trajectories(aligned_dir):
    files = sorted(aligned_dir.glob("*_aligned.txt"))
    trajectories = {}
    for p in files:
        ts, xyz = load_tum_trajectory(p)
        trajectories[p.stem] = {"ts": ts, "xyz": xyz}
    return trajectories


def pick_reference_name(trajectories):
    if REFERENCE_ALIGNED_STEM in trajectories:
        return REFERENCE_ALIGNED_STEM
    return next(iter(trajectories))


def sanitize_name(name):
    return name.replace("/", "_")


def delta_token(delta_m):
    return f"{delta_m:g}".replace(".", "p")


def color_for_name(name):
    if name in TRAJECTORY_COLORS:
        return TRAJECTORY_COLORS[name]
    idx = abs(hash(name)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def save_all_aligned_plot(trajectories, reference_name, reports, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))

    ref_start = trajectories[reference_name]["xyz"][0, :2]

    for name, traj in trajectories.items():
        xyz = traj["xyz"]
        xy = xyz[:, :2] - ref_start
        if name == reference_name:
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                linewidth=2,
                linestyle="--",
                color="black",
                label=name,
            )
        else:
            ax.plot(xy[:, 0], xy[:, 1], color=color_for_name(name), alpha=0.9, label=name)

    ax.plot(0, 0, "ro", markersize=7, label="Reference start")
    ax.set_title("All Aligned Trajectories (XY)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()

    ref_len = trajectory_length(trajectories[reference_name]["xyz"])
    lines = [f"Ref length: {ref_len:.2f} m"]
    for name in sorted(trajectories.keys()):
        if name == reference_name:
            continue
        est_len = trajectory_length(trajectories[name]["xyz"])
        ate_val = reports.get(name, {}).get("ate_rmse_m")
        if ate_val is None:
            lines.append(f"{name}: length={est_len:.2f} m")
        else:
            lines.append(f"{name}: ATE={ate_val:.3f} m, length={est_len:.2f} m")
    add_info_box(ax, "\n".join(lines))

    fig.tight_layout()

    out_path = output_dir / "all_aligned_xy.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_pairwise_plot(
    reference_name,
    reference_xyz,
    estimate_name,
    estimate_xyz,
    ate_rmse_m,
    reference_len_m,
    estimate_len_m,
    global_ratio,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    ref_start = reference_xyz[0, :2]
    ref_xy = reference_xyz[:, :2] - ref_start
    est_xy = estimate_xyz[:, :2] - ref_start

    ax.plot(ref_xy[:, 0], ref_xy[:, 1], linewidth=2, linestyle="--", color="black", label=reference_name)
    ax.plot(est_xy[:, 0], est_xy[:, 1], linewidth=1.5, color=color_for_name(estimate_name), alpha=0.95, label=estimate_name)

    ax.plot(0, 0, "ro", markersize=7, label="Reference start")
    ax.set_title(f"Trajectory vs Ground Truth: {estimate_name}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    add_info_box(
        ax,
        (
            f"ATE: {ate_rmse_m:.3f} m\n"
            f"{reference_name} length: {reference_len_m:.2f} m\n"
            f"{estimate_name} length: {estimate_len_m:.2f} m\n"
            f"Distance ratio (est/ref): {global_ratio:.3f}"
        ),
    )
    fig.tight_layout()

    out_path = output_dir / f"pair_{sanitize_name(estimate_name)}_vs_{sanitize_name(reference_name)}.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def interpolate_xyz(target_ts, source_ts, source_xyz):
    x = np.interp(target_ts, source_ts, source_xyz[:, 0], left=np.nan, right=np.nan)
    y = np.interp(target_ts, source_ts, source_xyz[:, 1], left=np.nan, right=np.nan)
    z = np.interp(target_ts, source_ts, source_xyz[:, 2], left=np.nan, right=np.nan)
    return np.column_stack((x, y, z))


def sync_pair(reference_traj, estimate_traj):
    ref_ts = reference_traj["ts"]
    ref_xyz = reference_traj["xyz"]
    est_ts = estimate_traj["ts"]
    est_xyz = estimate_traj["xyz"]

    est_interp = interpolate_xyz(ref_ts, est_ts, est_xyz)
    valid = ~np.isnan(est_interp[:, 0])
    return ref_ts[valid], ref_xyz[valid], est_interp[valid]


def build_reference_valid_edges(ref_ts, ref_xyz):
    if len(ref_ts) < 2:
        return np.array([], dtype=bool)

    dt = np.diff(ref_ts)
    step = np.linalg.norm(np.diff(ref_xyz, axis=0), axis=1)

    valid = np.ones_like(dt, dtype=bool)
    valid &= np.isfinite(dt) & np.isfinite(step) & (dt > 0)

    pos_dt = dt[dt > 0]
    if pos_dt.size > 0:
        dt_thr = max(REF_MIN_GAP_SECONDS, float(np.median(pos_dt)) * REF_GAP_FACTOR)
        valid &= dt <= dt_thr

    pos_step = step[step > 0]
    if pos_step.size > 0:
        step_thr = max(REF_MIN_STEP_JUMP_METERS, float(np.median(pos_step)) * REF_STEP_JUMP_FACTOR)
        valid &= step <= step_thr

    return valid


def keep_active_window(ref_ts, ref_xyz, est_xyz):
    """Trim leading/trailing sections where reference has not started or likely got lost."""
    if len(ref_xyz) < 2:
        return ref_ts, ref_xyz, est_xyz

    s_ref = cumulative_distance(ref_xyz)
    total_len = float(s_ref[-1])
    if total_len <= (ACTIVE_REF_START_DISTANCE_M + ACTIVE_REF_END_DISTANCE_M):
        return ref_ts, ref_xyz, est_xyz

    i0 = int(np.searchsorted(s_ref, ACTIVE_REF_START_DISTANCE_M, side="left"))
    end_target = total_len - ACTIVE_REF_END_DISTANCE_M
    i1 = int(np.searchsorted(s_ref, end_target, side="right") - 1)

    if i1 <= i0:
        return ref_ts, ref_xyz, est_xyz

    sl = slice(i0, i1 + 1)
    return ref_ts[sl], ref_xyz[sl], est_xyz[sl]


def build_invalid_edge_prefix(valid_edges):
    if valid_edges.size == 0:
        return None
    invalid = (~valid_edges).astype(np.int32)
    return np.concatenate(([0], np.cumsum(invalid)))


def segment_crosses_invalid(invalid_prefix, i, j):
    if invalid_prefix is None:
        return False
    return (invalid_prefix[j] - invalid_prefix[i]) > 0


def cumulative_distance(positions):
    if len(positions) < 2:
        return np.zeros((len(positions),), dtype=float)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))


def trajectory_length(positions):
    if len(positions) < 2:
        return 0.0
    return float(cumulative_distance(positions)[-1])


def add_info_box(ax, text):
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.6", "boxstyle": "round,pad=0.3"},
    )


def compute_local_rpe_samples(ref_ts, ref_xyz, est_xyz, delta_m, invalid_prefix=None):
    if len(ref_xyz) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    s_ref = cumulative_distance(ref_xyz)
    starts_xy = []
    errors_m = []
    target_indices = []
    start_times = []

    for i in range(len(ref_xyz) - 1):
        target_s = s_ref[i] + delta_m
        j = int(np.searchsorted(s_ref, target_s, side="left"))
        if j >= len(ref_xyz):
            continue

        if segment_crosses_invalid(invalid_prefix, i, j):
            continue

        d_ref = ref_xyz[j] - ref_xyz[i]
        d_est = est_xyz[j] - est_xyz[i]
        err = float(np.linalg.norm(d_est - d_ref))

        starts_xy.append(est_xyz[i, :2])
        errors_m.append(err)
        target_indices.append(j)
        start_times.append(ref_ts[i])

    if not errors_m:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.array(starts_xy),
        np.array(errors_m),
        np.array(target_indices),
        np.array(start_times),
    )


def compute_local_distance_ratio_samples(ref_ts, ref_xyz, est_xyz, delta_m, invalid_prefix=None):
    if len(ref_xyz) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    s_ref = cumulative_distance(ref_xyz)
    s_est = cumulative_distance(est_xyz)

    starts_xy = []
    ratios = []
    drift_pct = []
    start_times = []

    for i in range(len(ref_xyz) - 1):
        target_s = s_ref[i] + delta_m
        j = int(np.searchsorted(s_ref, target_s, side="left"))
        if j >= len(ref_xyz):
            continue

        if segment_crosses_invalid(invalid_prefix, i, j):
            continue

        ref_len = float(s_ref[j] - s_ref[i])
        if ref_len <= 1e-9:
            continue

        est_len = float(s_est[j] - s_est[i])
        ratio = est_len / ref_len
        starts_xy.append(est_xyz[i, :2])
        ratios.append(ratio)
        drift_pct.append((ratio - 1.0) * 100.0)
        start_times.append(ref_ts[i])

    if not ratios:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return np.array(starts_xy), np.array(ratios), np.array(drift_pct), np.array(start_times)


def save_rpe_timeseries_overlay(rpe_series, delta_m, output_dir):
    if len(rpe_series) < 1:
        return None, None

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for name in sorted(rpe_series.keys()):
        times = rpe_series[name]["times"]
        errors_m = rpe_series[name]["errors_m"]
        if times.size == 0 or errors_m.size == 0:
            continue

        t_rel = times - times[0]
        ax.plot(t_rel, errors_m, linewidth=1.4, color=color_for_name(name), alpha=0.95, label=name)

    ax.set_xlabel("Time from start [s]")
    ax.set_ylabel("Local RPE [m]")
    ax.set_title(f"RPE Time Series Overlay (delta={delta_m:g} m)")
    ax.grid(True)
    ax.legend()

    lines = []
    for name in sorted(rpe_series.keys()):
        ate_rmse_m = rpe_series[name].get("ate_rmse_m")
        est_len_m = rpe_series[name].get("estimate_len_m")
        global_ratio = rpe_series[name].get("global_distance_ratio")
        local_ratio = rpe_series[name].get("local_distance_ratio_median")
        if ate_rmse_m is None or est_len_m is None:
            continue
        lines.append(
            f"{name}: ATE={ate_rmse_m:.3f} m, len={est_len_m:.2f} m, "
            f"ratio={global_ratio:.3f}, local~{local_ratio:.3f}"
        )
    if lines:
        add_info_box(ax, "\n".join(lines))

    fig.tight_layout()

    out_path = output_dir / f"rpe_timeseries_overlay_delta{delta_token(delta_m)}m.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_rpe_boxplot_side_by_side(rpe_series, delta_m, output_dir):
    names = [name for name in sorted(rpe_series.keys()) if rpe_series[name].get("errors_m", np.array([])).size > 0]
    if len(names) < 1:
        return None, None

    data = [rpe_series[name]["errors_m"] for name in names]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    try:
        box = ax.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)
    except TypeError:
        box = ax.boxplot(data, labels=names, patch_artist=True, showfliers=False)

    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(color_for_name(names[idx]))
        patch.set_alpha(0.55)

    ax.set_ylabel("Local RPE [m]")
    ax.set_title(f"RPE Boxplots Side-by-Side (delta={delta_m:g} m)")
    ax.grid(True, axis="y", alpha=0.35)

    lines = []
    for name in names:
        arr = rpe_series[name]["errors_m"]
        ate_rmse_m = rpe_series[name].get("ate_rmse_m")
        global_ratio = rpe_series[name].get("global_distance_ratio")
        local_ratio = rpe_series[name].get("local_distance_ratio_median")
        med = float(np.median(arr)) if arr.size > 0 else float("nan")
        if ate_rmse_m is None:
            lines.append(f"{name}: median={med:.3f} m")
        else:
            lines.append(
                f"{name}: ATE={ate_rmse_m:.3f} m, median={med:.3f} m, "
                f"ratio={global_ratio:.3f}, local~{local_ratio:.3f}"
            )
    add_info_box(ax, "\n".join(lines))

    fig.tight_layout()

    out_path = output_dir / f"rpe_boxplot_side_by_side_delta{delta_token(delta_m)}m.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_spatial_rpe_plot(
    reference_name,
    reference_xyz,
    estimate_name,
    estimate_xyz,
    starts_xy,
    errors_m,
    delta_m,
    ate_rmse_m,
    reference_len_m,
    estimate_len_m,
    global_ratio,
    local_ratio_median,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    ref_start = reference_xyz[0, :2]
    ref_xy = reference_xyz[:, :2] - ref_start
    est_xy = estimate_xyz[:, :2] - ref_start
    starts_xy_centered = starts_xy - ref_start

    ax.plot(ref_xy[:, 0], ref_xy[:, 1], linewidth=2, linestyle="--", color="black", alpha=0.75, label=reference_name)
    ax.plot(est_xy[:, 0], est_xy[:, 1], linewidth=1.5, color=color_for_name(estimate_name), alpha=0.55, label=estimate_name)

    scatter = ax.scatter(
        starts_xy_centered[:, 0],
        starts_xy_centered[:, 1],
        c=errors_m,
        cmap="inferno",
        s=22,
        edgecolors="none",
        label=f"RPE start points (delta={delta_m:g} m)",
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Local RPE [m]")

    ax.plot(0, 0, "ro", markersize=7, label="Reference start")
    ax.set_title(f"Spatial RPE Map ({delta_m:g} m): {estimate_name}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    add_info_box(
        ax,
        (
            f"ATE: {ate_rmse_m:.3f} m\n"
            f"{reference_name} length: {reference_len_m:.2f} m\n"
            f"{estimate_name} length: {estimate_len_m:.2f} m\n"
            f"Distance ratio: {global_ratio:.3f}, local~{local_ratio_median:.3f}"
        ),
    )
    fig.tight_layout()

    out_path = output_dir / f"rpe_map_delta{delta_token(delta_m)}m_{sanitize_name(estimate_name)}.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_spatial_distance_ratio_plot(
    reference_name,
    reference_xyz,
    estimate_name,
    estimate_xyz,
    starts_xy,
    drift_pct,
    delta_m,
    global_ratio,
    local_ratio_median,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    ref_start = reference_xyz[0, :2]
    ref_xy = reference_xyz[:, :2] - ref_start
    est_xy = estimate_xyz[:, :2] - ref_start
    starts_xy_centered = starts_xy - ref_start

    ax.plot(ref_xy[:, 0], ref_xy[:, 1], linewidth=2, linestyle="--", color="black", alpha=0.75, label=reference_name)
    ax.plot(est_xy[:, 0], est_xy[:, 1], linewidth=1.5, color=color_for_name(estimate_name), alpha=0.55, label=estimate_name)

    vmax = float(max(1.0, np.percentile(np.abs(drift_pct), 95)))
    scatter = ax.scatter(
        starts_xy_centered[:, 0],
        starts_xy_centered[:, 1],
        c=drift_pct,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        s=22,
        edgecolors="none",
        label=f"Distance drift (delta={delta_m:g} m)",
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Local distance drift [%]")

    ax.plot(0, 0, "ro", markersize=7, label="Reference start")
    ax.set_title(f"Spatial Distance-Ratio Drift Map ({delta_m:g} m): {estimate_name}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    add_info_box(
        ax,
        (
            f"Global ratio (est/ref): {global_ratio:.3f}\n"
            f"Global drift: {(global_ratio - 1.0) * 100.0:+.2f}%\n"
            f"Local median ratio: {local_ratio_median:.3f}"
        ),
    )
    fig.tight_layout()

    out_path = output_dir / f"distance_ratio_map_delta{delta_token(delta_m)}m_{sanitize_name(estimate_name)}.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_distance_ratio_timeseries_overlay(distance_series, delta_m, output_dir):
    if len(distance_series) < 1:
        return None, None

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for name in sorted(distance_series.keys()):
        times = distance_series[name]["times"]
        ratios = distance_series[name]["ratios"]
        if times.size == 0 or ratios.size == 0:
            continue
        t_rel = times - times[0]
        ax.plot(t_rel, ratios, linewidth=1.4, color=color_for_name(name), alpha=0.95, label=name)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time from start [s]")
    ax.set_ylabel("Segment distance ratio [-]")
    ax.set_title(f"Distance-Ratio Time Series Overlay (delta={delta_m:g} m)")
    ax.grid(True)
    ax.legend()

    lines = []
    for name in sorted(distance_series.keys()):
        g = distance_series[name].get("global_ratio")
        m = distance_series[name].get("median_ratio")
        if g is None or m is None:
            continue
        lines.append(f"{name}: global={g:.3f}, local~{m:.3f}")
    if lines:
        add_info_box(ax, "\n".join(lines))

    fig.tight_layout()

    out_path = output_dir / f"distance_ratio_timeseries_overlay_delta{delta_token(delta_m)}m.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_distance_ratio_boxplot_side_by_side(distance_series, delta_m, output_dir):
    names = [name for name in sorted(distance_series.keys()) if distance_series[name].get("ratios", np.array([])).size > 0]
    if len(names) < 1:
        return None, None

    data = [distance_series[name]["ratios"] for name in names]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    try:
        box = ax.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)
    except TypeError:
        box = ax.boxplot(data, labels=names, patch_artist=True, showfliers=False)

    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(color_for_name(names[idx]))
        patch.set_alpha(0.55)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Segment distance ratio [-]")
    ax.set_title(f"Distance-Ratio Boxplots Side-by-Side (delta={delta_m:g} m)")
    ax.grid(True, axis="y", alpha=0.35)

    lines = []
    for name in names:
        arr = distance_series[name]["ratios"]
        med = float(np.median(arr)) if arr.size > 0 else float("nan")
        g = distance_series[name].get("global_ratio")
        lines.append(f"{name}: global={g:.3f}, median={med:.3f}")
    add_info_box(ax, "\n".join(lines))

    fig.tight_layout()
    out_path = output_dir / f"distance_ratio_boxplot_side_by_side_delta{delta_token(delta_m)}m.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def parse_delta_key(delta_key):
    if not delta_key.endswith("m"):
        return None
    try:
        return float(delta_key[:-1])
    except ValueError:
        return None


def save_rpe_plot(
    pair_metrics,
    estimate_name,
    reference_name,
    reference_len_m,
    estimate_len_m,
    local_ratio_median,
    output_dir,
):
    rpe_dict = pair_metrics.get("rpe", {})
    rows = []
    for delta_key, stats in rpe_dict.items():
        delta_value = parse_delta_key(delta_key)
        if delta_value is None:
            continue
        rows.append((delta_value, stats))

    if not rows:
        return None, None

    rows.sort(key=lambda x: x[0])

    deltas = np.array([x[0] for x in rows])
    rmse_m = np.array([x[1].get("rmse_m", np.nan) for x in rows])
    rmse_pct = np.array([x[1].get("rmse_percent", np.nan) for x in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(deltas, rmse_m, marker="o", color=color_for_name(estimate_name))
    ax1.set_ylabel("RPE RMSE [m]")
    ax1.set_title(f"RPE by Delta: {estimate_name}")
    ax1.grid(True)

    ax2.plot(deltas, rmse_pct, marker="o", color=color_for_name(estimate_name))
    ax2.set_xlabel("Delta [m]")
    ax2.set_ylabel("RPE RMSE [%]")
    ax2.grid(True)

    ate_rmse_m = pair_metrics.get("ate_rmse_m")
    global_ratio = estimate_len_m / reference_len_m if reference_len_m > 0 else float("nan")
    if ate_rmse_m is not None:
        add_info_box(
            ax1,
            (
                f"ATE: {ate_rmse_m:.3f} m\n"
                f"{reference_name} length: {reference_len_m:.2f} m\n"
                f"{estimate_name} length: {estimate_len_m:.2f} m\n"
                f"Distance ratio (est/ref): {global_ratio:.3f}, local~{local_ratio_median:.3f}"
            ),
        )

    fig.tight_layout()

    out_path = output_dir / f"rpe_{sanitize_name(estimate_name)}.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def load_metrics_reports(metrics_dir):
    reports = {}
    for p in sorted(metrics_dir.glob("*_vs_*.json")):
        with open(p, "r") as f:
            data = json.load(f)
        estimate = data.get("estimate")
        if estimate:
            reports[estimate] = data
    return reports


def run(run_folder, show_figures=False, highlight_delta=HIGHLIGHT_DELTA_METERS):
    run_dir = Path(run_folder)
    aligned_dir = run_dir / "aligned"
    metrics_dir = run_dir / "metrics"
    out_dir = run_dir / "plots" / "metrics_viz"

    if not run_dir.is_dir():
        print(f"Error: run folder not found: {run_dir}", file=sys.stderr)
        sys.exit(1)
    if not aligned_dir.is_dir():
        print(f"Error: aligned folder not found: {aligned_dir}", file=sys.stderr)
        sys.exit(1)
    if not metrics_dir.is_dir():
        print(f"Error: metrics folder not found: {metrics_dir}", file=sys.stderr)
        sys.exit(1)

    trajectories = load_aligned_trajectories(aligned_dir)
    if not trajectories:
        print(f"Error: no aligned trajectories found in {aligned_dir}", file=sys.stderr)
        sys.exit(1)

    reference_name = pick_reference_name(trajectories)
    reference_traj = trajectories[reference_name]
    reference_xyz = reference_traj["xyz"]

    reports = load_metrics_reports(metrics_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    rpe_series = {}
    distance_series = {}

    fig, path = save_all_aligned_plot(trajectories, reference_name, reports, out_dir)
    figures.append(fig)
    print(f"Saved: {path}")

    reference_len_m = trajectory_length(reference_xyz)

    for name, traj in trajectories.items():
        if name == reference_name:
            continue

        xyz = traj["xyz"]
        estimate_len_m = trajectory_length(xyz)
        global_ratio = estimate_len_m / reference_len_m if reference_len_m > 0 else float("nan")
        report = reports.get(name)
        ate_rmse_m = float(report.get("ate_rmse_m")) if report and report.get("ate_rmse_m") is not None else float("nan")

        fig_pair, path_pair = save_pairwise_plot(
            reference_name,
            reference_xyz,
            name,
            xyz,
            ate_rmse_m,
            reference_len_m,
            estimate_len_m,
            global_ratio,
            out_dir,
        )
        figures.append(fig_pair)
        print(f"Saved: {path_pair}")

        if report is None:
            print(f"Warning: no metrics report found for {name}; skipping RPE plot.")
            continue

        ref_ts_sync, ref_sync, est_sync = sync_pair(reference_traj, traj)
        ref_ts_sync, ref_sync, est_sync = keep_active_window(ref_ts_sync, ref_sync, est_sync)
        valid_edges = build_reference_valid_edges(ref_ts_sync, ref_sync)
        invalid_prefix = build_invalid_edge_prefix(valid_edges)

        starts_xy, errors_m, _, start_times = compute_local_rpe_samples(
            ref_ts_sync,
            ref_sync,
            est_sync,
            highlight_delta,
            invalid_prefix=invalid_prefix,
        )
        if errors_m.size == 0:
            print(f"Warning: no local RPE samples found for {name} at delta={highlight_delta:g}m.")
            continue

        starts_xy_ratio, ratios, drift_pct, ratio_times = compute_local_distance_ratio_samples(
            ref_ts_sync,
            ref_sync,
            est_sync,
            highlight_delta,
            invalid_prefix=invalid_prefix,
        )
        if ratios.size == 0:
            print(f"Warning: no local distance-ratio samples found for {name} at delta={highlight_delta:g}m.")
            continue

        local_ratio_median = float(np.median(ratios))

        fig_rpe, path_rpe = save_rpe_plot(
            report,
            name,
            reference_name,
            reference_len_m,
            estimate_len_m,
            local_ratio_median,
            out_dir,
        )
        if fig_rpe is None:
            print(f"Warning: metrics report for {name} has no valid RPE entries.")
            continue
        figures.append(fig_rpe)
        print(f"Saved: {path_rpe}")

        rpe_series[name] = {
            "times": start_times,
            "errors_m": errors_m,
            "ate_rmse_m": ate_rmse_m,
            "estimate_len_m": estimate_len_m,
            "global_distance_ratio": global_ratio,
            "local_distance_ratio_median": local_ratio_median,
        }

        distance_series[name] = {
            "times": ratio_times,
            "ratios": ratios,
            "drift_pct": drift_pct,
            "global_ratio": global_ratio,
            "median_ratio": local_ratio_median,
        }

        fig_map, path_map = save_spatial_rpe_plot(
            reference_name,
            ref_sync,
            name,
            est_sync,
            starts_xy,
            errors_m,
            highlight_delta,
            ate_rmse_m,
            reference_len_m,
            estimate_len_m,
            global_ratio,
            local_ratio_median,
            out_dir,
        )
        figures.append(fig_map)
        print(f"Saved: {path_map}")

        fig_ratio_map, path_ratio_map = save_spatial_distance_ratio_plot(
            reference_name,
            ref_sync,
            name,
            est_sync,
            starts_xy_ratio,
            drift_pct,
            highlight_delta,
            global_ratio,
            local_ratio_median,
            out_dir,
        )
        figures.append(fig_ratio_map)
        print(f"Saved: {path_ratio_map}")

    fig_ts, path_ts = save_rpe_timeseries_overlay(rpe_series, highlight_delta, out_dir)
    if fig_ts is not None:
        figures.append(fig_ts)
        print(f"Saved: {path_ts}")

    fig_box, path_box = save_rpe_boxplot_side_by_side(rpe_series, highlight_delta, out_dir)
    if fig_box is not None:
        figures.append(fig_box)
        print(f"Saved: {path_box}")

    fig_ratio_ts, path_ratio_ts = save_distance_ratio_timeseries_overlay(distance_series, highlight_delta, out_dir)
    if fig_ratio_ts is not None:
        figures.append(fig_ratio_ts)
        print(f"Saved: {path_ratio_ts}")

    fig_ratio_box, path_ratio_box = save_distance_ratio_boxplot_side_by_side(distance_series, highlight_delta, out_dir)
    if fig_ratio_box is not None:
        figures.append(fig_ratio_box)
        print(f"Saved: {path_ratio_box}")

    if show_figures:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


def main():
    args = parse_args()
    run(args.run_folder, show_figures=args.show, highlight_delta=args.highlight_delta)


if __name__ == "__main__":
    main()
