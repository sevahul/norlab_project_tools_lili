import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcap_extractor.metrics_calculations import (
    compute_local_distance_ratio_samples,
    compute_local_dte_samples,
    compute_local_rpe_samples,
    cumulative_distance,
    ratio_to_dte,
    trajectory_length,
)
from mcap_extractor.metrics_definitions import (
    DEFAULT_HIGHLIGHT_DELTA_METERS,
    PLOT_GROUP_DIRS,
    REFERENCE_ALIGNED_STEM,
)
from mcap_extractor.plot_colors import build_color_map, color_for_name

HIGHLIGHT_DELTA_METERS = DEFAULT_HIGHLIGHT_DELTA_METERS

# Heuristics to skip reference segments that likely cross total-station loss/reacquisition.
REF_GAP_FACTOR = 5.0
REF_MIN_GAP_SECONDS = 0.5
REF_STEP_JUMP_FACTOR = 8.0
REF_MIN_STEP_JUMP_METERS = 1.0
ACTIVE_REF_START_DISTANCE_M = 0.5
ACTIVE_REF_END_DISTANCE_M = 0.5

ACTIVE_COLOR_MAP = {}


def plot_color(name):
    return ACTIVE_COLOR_MAP.get(name, color_for_name(name))


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
        help="Delta in meters used for local RPE, DTE, and distance-ratio plots.",
    )
    parser.add_argument(
        "--pair-selection",
        choices=["non_intersecting", "all"],
        default="non_intersecting",
        help="How to form local segment pairs: non_intersecting (consecutive, default) or all.",
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


def delta_subdir_name(delta_m):
    return f"delta_{delta_token(delta_m)}m"


def save_all_aligned_plot(trajectories, reference_name, reports, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))

    ref_start = trajectories[reference_name]["xyz"][0, :2]

    for name, traj in trajectories.items():
        xyz = traj["xyz"]
        xy = xyz[:, :2] - ref_start
        if name == reference_name:
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                color="black",
                s=7,
                alpha=0.45,
                label=name,
            )
        else:
            ax.plot(xy[:, 0], xy[:, 1], color=plot_color(name), alpha=0.9, label=name)

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

    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], color="black", s=7, alpha=0.45, label=reference_name)
    ax.plot(est_xy[:, 0], est_xy[:, 1], linewidth=1.5, color=plot_color(estimate_name), alpha=0.95, label=estimate_name)

    ax.plot(0, 0, "ro", markersize=7, label="Reference start")
    ax.set_title(f"Trajectory vs Ground Truth: {estimate_name}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    global_dte = ratio_to_dte(global_ratio)
    add_info_box(
        ax,
        (
            f"ATE: {ate_rmse_m:.3f} m\n"
            f"{reference_name} length: {reference_len_m:.2f} m\n"
            f"{estimate_name} length: {estimate_len_m:.2f} m\n"
            f"Global DTE |1-ratio|: {global_dte:.3f}"
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
        ax.plot(t_rel, errors_m, linewidth=1.4, color=plot_color(name), alpha=0.95, label=name)

    ax.set_xlabel("Time from start [s]")
    ax.set_ylabel("Local RPE [m]")
    ax.set_title(f"RPE Time Series Overlay (delta={delta_m:g} m)")
    ax.grid(True)
    ax.legend()

    lines = []
    for name in sorted(rpe_series.keys()):
        ate_rmse_m = rpe_series[name].get("ate_rmse_m")
        est_len_m = rpe_series[name].get("estimate_len_m")
        global_dte = rpe_series[name].get("global_dte")
        local_dte = rpe_series[name].get("local_dte_median")
        if ate_rmse_m is None or est_len_m is None:
            continue
        lines.append(
            f"{name}: ATE={ate_rmse_m:.3f} m, len={est_len_m:.2f} m, "
            f"DTE={global_dte:.3f}, local~{local_dte:.3f}"
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
        patch.set_facecolor(plot_color(names[idx]))
        patch.set_alpha(0.55)

    y_min, y_max = ax.get_ylim()
    y_span = max(1e-9, y_max - y_min)
    ax.set_ylim(y_min, y_max + 0.14 * y_span)
    y_text = y_max + 0.04 * y_span
    for idx, name in enumerate(names):
        n_samples = int(rpe_series[name]["errors_m"].size)
        ax.text(idx + 1, y_text, f"n={n_samples}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Local RPE [m]")
    ax.set_title(f"RPE Boxplots Side-by-Side (delta={delta_m:g} m)")
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()

    out_path = output_dir / f"rpe_boxplot_delta{delta_token(delta_m)}m.png"
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
    global_dte,
    local_dte_median,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    ref_start = reference_xyz[0, :2]
    ref_xy = reference_xyz[:, :2] - ref_start
    est_xy = estimate_xyz[:, :2] - ref_start
    starts_xy_centered = starts_xy - ref_start

    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], color="black", s=7, alpha=0.45, label=reference_name)
    ax.plot(est_xy[:, 0], est_xy[:, 1], linewidth=1.5, color=plot_color(estimate_name), alpha=0.55, label=estimate_name)

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
            f"DTE: {global_dte:.3f}, local~{local_dte_median:.3f}"
        ),
    )
    fig.tight_layout()

    out_path = output_dir / f"rpe_map_delta{delta_token(delta_m)}m_{sanitize_name(estimate_name)}.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_spatial_dte_plot(
    reference_name,
    reference_xyz,
    estimate_name,
    estimate_xyz,
    starts_xy,
    dte_values,
    delta_m,
    global_dte,
    local_dte_median,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    ref_start = reference_xyz[0, :2]
    ref_xy = reference_xyz[:, :2] - ref_start
    est_xy = estimate_xyz[:, :2] - ref_start
    starts_xy_centered = starts_xy - ref_start

    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], color="black", s=7, alpha=0.45, label=reference_name)
    ax.plot(est_xy[:, 0], est_xy[:, 1], linewidth=1.5, color=plot_color(estimate_name), alpha=0.55, label=estimate_name)

    vmax = float(max(0.05, np.percentile(dte_values, 95)))
    scatter = ax.scatter(
        starts_xy_centered[:, 0],
        starts_xy_centered[:, 1],
        c=dte_values,
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        s=22,
        edgecolors="none",
        label=f"DTE start points (delta={delta_m:g} m)",
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Local DTE |1-ratio| [-]")

    ax.plot(0, 0, "ro", markersize=7, label="Reference start")
    ax.set_title(f"Spatial DTE Map ({delta_m:g} m): {estimate_name}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    add_info_box(
        ax,
        (
            f"Global DTE |1-ratio|: {global_dte:.3f}\n"
            f"Local median DTE: {local_dte_median:.3f}"
        ),
    )
    fig.tight_layout()

    out_path = output_dir / f"dte_map_delta{delta_token(delta_m)}m_{sanitize_name(estimate_name)}.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_dte_timeseries_overlay(distance_series, delta_m, output_dir):
    if len(distance_series) < 1:
        return None, None

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for name in sorted(distance_series.keys()):
        times = distance_series[name]["times"]
        dte_values = distance_series[name]["dte_values"]
        if times.size == 0 or dte_values.size == 0:
            continue
        t_rel = times - times[0]
        ax.plot(t_rel, dte_values, linewidth=1.4, color=plot_color(name), alpha=0.95, label=name)

    ax.set_xlabel("Time from start [s]")
    ax.set_ylabel("Local DTE |1-ratio| [-]")
    ax.set_title(f"DTE Time Series Overlay (delta={delta_m:g} m)")
    ax.grid(True)
    ax.legend()

    lines = []
    for name in sorted(distance_series.keys()):
        g = distance_series[name].get("global_dte")
        m = distance_series[name].get("median_dte")
        if g is None or m is None:
            continue
        lines.append(f"{name}: global DTE={g:.3f}, local~{m:.3f}")
    if lines:
        add_info_box(ax, "\n".join(lines))

    fig.tight_layout()

    out_path = output_dir / f"dte_timeseries_overlay_delta{delta_token(delta_m)}m.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_dte_boxplot_side_by_side(distance_series, delta_m, output_dir):
    names = [name for name in sorted(distance_series.keys()) if distance_series[name].get("dte_values", np.array([])).size > 0]
    if len(names) < 1:
        return None, None

    data = [distance_series[name]["dte_values"] for name in names]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    try:
        box = ax.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)
    except TypeError:
        box = ax.boxplot(data, labels=names, patch_artist=True, showfliers=False)

    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(plot_color(names[idx]))
        patch.set_alpha(0.55)

    y_min, y_max = ax.get_ylim()
    y_span = max(1e-9, y_max - y_min)
    ax.set_ylim(y_min, y_max + 0.14 * y_span)
    y_text = y_max + 0.04 * y_span
    for idx, name in enumerate(names):
        n_samples = int(distance_series[name]["dte_values"].size)
        ax.text(idx + 1, y_text, f"n={n_samples}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Local DTE |1-ratio| [-]")
    ax.set_title(f"DTE Boxplots Side-by-Side (delta={delta_m:g} m)")
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    out_path = output_dir / f"dte_boxplot_delta{delta_token(delta_m)}m.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_spatial_distance_ratio_plot(
    reference_name,
    reference_xyz,
    estimate_name,
    estimate_xyz,
    starts_xy,
    ratio_values,
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

    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], color="black", s=7, alpha=0.45, label=reference_name)
    ax.plot(est_xy[:, 0], est_xy[:, 1], linewidth=1.5, color=plot_color(estimate_name), alpha=0.55, label=estimate_name)

    vmax = float(max(1.0, np.percentile(np.abs(ratio_values), 95)))
    scatter = ax.scatter(
        starts_xy_centered[:, 0],
        starts_xy_centered[:, 1],
        c=ratio_values,
        cmap="coolwarm",
        vmin=1.0 - (vmax - 1.0),
        vmax=vmax,
        s=22,
        edgecolors="none",
        label=f"Distance ratio start points (delta={delta_m:g} m)",
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Local distance ratio [-]")

    ax.plot(0, 0, "ro", markersize=7, label="Reference start")
    ax.set_title(f"Spatial Distance-Ratio Map ({delta_m:g} m): {estimate_name}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    add_info_box(
        ax,
        (
            f"Global ratio (est/ref): {global_ratio:.3f}\n"
            f"Local median ratio: {local_ratio_median:.3f}"
        ),
    )
    fig.tight_layout()

    out_path = output_dir / f"distance_ratio_map_delta{delta_token(delta_m)}m_{sanitize_name(estimate_name)}.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_distance_ratio_timeseries_overlay(ratio_series, delta_m, output_dir):
    if len(ratio_series) < 1:
        return None, None

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for name in sorted(ratio_series.keys()):
        times = ratio_series[name]["times"]
        ratio_values = ratio_series[name]["ratio_values"]
        if times.size == 0 or ratio_values.size == 0:
            continue
        t_rel = times - times[0]
        ax.plot(t_rel, ratio_values, linewidth=1.4, color=plot_color(name), alpha=0.95, label=name)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time from start [s]")
    ax.set_ylabel("Local distance ratio [-]")
    ax.set_title(f"Distance-Ratio Time Series Overlay (delta={delta_m:g} m)")
    ax.grid(True)
    ax.legend()

    lines = []
    for name in sorted(ratio_series.keys()):
        g = ratio_series[name].get("global_ratio")
        m = ratio_series[name].get("median_ratio")
        if g is None or m is None:
            continue
        lines.append(f"{name}: global={g:.3f}, local~{m:.3f}")
    if lines:
        add_info_box(ax, "\n".join(lines))

    fig.tight_layout()

    out_path = output_dir / f"distance_ratio_timeseries_overlay_delta{delta_token(delta_m)}m.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def save_distance_ratio_boxplot_side_by_side(ratio_series, delta_m, output_dir):
    names = [name for name in sorted(ratio_series.keys()) if ratio_series[name].get("ratio_values", np.array([])).size > 0]
    if len(names) < 1:
        return None, None

    data = [ratio_series[name]["ratio_values"] for name in names]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    try:
        box = ax.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)
    except TypeError:
        box = ax.boxplot(data, labels=names, patch_artist=True, showfliers=False)

    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(plot_color(names[idx]))
        patch.set_alpha(0.55)

    y_min, y_max = ax.get_ylim()
    y_span = max(1e-9, y_max - y_min)
    ax.set_ylim(y_min, y_max + 0.14 * y_span)
    y_text = y_max + 0.04 * y_span
    for idx, name in enumerate(names):
        n_samples = int(ratio_series[name]["ratio_values"].size)
        ax.text(idx + 1, y_text, f"n={n_samples}", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Local distance ratio [-]")
    ax.set_title(f"Distance-Ratio Boxplots Side-by-Side (delta={delta_m:g} m)")
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    out_path = output_dir / f"distance_ratio_boxplot_delta{delta_token(delta_m)}m.png"
    fig.savefig(out_path, dpi=180)
    return fig, out_path


def parse_delta_key(delta_key):
    if not delta_key.endswith("m"):
        return None
    try:
        return float(delta_key[:-1])
    except ValueError:
        return None


def ensure_plot_dirs(base_output_dir):
    dirs = {}
    for key, subdir in PLOT_GROUP_DIRS.items():
        p = base_output_dir / subdir
        p.mkdir(parents=True, exist_ok=True)
        dirs[key] = p
    return dirs


def ensure_delta_plot_dirs(plot_dirs, delta_m):
    """Create per-delta subfolders for delta-dependent plot families."""
    delta_dir = delta_subdir_name(delta_m)
    delta_dirs = {}
    for key in ["rpe", "dte", "distance_ratio", "overlays"]:
        p = plot_dirs[key] / delta_dir
        p.mkdir(parents=True, exist_ok=True)
        delta_dirs[key] = p
    return delta_dirs


def save_rpe_plot(
    pair_metrics,
    estimate_name,
    reference_name,
    reference_len_m,
    estimate_len_m,
    local_dte_median,
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

    ax1.plot(deltas, rmse_m, marker="o", color=plot_color(estimate_name))
    ax1.set_ylabel("RPE RMSE [m]")
    ax1.set_title(f"RPE by Delta: {estimate_name}")
    ax1.grid(True)

    ax2.plot(deltas, rmse_pct, marker="o", color=plot_color(estimate_name))
    ax2.set_xlabel("Delta [m]")
    ax2.set_ylabel("RPE RMSE [%]")
    ax2.grid(True)

    ate_rmse_m = pair_metrics.get("ate_rmse_m")
    global_ratio = estimate_len_m / reference_len_m if reference_len_m > 0 else float("nan")
    global_dte = ratio_to_dte(global_ratio)
    if ate_rmse_m is not None:
        add_info_box(
            ax1,
            (
                f"ATE: {ate_rmse_m:.3f} m\n"
                f"{reference_name} length: {reference_len_m:.2f} m\n"
                f"{estimate_name} length: {estimate_len_m:.2f} m\n"
                f"DTE |1-ratio|: {global_dte:.3f}, local~{local_dte_median:.3f}"
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


def run(run_folder, show_figures=False, highlight_delta=HIGHLIGHT_DELTA_METERS, pair_selection="non_intersecting"):
    global ACTIVE_COLOR_MAP
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
    ACTIVE_COLOR_MAP = build_color_map([name for name in trajectories if name != reference_name])

    reports = load_metrics_reports(metrics_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dirs = ensure_plot_dirs(out_dir)
    delta_plot_dirs = ensure_delta_plot_dirs(plot_dirs, highlight_delta)

    figures = []
    rpe_series = {}
    dte_series = {}
    ratio_series = {}

    fig, path = save_all_aligned_plot(trajectories, reference_name, reports, plot_dirs["trajectories"])
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
            plot_dirs["trajectories"],
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
            pair_selection=pair_selection,
        )
        if errors_m.size == 0:
            print(f"Warning: no local RPE samples found for {name} at delta={highlight_delta:g}m.")
            continue

        starts_xy_ratio, ratio_values, ratio_times = compute_local_distance_ratio_samples(
            ref_ts_sync,
            ref_sync,
            est_sync,
            highlight_delta,
            invalid_prefix=invalid_prefix,
            pair_selection=pair_selection,
        )
        if ratio_values.size == 0:
            print(f"Warning: no local distance-ratio samples found for {name} at delta={highlight_delta:g}m.")
            continue

        starts_xy_dte, dte_values, dte_times = compute_local_dte_samples(
            ref_ts_sync,
            ref_sync,
            est_sync,
            highlight_delta,
            invalid_prefix=invalid_prefix,
            pair_selection=pair_selection,
        )
        if dte_values.size == 0:
            print(f"Warning: no local DTE samples found for {name} at delta={highlight_delta:g}m.")
            continue

        local_ratio_median = float(np.median(ratio_values))
        local_dte_median = float(np.median(dte_values))
        global_dte = ratio_to_dte(global_ratio)

        fig_rpe, path_rpe = save_rpe_plot(
            report,
            name,
            reference_name,
            reference_len_m,
            estimate_len_m,
            local_dte_median,
            plot_dirs["rpe"],
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
            "global_dte": global_dte,
            "local_dte_median": local_dte_median,
        }

        dte_series[name] = {
            "times": dte_times,
            "dte_values": dte_values,
            "global_dte": global_dte,
            "median_dte": local_dte_median,
        }

        ratio_series[name] = {
            "times": ratio_times,
            "ratio_values": ratio_values,
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
            global_dte,
            local_dte_median,
            delta_plot_dirs["rpe"],
        )
        figures.append(fig_map)
        print(f"Saved: {path_map}")

        fig_ratio_map, path_ratio_map = save_spatial_dte_plot(
            reference_name,
            ref_sync,
            name,
            est_sync,
            starts_xy_dte,
            dte_values,
            highlight_delta,
            global_dte,
            local_dte_median,
            delta_plot_dirs["dte"],
        )
        figures.append(fig_ratio_map)
        print(f"Saved: {path_ratio_map}")

        fig_dr_map, path_dr_map = save_spatial_distance_ratio_plot(
            reference_name,
            ref_sync,
            name,
            est_sync,
            starts_xy_ratio,
            ratio_values,
            highlight_delta,
            global_ratio,
            local_ratio_median,
            delta_plot_dirs["distance_ratio"],
        )
        figures.append(fig_dr_map)
        print(f"Saved: {path_dr_map}")

    fig_ts, path_ts = save_rpe_timeseries_overlay(rpe_series, highlight_delta, delta_plot_dirs["overlays"])
    if fig_ts is not None:
        figures.append(fig_ts)
        print(f"Saved: {path_ts}")

    fig_box, path_box = save_rpe_boxplot_side_by_side(rpe_series, highlight_delta, delta_plot_dirs["rpe"])
    if fig_box is not None:
        figures.append(fig_box)
        print(f"Saved: {path_box}")

    fig_ratio_ts, path_ratio_ts = save_dte_timeseries_overlay(dte_series, highlight_delta, delta_plot_dirs["overlays"])
    if fig_ratio_ts is not None:
        figures.append(fig_ratio_ts)
        print(f"Saved: {path_ratio_ts}")

    fig_ratio_box, path_ratio_box = save_dte_boxplot_side_by_side(dte_series, highlight_delta, delta_plot_dirs["dte"])
    if fig_ratio_box is not None:
        figures.append(fig_ratio_box)
        print(f"Saved: {path_ratio_box}")

    fig_dr_ts, path_dr_ts = save_distance_ratio_timeseries_overlay(ratio_series, highlight_delta, delta_plot_dirs["overlays"])
    if fig_dr_ts is not None:
        figures.append(fig_dr_ts)
        print(f"Saved: {path_dr_ts}")

    fig_dr_box, path_dr_box = save_distance_ratio_boxplot_side_by_side(ratio_series, highlight_delta, delta_plot_dirs["distance_ratio"])
    if fig_dr_box is not None:
        figures.append(fig_dr_box)
        print(f"Saved: {path_dr_box}")

    if show_figures:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


def main():
    args = parse_args()
    run(
        args.run_folder,
        show_figures=args.show,
        highlight_delta=args.highlight_delta,
        pair_selection=args.pair_selection,
    )


if __name__ == "__main__":
    main()
