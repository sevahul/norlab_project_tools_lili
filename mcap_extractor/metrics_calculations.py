import numpy as np


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


def get_best_rotation_2d(ref_xy, est_xy):
    """Return the 2D rotation matrix that best aligns est_xy to ref_xy."""
    h = est_xy.T @ ref_xy
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    # Enforce proper rotation (det = +1), no reflection.
    if np.linalg.det(r) < 0:
        vt[1, :] *= -1
        r = vt.T @ u.T

    return r


def cumulative_distance(positions):
    if len(positions) < 2:
        return np.zeros((len(positions),), dtype=float)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))


def trajectory_length(positions):
    if len(positions) < 2:
        return 0.0
    return float(cumulative_distance(positions)[-1])


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


def ratio_to_dte(ratio):
    return float(np.abs(1.0 - ratio))


def iter_delta_pairs(s_ref, delta_m, pair_selection):
    n = len(s_ref)
    if n < 2:
        return

    if pair_selection == "all":
        for i in range(n - 1):
            target_s = s_ref[i] + delta_m
            j = int(np.searchsorted(s_ref, target_s, side="left"))
            if j >= n:
                continue
            yield i, j
        return

    if pair_selection != "non_intersecting":
        raise ValueError(f"Unsupported pair_selection: {pair_selection}")

    # Consecutive non-overlapping pairs along reference distance.
    i = 0
    while i < n - 1:
        target_s = s_ref[i] + delta_m
        j = int(np.searchsorted(s_ref, target_s, side="left"))
        if j >= n:
            break
        yield i, j
        i = j


def compute_local_rpe_samples(ref_ts, ref_xyz, est_xyz, delta_m, invalid_prefix=None, pair_selection="non_intersecting"):
    if len(ref_xyz) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    s_ref = cumulative_distance(ref_xyz)
    starts_xy = []
    errors_m = []
    target_indices = []
    start_times = []

    for i, j in iter_delta_pairs(s_ref, delta_m, pair_selection):

        if invalid_prefix is not None and (invalid_prefix[j] - invalid_prefix[i]) > 0:
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


def compute_local_dte_samples(ref_ts, ref_xyz, est_xyz, delta_m, invalid_prefix=None, pair_selection="non_intersecting"):
    if len(ref_xyz) < 2:
        return np.array([]), np.array([]), np.array([])

    s_ref = cumulative_distance(ref_xyz)
    s_est = cumulative_distance(est_xyz)

    starts_xy = []
    dte_values = []
    start_times = []

    for i, j in iter_delta_pairs(s_ref, delta_m, pair_selection):

        if invalid_prefix is not None and (invalid_prefix[j] - invalid_prefix[i]) > 0:
            continue

        ref_len = float(s_ref[j] - s_ref[i])
        if ref_len <= 1e-9:
            continue

        est_len = float(s_est[j] - s_est[i])
        ratio = est_len / ref_len
        starts_xy.append(est_xyz[i, :2])
        dte_values.append(ratio_to_dte(ratio))
        start_times.append(ref_ts[i])

    if not dte_values:
        return np.array([]), np.array([]), np.array([])

    return np.array(starts_xy), np.array(dte_values), np.array(start_times)


def compute_local_distance_ratio_samples(ref_ts, ref_xyz, est_xyz, delta_m, invalid_prefix=None, pair_selection="non_intersecting"):
    if len(ref_xyz) < 2:
        return np.array([]), np.array([]), np.array([])

    s_ref = cumulative_distance(ref_xyz)
    s_est = cumulative_distance(est_xyz)

    starts_xy = []
    ratio_values = []
    start_times = []

    for i, j in iter_delta_pairs(s_ref, delta_m, pair_selection):

        if invalid_prefix is not None and (invalid_prefix[j] - invalid_prefix[i]) > 0:
            continue

        ref_len = float(s_ref[j] - s_ref[i])
        if ref_len <= 1e-9:
            continue

        est_len = float(s_est[j] - s_est[i])
        ratio = est_len / ref_len
        starts_xy.append(est_xyz[i, :2])
        ratio_values.append(ratio)
        start_times.append(ref_ts[i])

    if not ratio_values:
        return np.array([]), np.array([]), np.array([])

    return np.array(starts_xy), np.array(ratio_values), np.array(start_times)


def compute_local_late_samples(ref_ts, ref_xyz, est_xyz, delta_m, invalid_prefix=None, pair_selection="non_intersecting"):
    """Compute Local ATE (LATE) over intervals after local rigid alignment.

    For each interval [i, j], estimate and reference segments are aligned by
    start translation and best-fit 2D rotation (XY), then interval ATE RMSE is
    computed using compute_ate_rmse.
    """
    if len(ref_xyz) < 2:
        return np.array([]), np.array([]), np.array([])

    s_ref = cumulative_distance(ref_xyz)
    starts_xy = []
    late_values = []
    start_times = []

    for i, j in iter_delta_pairs(s_ref, delta_m, pair_selection):

        if invalid_prefix is not None and (invalid_prefix[j] - invalid_prefix[i]) > 0:
            continue

        ref_seg = ref_xyz[i : j + 1]
        est_seg = est_xyz[i : j + 1]
        if ref_seg.shape[0] < 2 or est_seg.shape[0] < 2:
            continue

        # Local start alignment + best-fit XY rotation.
        ref_xy = ref_seg[:, :2]
        est_xy = est_seg[:, :2]

        ref_xy_rel = ref_xy - ref_xy[0]
        est_xy_rel = est_xy - est_xy[0]

        r2d = get_best_rotation_2d(ref_xy_rel, est_xy_rel)
        est_xy_aligned = (est_xy_rel @ r2d.T) + ref_xy[0]

        # Keep Z aligned by start translation only.
        est_z_aligned = (est_seg[:, 2:3] - est_seg[0, 2:3]) + ref_seg[0, 2:3]
        est_seg_aligned = np.column_stack((est_xy_aligned, est_z_aligned))

        late_rmse, _ = compute_ate_rmse(ref_seg, est_seg_aligned)

        starts_xy.append(est_xyz[i, :2])
        late_values.append(late_rmse)
        start_times.append(ref_ts[i])

    if not late_values:
        return np.array([]), np.array([]), np.array([])

    return np.array(starts_xy), np.array(late_values), np.array(start_times)
