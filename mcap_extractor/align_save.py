import json
import sys
from pathlib import Path

import numpy as np

# Hardcoded alignment method.
# Available modes: "se2_full", "start_then_rot"
ALIGNMENT_MODE = "se2_full"

# If True, apply a constant z offset based on the first overlapping synchronized sample.
APPLY_Z_OFFSET = False

# Prefer this trajectory name as ground truth reference when available.
REFERENCE_NAME = "theodolite_trajectory"


def load_tum(filepath):
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    timestamps = data[:, 0]
    positions_xyz = data[:, 1:4]
    quats_xyzw = data[:, 4:8]

    order = np.argsort(timestamps)
    return timestamps[order], positions_xyz[order], quats_xyzw[order]


def save_tum(filepath, timestamps, positions_xyz, quats_xyzw):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for t, p, q in zip(timestamps, positions_xyz, quats_xyzw):
            f.write(
                f"{t:.9f} {p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n"
            )


def interp_xyz(target_ts, source_ts, source_xyz):
    x = np.interp(target_ts, source_ts, source_xyz[:, 0], left=np.nan, right=np.nan)
    y = np.interp(target_ts, source_ts, source_xyz[:, 1], left=np.nan, right=np.nan)
    z = np.interp(target_ts, source_ts, source_xyz[:, 2], left=np.nan, right=np.nan)
    return np.column_stack((x, y, z))


def get_best_rotation_2d(ref_xy, est_xy):
    h = est_xy.T @ ref_xy
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[1, :] *= -1
        r = vt.T @ u.T

    return r


def compute_transform(ref_sync_xyz, est_sync_xyz, mode):
    ref_xy = ref_sync_xyz[:, :2]
    est_xy = est_sync_xyz[:, :2]

    if mode == "se2_full":
        ref_centroid = np.mean(ref_xy, axis=0)
        est_centroid = np.mean(est_xy, axis=0)
        ref_zero = ref_xy - ref_centroid
        est_zero = est_xy - est_centroid
        r = get_best_rotation_2d(ref_zero, est_zero)
        t = ref_centroid - (r @ est_centroid)
    elif mode == "start_then_rot":
        ref_start = ref_xy[0]
        est_start = est_xy[0]
        ref_zero = ref_xy - ref_start
        est_zero = est_xy - est_start
        r = get_best_rotation_2d(ref_zero, est_zero)
        t = ref_start - (r @ est_start)
    else:
        raise ValueError(f"Unsupported ALIGNMENT_MODE: {mode}")

    return r, t


def apply_transform_xyz(xyz, r_2d, t_2d, z_offset=0.0):
    xy_aligned = (xyz[:, :2] @ r_2d.T) + t_2d
    z_aligned = xyz[:, 2:3] + z_offset
    return np.column_stack((xy_aligned, z_aligned))


def align_folder(run_folder):
    run_dir = Path(run_folder)
    if not run_dir.is_dir():
        print(f"Error: {run_folder} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    unaligned_dir = run_dir / "unaligned"
    aligned_dir = run_dir / "aligned"

    if not unaligned_dir.is_dir():
        print(f"Error: Missing unaligned directory at {unaligned_dir}", file=sys.stderr)
        sys.exit(1)

    input_files = [
        p for p in sorted(unaligned_dir.glob("*.txt")) if p.name != "theodolite_raw.txt"
    ]
    if not input_files:
        print(f"Error: No trajectory .txt files found in {unaligned_dir}", file=sys.stderr)
        sys.exit(1)

    trajectories = {}
    for p in input_files:
        ts, xyz, q = load_tum(p)
        if ts.size > 0:
            trajectories[p.stem] = {"ts": ts, "xyz": xyz, "q": q}

    if not trajectories:
        print("Error: No valid trajectory data loaded.", file=sys.stderr)
        sys.exit(1)

    ref_name = REFERENCE_NAME if REFERENCE_NAME in trajectories else next(iter(trajectories))
    ref = trajectories[ref_name]

    metadata = {
        "alignment_mode": ALIGNMENT_MODE,
        "reference": ref_name,
        "entries": [],
    }

    # Save reference trajectory as-is to aligned folder for consistent downstream processing.
    ref_out = aligned_dir / f"{ref_name}_aligned.txt"
    save_tum(ref_out, ref["ts"], ref["xyz"], ref["q"])
    metadata["entries"].append(
        {
            "name": ref_name,
            "source": str((unaligned_dir / f"{ref_name}.txt").resolve()),
            "output": str(ref_out.resolve()),
            "status": "reference_copied",
        }
    )

    for name, traj in trajectories.items():
        if name == ref_name:
            continue

        interp_est = interp_xyz(ref["ts"], traj["ts"], traj["xyz"])
        valid = ~np.isnan(interp_est[:, 0])

        if not np.any(valid):
            metadata["entries"].append(
                {
                    "name": name,
                    "source": str((unaligned_dir / f"{name}.txt").resolve()),
                    "status": "skipped_no_overlap",
                }
            )
            continue

        ref_sync = ref["xyz"][valid]
        est_sync = interp_est[valid]

        r, t = compute_transform(ref_sync, est_sync, ALIGNMENT_MODE)

        z_offset = 0.0
        if APPLY_Z_OFFSET:
            z_offset = float(ref_sync[0, 2] - est_sync[0, 2])

        aligned_xyz = apply_transform_xyz(traj["xyz"], r, t, z_offset=z_offset)

        out_path = aligned_dir / f"{name}_aligned.txt"
        save_tum(out_path, traj["ts"], aligned_xyz, traj["q"])

        metadata["entries"].append(
            {
                "name": name,
                "source": str((unaligned_dir / f"{name}.txt").resolve()),
                "output": str(out_path.resolve()),
                "overlap_samples": int(valid.sum()),
                "rotation": [[float(r[0, 0]), float(r[0, 1])], [float(r[1, 0]), float(r[1, 1])]],
                "translation_xy": [float(t[0]), float(t[1])],
                "z_offset": float(z_offset),
                "status": "aligned",
            }
        )

    aligned_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = aligned_dir / "alignment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Alignment completed with mode={ALIGNMENT_MODE}")
    print(f" - input:  {unaligned_dir}")
    print(f" - output: {aligned_dir}")
    print(f" - metadata: {metadata_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: poetry run align_data <output_run_folder>")
        print("Example: poetry run align_data output/my_bag")
        sys.exit(1)

    align_folder(sys.argv[1])


if __name__ == "__main__":
    main()
