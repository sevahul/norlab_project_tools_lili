import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mcap_extractor.plot_colors import build_color_map

def load_tum(filepath):
    """Loads a TUM trajectory file and returns timestamps and x-y-z coordinates."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1:4]

def resolve_aligned_dir(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    # Accept either the run folder (containing aligned/) or the aligned folder directly.
    aligned_dir = folder / "aligned"
    if aligned_dir.is_dir():
        return aligned_dir

    has_aligned_files = any(folder.glob("*_aligned.txt"))
    if has_aligned_files:
        return folder

    print(f"Error: could not find aligned trajectories in {folder_path}")
    sys.exit(1)

def visualize_aligned(folder_path, show=True):
    aligned_dir = resolve_aligned_dir(folder_path)

    traj_files = sorted(aligned_dir.glob("*_aligned.txt"))
    if not traj_files:
        print(f"No aligned trajectory files found in {aligned_dir}.")
        sys.exit(1)

    trajectories = {}
    for f in traj_files:
        ts, xyz = load_tum(f)
        if len(ts) > 0:
            trajectories[f.stem] = (ts, xyz)

    if not trajectories:
        print("No valid data found in the text files.")
        sys.exit(1)

    # Prefer theodolite aligned track as visual reference, fallback to first available.
    ref_name = "theodolite_trajectory_aligned"
    if ref_name not in trajectories:
        ref_name = list(trajectories.keys())[0]

    color_map = build_color_map([name for name in trajectories if name != ref_name])

    _, ref_xyz = trajectories[ref_name]

    plt.figure(figsize=(10, 8))

    # Center plots at the reference start only for easier visual comparison.
    center = ref_xyz[0, :2]

    for name, (_, xyz) in trajectories.items():
        xy = xyz[:, :2] - center
        if name == ref_name:
            # Show known reference points explicitly instead of connecting with lines.
            plt.scatter(xy[:, 0], xy[:, 1], s=7, color='black', alpha=0.45, label=name)
        else:
            plt.plot(xy[:, 0], xy[:, 1], color=color_map[name], alpha=0.9, label=name)

    plt.plot(0, 0, 'ro', markersize=8, label="Start Point (0,0)")
    plt.title("Saved Aligned Trajectories (XY)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: poetry run align_visualize <output_run_folder_or_aligned_folder>")
        sys.exit(1)

    visualize_aligned(sys.argv[1], show=True)

if __name__ == "__main__":
    main()