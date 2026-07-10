import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_tum(filepath):
    """Loads a TUM trajectory file and returns timestamps and x-y coordinates."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Return timestamps (N,) and X-Y positions (N, 2)
    return data[:, 0], data[:, 1:3]

def get_optimal_rotation(A, B):
    """
    Finds the optimal 2D rotation matrix R to align B to A using SVD.
    A and B are (N, 2) arrays centered at the origin.
    """
    H = B.T @ A
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
        
    return R

def align_and_plot(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)
        
    # Grab all txt files, but explicitly exclude the raw spherical data file
    traj_files = [f for f in folder.glob("*.txt") if f.name != "theodolite_raw.txt"]
    
    if not traj_files:
        print(f"No valid .txt trajectory files found in {folder_path}.")
        sys.exit(1)
        
    trajectories = {}
    for f in traj_files:
        ts, xy = load_tum(f)
        if len(ts) > 0:
            trajectories[f.stem] = (ts, xy)
            
    if not trajectories:
        print("No valid data found in the text files.")
        sys.exit(1)

    # Prefer theodolite as the reference ground truth, fallback to the first available
    ref_name = "theodolite_trajectory"
    if ref_name not in trajectories:
        ref_name = list(trajectories.keys())[0]
        
    ref_ts, ref_xy_raw = trajectories[ref_name]
    
    plt.figure(figsize=(10, 8))
    
    # Store and plot the reference trajectory centered at origin
    ref_xy_centered = ref_xy_raw - ref_xy_raw[0]
    plt.plot(ref_xy_centered[:, 0], ref_xy_centered[:, 1], 
             label=f"{ref_name} (Reference)", linewidth=2, linestyle='--', color='black')

    # Align and plot the other trajectories
    for name, (ts, xy_raw) in trajectories.items():
        if name == ref_name:
            continue
            
        # 1. Synchronize: Interpolate X and Y to match reference timestamps
        interp_x = np.interp(ref_ts, ts, xy_raw[:, 0], left=np.nan, right=np.nan)
        interp_y = np.interp(ref_ts, ts, xy_raw[:, 1], left=np.nan, right=np.nan)
        interp_xy = np.column_stack((interp_x, interp_y))
        
        # Mask out non-overlapping times
        valid_mask = ~np.isnan(interp_x)
        if not np.any(valid_mask):
            print(f"Warning: {name} has no overlapping timestamps with {ref_name}. Skipping.")
            continue
            
        valid_ref = ref_xy_centered[valid_mask]
        valid_interp = interp_xy[valid_mask]
        
        # 2. Translate: Ensure both start at (0,0) for the overlapping segment
        valid_ref_origin = valid_ref - valid_ref[0]
        valid_interp_origin = valid_interp - valid_interp[0]
        
        # 3. Compute optimal 2D rotation matrix
        R = get_optimal_rotation(valid_ref_origin, valid_interp_origin)
        
        # 4. Apply rotation
        aligned_xy = valid_interp_origin @ R.T
        
        # Re-apply the initial reference offset so it sits properly on the global graph
        aligned_xy_global = aligned_xy + valid_ref[0]
        
        # 5. Visualize
        plt.plot(aligned_xy_global[:, 0], aligned_xy_global[:, 1], label=f"{name} (Aligned)", alpha=0.8)

    plt.plot(0, 0, 'ro', markersize=8, label="Start Point (0,0)")
    plt.title("2D Trajectory Alignment (First Point Translation + SVD Rotation)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: poetry run python align_visualize.py <folder_with_trajectories>")
        sys.exit(1)

    align_and_plot(sys.argv[1])

if __name__ == "__main__":
    main()