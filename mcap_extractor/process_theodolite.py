import sys
import math
from pathlib import Path

def process_raw_theodolite(raw_file_path, output_path=None):
    raw_path = Path(raw_file_path)
    if not raw_path.is_file():
        print(f"Error: Could not find {raw_file_path}")
        sys.exit(1)

    # Default output sits next to the raw folder (output/<bag-name>/theodolite_trajectory.txt)
    if output_path is None:
        out_theo = raw_path.parent.parent / "theodolite_trajectory.txt"
    else:
        out_theo = Path(output_path)
    out_theo.parent.mkdir(parents=True, exist_ok=True)
    
    with open(raw_path, "r") as f_in, open(out_theo, "w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
                
            timestamp = float(parts[0])
            azimuth = float(parts[1]) 
            elevation = float(parts[2]) - (math.pi / 2)  # Adjust elevation to be relative to horizontal plane
            distance = float(parts[3])
            
            # Convert spherical to Cartesian
            x = - distance * math.cos(elevation) * math.cos(azimuth)
            y = distance * math.cos(elevation) * math.sin(azimuth)
            z = distance * math.sin(elevation)

            # Identity quaternion
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

            out_line = f"{timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
            f_out.write(out_line)

    print(f"Processed raw theodolite data. TUM trajectory saved to:\n - {out_theo}")
    return out_theo


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: poetry run python process_theodolite.py "
            "<path_to_theodolite_raw.txt> [output_theodolite_trajectory_path]"
        )
        sys.exit(1)

    raw_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    process_raw_theodolite(raw_file, output_file)


if __name__ == "__main__":
    main()