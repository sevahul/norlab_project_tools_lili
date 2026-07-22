import sys
from pathlib import Path
from mcap_ros2.reader import read_ros2_messages
from mcap_extractor.process_theodolite import process_raw_theodolite

# Configuration for timestamp sources
TIMESTAMP_SOURCES = {
    "/t265/odom/sample": "header",
    "/legged_odometry/pose_in_odom": "header",
    "/theodolite_data": "arrival"
}

def extract_trajectories(mcap_file, output_root="output"):
    mcap_path = Path(mcap_file)
    output_dir = Path(output_root) / mcap_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    unaligned_dir = output_dir / "unaligned"
    unaligned_dir.mkdir(parents=True, exist_ok=True)
    output_theodolite_dir = unaligned_dir / "raw"
    output_theodolite_dir.mkdir(parents=True, exist_ok=True)
    
    out_t265 = unaligned_dir / "t265_trajectory.txt"
    out_legged = unaligned_dir / "legged_trajectory.txt"
    out_theo_raw = output_theodolite_dir / "theodolite_raw.txt"

    TOPIC_T265 = "/t265/odom/sample"
    TOPIC_LEGGED = "/legged_odometry/pose_in_odom"
    TOPIC_THEO = "/theodolite_data"

    with \
         open(out_t265, "w") as f_t265, \
         open(out_legged, "w") as f_legged, \
         open(out_theo_raw, "w") as f_theo_raw:

        print(f"Reading {mcap_file}...")
        
        try:
            for msg in read_ros2_messages(mcap_file, topics=[TOPIC_T265, TOPIC_LEGGED, TOPIC_THEO]):
                topic = msg.channel.topic
                ros_msg = msg.ros_msg
                source = TIMESTAMP_SOURCES.get(topic, "header")

                if source == "arrival":
                    timestamp = msg.log_time.timestamp()
                else:
                    try:
                        sec = ros_msg.header.stamp.sec
                        nanosec = ros_msg.header.stamp.nanosec
                        timestamp = sec + (nanosec * 1e-9)
                    except AttributeError:
                        print(f"Warning: Message on {topic} missing header.stamp. Skipping.")
                        continue

                if topic in (TOPIC_T265, TOPIC_LEGGED):
                    position = ros_msg.pose.pose.position
                    orientation = ros_msg.pose.pose.orientation
                    
                    line = f"{timestamp:.9f} {position.x} {position.y} {position.z} {orientation.x} {orientation.y} {orientation.z} {orientation.w}\n"
                    
                    if topic == TOPIC_T265:
                        f_t265.write(line)
                    else:
                        f_legged.write(line)

                elif topic == TOPIC_THEO:
                    # STRICT FILTER: Only accept valid tracking status
                    if ros_msg.status != 0:
                        continue

                    # Dump raw spherical data
                    azimuth = ros_msg.azimuth
                    elevation = ros_msg.elevation
                    distance = ros_msg.distance

                    line = f"{timestamp:.9f} {azimuth} {elevation} {distance}\n"
                    f_theo_raw.write(line)

        except Exception as e:
            print(f"An error occurred while parsing the MCAP file: {e}")
            sys.exit(1)

    # Auto-generate output/<bag-name>/theodolite_trajectory.txt from raw data.
    out_theo_processed = process_raw_theodolite(
        out_theo_raw,
        output_path=unaligned_dir / "theodolite_trajectory.txt",
    )

    print("Extraction complete. The following files have been generated:")
    print(f" - {out_t265}")
    print(f" - {out_legged}")
    print(f" - {out_theo_raw}")
    print(f" - {out_theo_processed}")

    return {
        "output_dir": output_dir,
        "unaligned_dir": unaligned_dir,
        "t265": out_t265,
        "legged": out_legged,
        "theodolite_raw": out_theo_raw,
        "theodolite_processed": out_theo_processed,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: poetry run python extract_tum.py <path_to_mcap_file> [output_root]")
        sys.exit(1)

    mcap_path = sys.argv[1]
    output_root = sys.argv[2] if len(sys.argv) > 2 else "output"
    extract_trajectories(mcap_path, output_root=output_root)

if __name__ == "__main__":
    main()