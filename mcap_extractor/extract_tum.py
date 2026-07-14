import argparse
import sys
from pathlib import Path

import yaml
from mcap_ros2.reader import read_ros2_messages

from mcap_extractor.process_theodolite import process_raw_theodolite

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
DEFAULT_OUTPUT_ROOT = "output"
DEFAULT_TYPE = "odometry"
DEFAULT_TIMESTAMP_BY_TYPE = {
    "odometry": "header",
    "theodolite": "arrival",
}


def load_config(config_path):
    path = Path(config_path)
    if not path.is_file():
        print(f"Error: Missing config file: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    trajectories = cfg.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        print("Error: config must define a non-empty 'trajectories' mapping.", file=sys.stderr)
        sys.exit(1)

    normalized = {}
    for name, spec in trajectories.items():
        if not isinstance(spec, dict):
            print(f"Error: trajectory '{name}' must be a mapping.", file=sys.stderr)
            sys.exit(1)

        topic = spec.get("topic")
        if not topic:
            print(f"Error: trajectory '{name}' is missing required field 'topic'.", file=sys.stderr)
            sys.exit(1)

        traj_type = spec.get("type", DEFAULT_TYPE)
        if traj_type not in DEFAULT_TIMESTAMP_BY_TYPE:
            print(
                f"Error: trajectory '{name}' has unsupported type '{traj_type}'.",
                file=sys.stderr,
            )
            sys.exit(1)

        timestamp_source = spec.get("timestamp_source", DEFAULT_TIMESTAMP_BY_TYPE[traj_type])
        if timestamp_source not in ("header", "arrival"):
            print(
                f"Error: trajectory '{name}' has invalid timestamp_source '{timestamp_source}'.",
                file=sys.stderr,
            )
            sys.exit(1)

        normalized[name] = {
            "topic": topic,
            "type": traj_type,
            "timestamp_source": timestamp_source,
        }

    return cfg, normalized


def resolve_paths(cfg, mcap_file, cli_output_root=None):
    input_from_config = cfg.get("input")
    mcap_path = mcap_file if mcap_file is not None else input_from_config
    if not mcap_path:
        print("Error: MCAP input path not provided (CLI or config.input).", file=sys.stderr)
        sys.exit(1)

    output_cfg = cfg.get("output") or {}
    output_root = cli_output_root if cli_output_root else output_cfg.get("folder", DEFAULT_OUTPUT_ROOT)

    mcap_path = Path(mcap_path)
    output_dir = Path(output_root) / mcap_path.stem
    unaligned_dir = output_dir / "unaligned"
    raw_dir = unaligned_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    unaligned_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return mcap_path, output_dir, unaligned_dir, raw_dir


def get_timestamp(msg, topic, timestamp_source):
    if timestamp_source == "arrival":
        return msg.log_time.timestamp()

    ros_msg = msg.ros_msg
    try:
        sec = ros_msg.header.stamp.sec
        nanosec = ros_msg.header.stamp.nanosec
        return sec + (nanosec * 1e-9)
    except AttributeError:
        print(f"Warning: Message on {topic} missing header.stamp. Skipping.")
        return None


def trajectory_output_path(name, traj_type, unaligned_dir, raw_dir):
    if traj_type == "theodolite":
        raw_name = "theodolite_raw.txt" if name == "theodolite" else f"{name}_raw.txt"
        proc_name = f"{name}_trajectory.txt"
        return raw_dir / raw_name, unaligned_dir / proc_name

    return unaligned_dir / f"{name}_trajectory.txt", None


def extract_trajectories(mcap_file=None, output_root=None, config_path=DEFAULT_CONFIG_PATH):
    cfg, trajectories = load_config(config_path)
    mcap_path, output_dir, unaligned_dir, raw_dir = resolve_paths(cfg, mcap_file, output_root)

    topic_to_name = {}
    topic_list = []
    for name, spec in trajectories.items():
        topic = spec["topic"]
        topic_to_name[topic] = name
        topic_list.append(topic)

    out_files = {}
    writers = {}
    theo_postprocess = []

    try:
        for name, spec in trajectories.items():
            raw_or_traj_path, maybe_proc_path = trajectory_output_path(
                name,
                spec["type"],
                unaligned_dir,
                raw_dir,
            )
            out_files[name] = {"type": spec["type"]}

            if spec["type"] == "theodolite":
                writers[name] = open(raw_or_traj_path, "w")
                out_files[name]["raw"] = raw_or_traj_path
                out_files[name]["trajectory"] = maybe_proc_path
                theo_postprocess.append((name, raw_or_traj_path, maybe_proc_path))
            else:
                writers[name] = open(raw_or_traj_path, "w")
                out_files[name]["trajectory"] = raw_or_traj_path

        print(f"Reading {mcap_path}...")

        try:
            for msg in read_ros2_messages(str(mcap_path), topics=topic_list):
                topic = msg.channel.topic
                name = topic_to_name.get(topic)
                if name is None:
                    continue

                spec = trajectories[name]
                ros_msg = msg.ros_msg
                timestamp = get_timestamp(msg, topic, spec["timestamp_source"])
                if timestamp is None:
                    continue

                if spec["type"] == "odometry":
                    position = ros_msg.pose.pose.position
                    orientation = ros_msg.pose.pose.orientation
                    line = (
                        f"{timestamp:.9f} {position.x} {position.y} {position.z} "
                        f"{orientation.x} {orientation.y} {orientation.z} {orientation.w}\n"
                    )
                    writers[name].write(line)
                elif spec["type"] == "theodolite":
                    if ros_msg.status != 0:
                        continue

                    azimuth = ros_msg.azimuth
                    elevation = ros_msg.elevation
                    distance = ros_msg.distance
                    line = f"{timestamp:.9f} {azimuth} {elevation} {distance}\n"
                    writers[name].write(line)

        except Exception as e:
            print(f"An error occurred while parsing the MCAP file: {e}")
            sys.exit(1)
    finally:
        for f in writers.values():
            f.close()

    for name, raw_path, proc_path in theo_postprocess:
        processed = process_raw_theodolite(raw_path, output_path=proc_path)
        out_files[name]["trajectory"] = processed

    print("Extraction complete. The following files have been generated:")
    for name, info in out_files.items():
        if info["type"] == "theodolite":
            print(f" - {name} raw: {info['raw']}")
            print(f" - {name} trajectory: {info['trajectory']}")
        else:
            print(f" - {name} trajectory: {info['trajectory']}")

    result = {
        "output_dir": output_dir,
        "unaligned_dir": unaligned_dir,
        "trajectories": out_files,
    }

    # Backward-compatible keys for the current default config names.
    if "t265" in out_files:
        result["t265"] = out_files["t265"].get("trajectory")
    if "legged" in out_files:
        result["legged"] = out_files["legged"].get("trajectory")
    if "theodolite" in out_files:
        result["theodolite_raw"] = out_files["theodolite"].get("raw")
        result["theodolite_processed"] = out_files["theodolite"].get("trajectory")

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract trajectories from an MCAP file.")
    parser.add_argument("mcap_file", nargs="?", help="Path to input MCAP file.")
    parser.add_argument(
        "output_root",
        nargs="?",
        help="Optional output root (overrides config.output.folder).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file (default: config/default.yaml).",
    )
    args = parser.parse_args()

    extract_trajectories(
        mcap_file=args.mcap_file,
        output_root=args.output_root,
        config_path=args.config,
    )

if __name__ == "__main__":
    main()