import argparse
from pathlib import Path

from mcap_extractor.align_save import align_folder
from mcap_extractor.align_visualize import visualize_aligned
from mcap_extractor.evaluate_metrics import evaluate_run_folder
from mcap_extractor.metrics_definitions import DEFAULT_HIGHLIGHT_DELTA_METERS
from mcap_extractor.visualize_metrics import run as visualize_metrics_run


def delta_token(delta_m):
    return f"{delta_m:g}".replace(".", "p")


def delta_subdir_name(delta_m):
    return f"delta_{delta_token(delta_m)}m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full post-processing pipeline from unaligned trajectories."
    )
    parser.add_argument(
        "run_folder",
        help="Path to run folder containing unaligned trajectories, for example: output/my_bag",
    )
    parser.add_argument(
        "--highlight-delta",
        type=float,
        default=DEFAULT_HIGHLIGHT_DELTA_METERS,
        help="Delta in meters for local RPE/DTE/distance-ratio metric visualizations.",
    )
    parser.add_argument(
        "--pair-selection",
        choices=["non_intersecting", "all"],
        default="non_intersecting",
        help="How to form local segment pairs: non_intersecting (default) or all.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show generated figures interactively.",
    )
    return parser.parse_args()


def run_pipeline(run_folder, highlight_delta=DEFAULT_HIGHLIGHT_DELTA_METERS, pair_selection="non_intersecting", show=False):
    run_dir = Path(run_folder)

    print("Step 1/4: Align trajectories")
    align_folder(str(run_dir))

    print("Step 2/4: Visualize aligned trajectories")
    # Keep this step non-blocking by default.
    visualize_aligned(str(run_dir), show=show)

    print("Step 3/4: Evaluate metrics")
    evaluate_run_folder(str(run_dir))

    print("Step 4/4: Visualize metrics")
    visualize_metrics_run(
        str(run_dir),
        show_figures=show,
        highlight_delta=highlight_delta,
        pair_selection=pair_selection,
    )

    metrics_dir = run_dir / "metrics"
    metrics_summary_path = metrics_dir / "metrics_summary.json"
    delta_dir = delta_subdir_name(highlight_delta)
    metrics_viz_dir = run_dir / "plots" / "metrics_viz"

    all_aligned_plot_path = metrics_viz_dir / "trajectories" / "all_aligned_xy.png"

    boxplot_paths = [
        metrics_viz_dir / "rpe" / delta_dir / f"rpe_boxplot_delta{delta_token(highlight_delta)}m.png",
        metrics_viz_dir / "dte" / delta_dir / f"dte_boxplot_delta{delta_token(highlight_delta)}m.png",
        metrics_viz_dir / "distance_ratio" / delta_dir / f"distance_ratio_boxplot_delta{delta_token(highlight_delta)}m.png",
    ]

    print("\nMain results")
    print(f"- All aligned trajectories plot: {all_aligned_plot_path}")
    print("- Boxplot locations:")
    for p in boxplot_paths:
        print(f"  - {p}")
    print(f"- Metrics summary JSON: {metrics_summary_path}")


def main():
    args = parse_args()
    run_pipeline(
        args.run_folder,
        highlight_delta=args.highlight_delta,
        pair_selection=args.pair_selection,
        show=args.show,
    )


if __name__ == "__main__":
    main()
