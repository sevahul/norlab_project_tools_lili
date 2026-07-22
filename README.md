# Tools for Seva's project in Norlab
This workspace now supports a 4-command workflow:

1. `extract_data` extracts trajectories and stores them in `unaligned/`.
2. `align_data` aligns all trajectories against the reference and saves them in `aligned/`.
3. `eval_metrics` computes translation-only ATE/RPE metrics from saved aligned trajectories.
4. `align_visualize` visualizes already saved aligned trajectories.
5. `visualize_metrics` renders dedicated metrics figures from `aligned/` and `metrics/`.
6. `run_pipeline` runs alignment, aligned visualization, metrics evaluation, and metrics visualization from `unaligned/`.

## Dependencies

### install poetry

```bash
sudo apt update && sudo apt install curl python3-venv -y
curl -sSL https://install.python-poetry.org | python3.12 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### init poetry

```bash
cd /home/seva/repos/deg-tunnel-tools
poetry env use python3.12
poetry install
```

## Command 1: Extract + Process

```bash
cd /home/seva/repos/deg-tunnel-tools
poetry run extract_data <path_to_mcap_file>
```

This generates:

```text
output/<bag-name>/
	unaligned/
		t265_trajectory.txt
		legged_trajectory.txt
		theodolite_trajectory.txt
		raw/
			theodolite_raw.txt
```

## Command 2: Align + Save

```bash
cd /home/seva/repos/deg-tunnel-tools
poetry run align_data output/<bag-name>
```

This generates:

```text
output/<bag-name>/
	aligned/
		theodolite_trajectory_aligned.txt
		t265_trajectory_aligned.txt
		legged_trajectory_aligned.txt
		alignment_metadata.json
```

## Command 3: Evaluate Metrics (Translation-Only)

```bash
cd /home/seva/repos/deg-tunnel-tools
poetry run eval_metrics output/<bag-name>
```

This generates:

```text
output/<bag-name>/
	metrics/
		t265_trajectory_aligned_vs_theodolite_trajectory_aligned.json
		legged_trajectory_aligned_vs_theodolite_trajectory_aligned.json
		metrics_summary.json
```

## Command 4: Visualize Saved Aligned Trajectories

```bash
cd /home/seva/repos/deg-tunnel-tools
poetry run align_visualize output/<bag-name>
```

## Command 5: Visualize Metrics

```bash
cd /home/seva/repos/deg-tunnel-tools
poetry run visualize_metrics output/<bag-name>
```

Optional interactive display:

```bash
poetry run visualize_metrics output/<bag-name> --show
```

Optional local-metrics delta in meters (RPE/LATE/DTE/distance-ratio):

```bash
poetry run visualize_metrics output/<bag-name> --highlight-delta 10
```

Optional local pair selection mode (default is non-intersecting/consecutive pairs):

```bash
poetry run visualize_metrics output/<bag-name> --pair-selection non_intersecting
poetry run visualize_metrics output/<bag-name> --pair-selection all
```

Note: local segment metrics skip segments that cross likely total-station loss/reacquisition gaps.

Implementation structure:
- Metric definitions/defaults are centralized in `mcap_extractor/metrics_definitions.py`.
- Metric calculations are centralized in `mcap_extractor/metrics_calculations.py`.
- `eval_metrics` and `visualize_metrics` import these shared modules.

This generates:

```text
output/<bag-name>/
	plots/
		metrics_viz/
			trajectories/
				all_aligned_xy.png
				pair_<trajectory>_vs_<reference>.png
			rpe/
				rpe_<trajectory>.png
				delta_10m/
					rpe_map_delta10m_<trajectory>.png
					rpe_boxplot_delta10m.png
					rpe_timeseries_overlay_delta10m.png
			late/
				delta_10m/
					late_map_delta10m_<trajectory>.png
					late_boxplot_delta10m.png
					late_timeseries_overlay_delta10m.png
			dte/
				delta_10m/
					dte_map_delta10m_<trajectory>.png
					dte_boxplot_delta10m.png
					dte_timeseries_overlay_delta10m.png
			distance_ratio/
				delta_10m/
					distance_ratio_map_delta10m_<trajectory>.png
					distance_ratio_boxplot_delta10m.png
					distance_ratio_timeseries_overlay_delta10m.png
			overlays/
				delta_10m/
```

## Command 6: Full Post-Processing Pipeline

```bash
cd /home/seva/repos/deg-tunnel-tools
poetry run run_pipeline output/<bag-name>
```

Optional arguments:

```bash
poetry run run_pipeline output/<bag-name> --highlight-delta 10 --pair-selection non_intersecting
poetry run run_pipeline output/<bag-name> --highlight-delta 10 --pair-selection all
poetry run run_pipeline output/<bag-name> --show
```

The pipeline prints these main result paths in the terminal:
- all trajectories aligned plot location
- boxplot locations (RPE, LATE, DTE, distance ratio)
- metrics summary JSON path