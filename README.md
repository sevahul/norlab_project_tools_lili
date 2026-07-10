# Tools for Seva's project in Norlab
This workspace now supports a 2-command workflow:

1. `extract_data` extracts trajectories and also processes raw theodolite data.
2. `align_visualize` aligns trajectories and visualizes the result.

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
	t265_trajectory.txt
	legged_trajectory.txt
	theodolite_trajectory.txt
	raw/
		theodolite_raw.txt
```

## Command 2: Align + Visualize

```bash
cd /home/seva/repos/deg-tunnel-tools
poetry run align_visualize output/<bag-name>
```