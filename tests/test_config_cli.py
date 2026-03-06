# tests/test_config_cli.py
import math
from pathlib import Path
import numpy as np
import pytest
from dolgov_cbmp.settings import parse_planning_inputs, load_world_model #, PlanningRunConfigModel
from dolgov_cbmp.structs import GridSpec, PlannerConfig, VehicleParams


def test_planner_config_rejects_even_kappa_rate_samples():
    with pytest.raises(ValueError):
        PlannerConfig(
            grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11), #, kappa_max=0.2),
            vehicle=VehicleParams(),
            curvature={"kappa_rate_samples": 4},
        )


def test_planner_config_rejects_step_size_larger_than_step_size_max():
    with pytest.raises(ValueError):
        PlannerConfig(
            grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11), # kappa_max=0.2),
            vehicle=VehicleParams(),
            step_size=2.0,
            step_policy={"step_size_max": 1.5},
        )


def test_load_world_model_from_legacy_npz(grid_npz_input_dir: Path):
    planner_cfg = PlannerConfig(grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11), vehicle=VehicleParams())
    occ = np.zeros((5, 5), dtype=bool)
    poses = np.array([[1, 1], [3, 3]])
    npz_path = grid_npz_input_dir / "loading_test_maze.npz"
    np.savez(npz_path, occupancy=occ, poses=poses)
    world, planner_cfg = load_world_model(npz_path, planner_cfg)
    assert world.occupancy_grid.height == 5
    assert world.start.x >= 0.0
    assert world.goal.pose.x >= 0.0
    assert planner_cfg.grid.resolution == 1.0

def test_load_world_model_from_yaml_and_npy(grid_input_dir: Path, cfg_input_dir: Path):
    planner_cfg =PlannerConfig(grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11), vehicle=VehicleParams())
    occ = np.zeros((6, 7), dtype=bool)
    occ[0,0] = True
    occ_path = grid_input_dir / "occ.npy"
    np.save(occ_path, occ)
    cfg_path = cfg_input_dir / "load_world_test.yaml"
    yaml_input = f"""
format_version: "1.0"
grid:
    resolution: 0.5
    theta_bins: 48
    origin_xy: [0.0, 0.0]
    kappa_bins: 11
occupancy:
    path: {occ_path}
start:
    x: 1.0
    y: 1.5
    theta: 0.1
    kappa: 0.0
goal:
    pose:
        x: 2.0
        y: 2.5
        theta: 0.2
        kappa: 0.0
    pos_tol: 0.8
    theta_tol: 0.1
step_size: 1.5
        """
    cfg_path.write_text(yaml_input, encoding="utf-8")
    world, planner_cfg = load_world_model(cfg_path, planner_cfg)
    assert world.occupancy_grid.occ.shape == (6, 7)
    assert planner_cfg.grid.resolution == 0.5
    assert planner_cfg.step_size == 1.5
    assert world.start.x == 1.0
    assert world.goal.pose.y == 2.5


def test_cli_supports_grouped_dot_overrides():
    args = parse_planning_inputs(
        [
            "--set",
            "weights.reverse_penalty=2.5",
            "--set",
            "heuristics.nh_table_xy_radius=12.0",
            "--set",
            "step_policy.step_size_max=4.0",
        ]
    )
    assert args.planner_config.weights.reverse_penalty == 2.5
    assert args.planner_config.heuristics.nh_table_xy_radius == 12.0
    assert args.planner_config.step_policy.step_size_max == 4.0


def test_cli_merges_yaml_and_command_line_overrides(cfg_input_dir: Path):
    # TODO: give these more descriptive names reflecting test parameters for when I add more robust tests of this functionality
    cfg = cfg_input_dir / "merged_cli_experiment.yaml"
    yaml_contents = """
backend: python
max_expansions: 12345
planner:
    grid:
        resolution: 0.8
        theta_bins: 64
    weights:
        reverse_penalty: 1.9
start:
    x: 1.0
    y: 2.0
    theta: 0.3
goal:
    pose:
        x: 9.0
        y: 8.0
        theta: 1.2
    pos_tol: 1.2
    theta_tol: 0.2
    """
    cfg.write_text(yaml_contents, encoding="utf-8")
    args = parse_planning_inputs(["--config", str(cfg), "--set", "weights.reverse_penalty=2.1"])
    assert args.backend == "python"
    assert args.max_expansions == 12345
    assert args.start.x == 1.0
    assert args.goal.pose.theta == pytest.approx(1.2)
    assert args.goal.theta_tol == pytest.approx(0.2)
    assert args.planner_config.grid.resolution == pytest.approx(0.8)
    assert args.planner_config.grid.theta_bins == 64
    assert args.planner_config.weights.reverse_penalty == pytest.approx(2.1)


def test_cli_rejects_invalid_dotted_set_format():
    with pytest.raises(ValueError):
        parse_planning_inputs(["--set", "weights.reverse_penalty"])


def test_cli_rejects_non_mapping_yaml_root(cfg_input_dir: Path):
    cfg = cfg_input_dir / "bad_mapping.yaml"
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_planning_inputs(["--config", str(cfg)])


def test_cli_rejects_invalid_connector_mode():
    with pytest.raises(ValueError):
        parse_planning_inputs(["--set", "connector.mode=invalid"])


def test_cli_accepts_numeric_pose_flags():
    args = parse_planning_inputs(
        [
            "--start.x",
            "3",
            "--start.y",
            "4",
            "--start.theta",
            "0.5",
            "--goal.theta",
            str(math.pi),
        ]
    )
    assert args.start.x == 3.0
    assert args.start.y == 4.0
    assert args.start.theta == pytest.approx(0.5)
    assert args.goal.pose.theta == pytest.approx(math.pi)


def test_cli_rejects_missing_world_cfg_file():
    with pytest.raises(ValueError):
        parse_planning_inputs(["--world-cfg", "missing_world.yaml"])