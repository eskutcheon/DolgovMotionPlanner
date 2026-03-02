import math
from pathlib import Path
import pytest
from src.settings import parse_planning_inputs #, PlanningRunConfigModel
from src.structs import GridSpec, PlannerConfig, VehicleParams


def test_planner_config_rejects_even_kappa_rate_samples():
    with pytest.raises(ValueError):
        PlannerConfig(
            grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11, kappa_max=0.2),
            vehicle=VehicleParams(),
            kappa_rate_samples=4,
        )


def test_planner_config_rejects_step_size_larger_than_step_size_max():
    with pytest.raises(ValueError):
        PlannerConfig(
            grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11, kappa_max=0.2),
            vehicle=VehicleParams(),
            step_size=2.0,
            step_policy={"step_size_max": 1.5},
        )


def test_planner_config_syncs_kappa_max_from_grid():
    cfg = PlannerConfig(
        grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11, kappa_max=0.33),
        vehicle=VehicleParams(),
        kappa_max=0.2,
    )
    assert cfg.kappa_max == pytest.approx(0.33)



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
    # TODO: give these more descriptive names for the chosen test parameters for when I add tests with different parameters
    cfg = cfg_input_dir / "experiment.yaml"
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
    cfg = cfg_input_dir / "bad.yaml"
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
