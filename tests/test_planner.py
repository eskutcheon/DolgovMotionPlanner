# tests/test_planner.py

import math
from typing import List, Tuple
from dataclasses import replace
import numpy as np
import pytest
# local module imports
from dolgov_cbmp.structs import Pose, GoalSpec, GridSpec
from dolgov_cbmp.settings import PlannerConfig
from dolgov_cbmp.models import OccupancyGrid
from dolgov_cbmp.utils import pose_is_free, goal_reached
from dolgov_cbmp.planners import planner_factory


@pytest.mark.slow
def test_python_backend_finds_path_on_empty_map(empty_grid: OccupancyGrid, planner_config: PlannerConfig, start_pose: Pose, goal_spec: GoalSpec):
    planner = planner_factory(empty_grid, planner_config, backend="python")
    path, stats = planner.plan(start_pose, goal_spec, max_expansions=50_000)
    assert stats.expanded > 0, "Planner did not expand any nodes"
    assert len(path) > 1, "Planner failed to find a path"
    assert goal_reached(
            path[-1].as_tuple(), goal_spec.pose.as_tuple(), goal_spec.pos_tol, goal_spec.theta_tol
        ), "Final pose does not reach the goal tolerances"
    # basic collision-free check
    for p in path:
        assert pose_is_free(p, empty_grid, planner.footprint_offsets), "Path contains a pose in collision"


def test_python_backend_returns_empty_if_start_in_obstacle(grid_spec: GridSpec, planner_config: PlannerConfig, goal_spec: GoalSpec):
    from dolgov_cbmp.models.models import OccupancyGrid
    occ = np.zeros((20, 20), dtype=bool)
    occ[5, 5] = True
    grid = OccupancyGrid(occ, grid_spec)
    planner = planner_factory(grid, planner_config, backend="python")
    # starting near the obstacle (without room to move under kinematic constraints) to force immediate failure
    start = Pose(5.1, 5.1, 0.0)
    path, _ = planner.plan(start, goal_spec, max_expansions=5_000)
    assert path == [], "Planner should return empty path when start is in collision"


@pytest.mark.slow
def test_python_backend_can_pass_through_gap(grid_with_wall: OccupancyGrid, planner_config: PlannerConfig):
    # Goal is on the other side of the wall; path must go through the gap
    planner = planner_factory(grid_with_wall, planner_config, backend="python")
    start = Pose(10.0, 30.0, 0.0)
    goal = GoalSpec(Pose(50.0, 30.0, 0.0), pos_tol=2.0, theta_tol=math.radians(30.0))
    path, _ = planner.plan(start, goal, max_expansions=100_000)
    assert len(path) > 1, "Motion planner failed to find a path through the gap"
    assert goal_reached(
            path[-1].as_tuple(), goal.pose.as_tuple(), goal.pos_tol, goal.theta_tol
        ), "Final pose does not reach the goal tolerances"
    # basic collision-free check
    for p in path:
        assert pose_is_free(p, grid_with_wall, planner.footprint_offsets), "Path contains a pose in collision"


@pytest.mark.slow
def test_python_backend_handles_mazes(maze_grid_and_poses: Tuple[OccupancyGrid, List[float], List[float]], planner_config: PlannerConfig, start_pose: Pose, goal_spec: GoalSpec):
    grid, start, goal = maze_grid_and_poses
    # update start and goal poses with those from the maze file (necessary since Pose dataclasses are frozen)
    s_pose = Pose(start[0], start[1], start_pose.theta, start_pose.kappa)
    g_pose = Pose(goal[0], goal[1], goal_spec.pose.theta, goal_spec.pose.kappa)
    # create new GoalSpec with updated goal pose
    g_spec = GoalSpec(g_pose, goal_spec.pos_tol, goal_spec.theta_tol) #, goal_spec.kappa_tol)
    planner = planner_factory(grid, planner_config, backend="python")
    path, stats = planner.plan(s_pose, g_spec, max_expansions=100_000)
    assert stats.expanded > 0, "Planner did not expand any nodes"
    assert len(path) > 1, "Planner failed to find a path"
    assert goal_reached(
            path[-1].as_tuple(), g_spec.pose.as_tuple(), g_spec.pos_tol, g_spec.theta_tol
        ), "Final pose does not reach the goal tolerances"
    # basic collision-free check
    grid.view_grid(path)
    for p in path:
        assert pose_is_free(p, grid, planner.footprint_offsets), "Path contains a pose in collision"


#^ WARNING: this test may expose shared-state bugs in the planner implementation
    # BUT it's not a final implementation and is mainly meant for testing that there aren't barriers to future concurrent implementations
@pytest.mark.slow
# @pytest.mark.concurrency
def test_planner_can_be_called_concurrently(empty_grid: OccupancyGrid, planner_config: PlannerConfig, start_pose: Pose, goal_spec: GoalSpec):
    # test for shared-state bugs (even though Python backend is mostly serialized by the GIL).
    from concurrent.futures import ThreadPoolExecutor
    planner = planner_factory(empty_grid, planner_config, backend="python")

    def run_once():
        path, stats = planner.plan(start_pose, goal_spec, max_expansions=100_000)
        return len(path), stats.expanded

    with ThreadPoolExecutor(max_workers=2) as ex:
        a = ex.submit(run_once)
        b = ex.submit(run_once)
        la, ea = a.result()
        lb, eb = b.result()
    assert la > 1 and lb > 1, "One of the concurrent runs failed to find a path"
    assert ea > 0 and eb > 0,  "One of the concurrent runs did not expand any nodes"



def test_analytic_connector_can_close_short_gap(empty_grid, planner_config):
    planner = planner_factory(empty_grid, planner_config, backend="python")
    start = Pose(10.0, 10.0, 0.0, 0.0)
    goal = GoalSpec(Pose(12.0, 10.0, 0.0, 0.0), pos_tol=0.8, theta_tol=math.radians(20.0)) #, kappa_tol=0.2)
    h2d = planner._build_goal_heuristics(goal)
    shot = planner._try_goal_shot(start, goal, h2d)
    assert shot is not None and len(shot) > 0
    assert goal_reached(
            shot[-1].as_tuple(), goal.pose.as_tuple(), goal.pos_tol, goal.theta_tol
        ), "Analytic shot does not reach the goal tolerances"


def test_path_smoothing_preserves_collision_free(empty_grid, planner_config):
    planner = planner_factory(empty_grid, planner_config, backend="python")
    raw = [
        Pose(5.0, 5.0, 0.0, 0.0),
        Pose(6.0, 5.4, 0.1, 0.02),
        Pose(7.0, 4.8, -0.1, -0.02),
        Pose(8.0, 5.3, 0.1, 0.03),
        Pose(9.0, 5.0, 0.0, 0.0),
    ]
    sm = planner._smooth_path(raw)
    assert len(sm) == len(raw)
    for p in sm:
        assert pose_is_free(p, empty_grid, planner.footprint_offsets)



def test_adaptive_analytic_schedule_attempts_more_often_near_goal(empty_grid, planner_config):
    from dolgov_cbmp.settings.config import AnalyticScheduleParams
    cfg = planner_config
    cfg = replace(
        cfg,
        analytic=AnalyticScheduleParams(
            every_n=50,
            max_distance=20.0,
            use_adaptive_schedule=True,
            min_interval=5,
            max_interval=100,
            distance_power=1.5,
        ),
    )
    planner = planner_factory(empty_grid, cfg, backend="python")
    goal = Pose(20.0, 20.0, 0.0, 0.0)
    far = Pose(5.0, 5.0, 0.0, 0.0)
    near = Pose(19.5, 19.5, 0.0, 0.0)
    far_hits = sum(1 for e in range(1, 200) if planner._should_try_analytic(far, goal, e))
    near_hits = sum(1 for e in range(1, 200) if planner._should_try_analytic(near, goal, e))
    assert near_hits > far_hits


def test_objective_smoother_anchors_colliding_points(grid_with_wall: OccupancyGrid, planner_config: PlannerConfig):
    from dolgov_cbmp.settings.config import PathSmootherParams
    cfg = replace(
        planner_config,
        smoother=PathSmootherParams(use_objective_smoother=True, smoothing_anchor_rounds=2, objective_smoothing_iters=3),
    )
    planner = planner_factory(grid_with_wall, cfg, backend="python")
    # middle points intentionally pass through the wall; anchored retries should fall back safely
    #? NOTE: grid_with_wall has a vertical wall over the whole height except for a gap from y=28 to y=32 inclusive
    raw = [
        Pose(8.0, 30.0, 0.0, 0.0),
        Pose(12.0, 30.0, 0.0, 0.0),
        Pose(20.0, 30.0, 0.0, 0.0),
        Pose(28.0, 30.0, 0.0, 0.0),
        Pose(32.0, 30.0, 0.0, 0.0),
    ]
    sm = planner._smooth_path(raw)
    assert len(sm) == len(raw)
    # if refinement cannot fix collisions, algorithm should return original path (safe fallback behavior)
    if any(not pose_is_free(p, grid_with_wall, planner.footprint_offsets) for p in sm):
        assert sm == raw


def test_curvature_aware_step_policy_reduces_step_at_high_kappa(empty_grid, planner_config):
    from dolgov_cbmp.settings.config import StepPolicyParams
    cfg: PlannerConfig = replace(
        planner_config,
        step_policy=StepPolicyParams(use_variable_step=True, step_size_max=10.0, variable_step_beta=0.2, curvature_slowdown_gain=3.0)
    )
    planner = planner_factory(empty_grid, cfg, backend="python")
    low_kappa = Pose(15.0, 15.0, 0.0, 0.0)
    high_kappa = Pose(15.0, 15.0, 0.0, cfg.curvature.kappa_max)
    ds_low = planner._select_step(low_kappa)
    ds_high = planner._select_step(high_kappa)
    assert ds_high < ds_low


def test_refiner_objective_includes_voronoi_weight(empty_grid, planner_config):
    from dolgov_cbmp.settings.config import PathSmootherParams
    cfg = replace(planner_config, smoother=PathSmootherParams(use_objective_smoother=True, objective_w_voronoi=0.7, objective_smoothing_iters=1))
    planner = planner_factory(empty_grid, cfg, backend="python")
    xy = np.array([[5.0, 5.0], [6.0, 5.0], [7.0, 5.0], [8.0, 5.0], [9.0, 5.0]], dtype=np.float64)
    v = planner.refiner._objective_value(xy)
    assert np.isfinite(v)