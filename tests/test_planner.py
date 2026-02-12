# tests/test_planner.py

import math
from typing import List, Tuple, Any
import numpy as np

np.set_printoptions(precision=3, suppress=True, threshold=100000)
import pytest
# local module imports
from src.structs import Pose, GoalSpec, GridSpec, PlannerConfig
from src.models import OccupancyGrid
from src.utils import pose_is_free
from src.planners import planner_factory


def _reached(p: Pose, goal: Pose, pos_tol: float, th_tol: float) -> bool:
    dx = p.x - goal.x
    dy = p.y - goal.y
    if dx * dx + dy * dy > pos_tol * pos_tol:
        return False
    dth = (p.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi
    return abs(dth) <= th_tol

@pytest.mark.slow
def test_python_backend_finds_path_on_empty_map(empty_grid: OccupancyGrid, planner_config: PlannerConfig, start_pose: Pose, goal_spec: GoalSpec):
    planner = planner_factory(empty_grid, planner_config, backend="python")
    path, stats = planner.plan(start_pose, goal_spec, max_expansions=50_000)
    assert stats.expanded > 0, "Planner did not expand any nodes"
    assert len(path) > 1, "Planner failed to find a path"
    assert _reached(path[-1], goal_spec.pose, goal_spec.pos_tol, goal_spec.theta_tol), "Final pose does not reach the goal tolerances"
    # basic collision-free check
    for p in path:
        assert pose_is_free(p, empty_grid, planner.footprint_offsets), "Path contains a pose in collision"


def test_python_backend_returns_empty_if_start_in_obstacle(grid_spec: GridSpec, planner_config: PlannerConfig, goal_spec: GoalSpec):
    from src.models import OccupancyGrid
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
    assert _reached(path[-1], goal.pose, goal.pos_tol, goal.theta_tol), "Final pose does not reach the goal tolerances"
    # basic collision-free check
    for p in path:
        assert pose_is_free(p, grid_with_wall, planner.footprint_offsets), "Path contains a pose in collision"


@pytest.mark.slow
def test_python_backend_handles_mazes(maze_grid_and_poses: Tuple[Any, List[float], List[float]], planner_config: PlannerConfig, start_pose: Pose, goal_spec: GoalSpec):
    grid, start, goal = maze_grid_and_poses
    # update start and goal poses with those from the maze file (necessary since Pose dataclasses are frozen)
    # print("start pose values (world frame): ", start)
    # print("start pose values (grid frame): ", grid.world_to_grid(start[0], start[1])[::-1])
    s_pose = Pose(start[0], start[1], start_pose.theta, start_pose.kappa)
    # print("goal pose values (world frame): ", goal)
    # print("goal pose values (grid frame): ", grid.world_to_grid(goal[0], goal[1])[::-1])
    g_pose = Pose(goal[0], goal[1], goal_spec.pose.theta, goal_spec.pose.kappa)
    # create new GoalSpec with updated goal pose
    g_spec = GoalSpec(g_pose, goal_spec.pos_tol, goal_spec.theta_tol, goal_spec.kappa_tol)
    planner = planner_factory(grid, planner_config, backend="python")
    path, stats = planner.plan(s_pose, g_spec, max_expansions=100_000)
    #!!! FIXME: seemingly happens when it starts (but not really) in collision
    assert stats.expanded > 0, "Planner did not expand any nodes"
    assert len(path) > 1, "Planner failed to find a path"
    assert _reached(path[-1], g_spec.pose, g_spec.pos_tol, g_spec.theta_tol), "Final pose does not reach the goal tolerances"
    # basic collision-free check
    #!!! FIXME: we also occasionally get errors here where one part of the path contains obstacles for some reason
        # added verbose flag to check the whole path when it fails
    grid.view_grid(path)
    for p in path:
        assert pose_is_free(p, grid, planner.footprint_offsets), "Path contains a pose in collision"


#^ WARNING: this test may expose shared-state bugs in the planner implementation
    # BUT it's not a final implementation and is mainly meant for testing that there aren't barriers to future concurrent implementations
@pytest.mark.slow
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
    goal = GoalSpec(Pose(12.0, 10.0, 0.0, 0.0), pos_tol=0.8, theta_tol=math.radians(20.0), kappa_tol=0.2)
    h2d = planner._build_goal_heuristics(goal)
    shot = planner._try_goal_shot(start, goal, h2d)
    assert shot is not None and len(shot) > 0
    assert _reached(shot[-1], goal.pose, goal.pos_tol, goal.theta_tol)


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