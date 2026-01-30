# tests/test_planner.py

import math
import numpy as np
import pytest
# local module imports
from models import OccupancyGrid, pose_is_free
from planners import planner_factory
from structs import Pose


def _reached(p: Pose, goal: Pose, pos_tol: float, th_tol: float) -> bool:
    dx = p.x - goal.x
    dy = p.y - goal.y
    if dx * dx + dy * dy > pos_tol * pos_tol:
        return False
    dth = (p.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi
    return abs(dth) <= th_tol

#^ WARNING: one of the slowest (if not THE slowest test) in the suite - worst case needs a lot of improvement
def test_python_backend_finds_path_on_empty_map(empty_grid, planner_config, start_pose, goal_spec):
    planner = planner_factory(empty_grid, planner_config, backend="python")
    path, stats = planner.plan(start_pose, goal_spec, max_expansions=30_000)
    assert stats.expanded > 0
    assert len(path) > 1
    assert _reached(path[-1], goal_spec.pose, goal_spec.pos_tol, goal_spec.theta_tol)
    # basic collision-free check
    for p in path:
        assert pose_is_free(p, empty_grid, planner.footprint_offsets)


def test_python_backend_returns_empty_if_start_in_obstacle(grid_spec, planner_config, goal_spec):
    occ = np.zeros((20, 20), dtype=np.bool_)
    occ[5, 5] = True
    grid = OccupancyGrid(occ, grid_spec)
    planner = planner_factory(grid, planner_config, backend="python")
    # starting near the obstacle (without room to move under kinematic constraints) to force immediate failure
    start = Pose(5.1, 5.1, 0.0)
    path, _ = planner.plan(start, goal_spec, max_expansions=5_000)
    assert path == []


@pytest.mark.slow
def test_python_backend_can_pass_through_gap(grid_with_wall, planner_config):
    # Goal is on the other side of the wall; path must go through the gap
    from structs import GoalSpec

    planner = planner_factory(grid_with_wall, planner_config, backend="python")
    start = Pose(10.0, 30.0, 0.0)
    goal = GoalSpec(Pose(50.0, 30.0, 0.0), pos_tol=2.0, theta_tol=math.radians(30.0))
    path, _ = planner.plan(start, goal, max_expansions=80_000)
    assert len(path) > 1
    assert _reached(path[-1], goal.pose, goal.pos_tol, goal.theta_tol)
    # basic collision-free check
    for p in path:
        assert pose_is_free(p, grid_with_wall, planner.footprint_offsets)

