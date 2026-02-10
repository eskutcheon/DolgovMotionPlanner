

import math

import numpy as np

from src.models import (
    HolonomicWithObstacles2D,
    NonHolonomicWithoutObstaclesTable,
    OccupancyGrid,
)
from src.structs import GridSpec, PlannerConfig, VehicleParams, Pose
from src.utils import compute_distance_to_obstacles_m


def test_distance_to_obstacles_zero_on_obstacles():
    occ = np.zeros((10, 10), dtype=bool)
    occ[4, 7] = True
    dist = compute_distance_to_obstacles_m(occ, resolution=1.0)
    assert dist[4, 7] == 0.0
    assert dist[0, 0] >= 0.0


def test_holonomic_heuristic_on_empty_grid_has_reasonable_values():
    grid = GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11)
    occ = np.zeros((20, 20), dtype=bool)
    og = OccupancyGrid(occ, grid)
    h2d = HolonomicWithObstacles2D(og)
    goal = Pose(10.0, 10.0, 0.0, 0.0)
    h2d.compute(goal)
    assert h2d(goal) == 0.0, "Heuristic at goal should be zero"
    # NOTE: Neighbor 1 cell away has distance at most sqrt(2) away since it's 8-connected
    p = Pose(11.0, 10.0, 0.0, 0.0)
    heuristic_val = h2d(p)
    assert np.isfinite(heuristic_val), "Heuristic should be finite on empty grid"
    assert 0.0 < heuristic_val <= 1.0 + 1e-8, f"Heuristic outside of range (0, 1): {heuristic_val}"


def test_nonholonomic_table_zero_at_goal_and_euclidean_far():
    cfg = PlannerConfig(
        grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11),
        vehicle=VehicleParams(),
        # steering_samples=5,
        kappa_rate_samples=5,
        nh_table_xy_radius=6.0,
        nh_table_xy_res=1.0,
        nh_table_theta_res=math.radians(15.0),
    )
    nh = NonHolonomicWithoutObstaclesTable(cfg)
    nh.build_offline()
    goal = Pose(0.0, 0.0, 0.0)
    assert nh(goal, goal) == 0.0, "Heuristic at goal should be zero"
    far = Pose(100.0, 0.0, 0.0)
    val = nh(far, goal)
    assert abs(val - 100.0) < 1e-6, f"Heuristic at far point should approximate Euclidean distance: {val}"
