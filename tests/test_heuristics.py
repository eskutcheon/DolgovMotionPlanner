

import math

import numpy as np

from models import (
    HolonomicWithObstacles2D,
    NonHolonomicWithoutObstaclesTable,
    OccupancyGrid,
    compute_distance_to_obstacles_m,
)
from structs import GridSpec, PlannerConfig, Pose, VehicleParams


def test_distance_to_obstacles_zero_on_obstacles():
    occ = np.zeros((10, 10), dtype=np.bool_)
    occ[4, 7] = True
    dist = compute_distance_to_obstacles_m(occ, resolution=1.0)
    assert dist[4, 7] == 0.0
    assert dist[0, 0] >= 0.0


def test_holonomic_heuristic_on_empty_grid_has_reasonable_values():
    grid = GridSpec(resolution=1.0, theta_bins=72)
    occ = np.zeros((20, 20), dtype=np.bool_)
    og = OccupancyGrid(occ, grid)

    h2d = HolonomicWithObstacles2D(og)
    goal = Pose(10.0, 10.0, 0.0)
    h2d.compute(goal)

    assert h2d(goal) == 0.0
    # Neighbor 1 cell away has at most sqrt(2) away since 8-connected.
    p = Pose(11.0, 10.0, 0.0)
    assert 0.0 < h2d(p) <= 1.0 + 1e-9


def test_nonholonomic_table_zero_at_goal_and_euclidean_far():
    cfg = PlannerConfig(
        grid=GridSpec(resolution=1.0, theta_bins=72),
        vehicle=VehicleParams(),
        steering_samples=5,
        nonholonomic_table_xy_radius=6.0,
        nonholonomic_table_xy_res=1.0,
        nonholonomic_table_theta_res=math.radians(15.0),
    )

    nh = NonHolonomicWithoutObstaclesTable(cfg)
    nh.build_offline()

    goal = Pose(0.0, 0.0, 0.0)
    assert nh(goal, goal) == 0.0

    far = Pose(100.0, 0.0, 0.0)
    val = nh(far, goal)
    assert abs(val - 100.0) < 1e-6

