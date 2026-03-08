# tests/test_heuristics.py
import math
import numpy as np

from dolgov_cbmp.structs import GridSpec, Pose
from dolgov_cbmp.settings import PlannerConfig, VehicleParams
from dolgov_cbmp.utils import compute_distance_to_obstacles_m, compute_gvd_distance_m
from dolgov_cbmp.models import OccupancyGrid, HolonomicWithObstacles2D, NonHolonomicWithoutObstaclesTable


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
    from dolgov_cbmp.settings.config import HeuristicParams
    heuristics = HeuristicParams(
        # keep the nonholonomic table smaller for tests
        nh_table_xy_radius=6.0,
        nh_table_xy_res=1.0,
        nh_table_theta_res=math.radians(15.0),
    )
    cfg = PlannerConfig(
        # grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11),
        vehicle=VehicleParams(),
        curvature={"kappa_rate_samples": 5},
        heuristics=heuristics,
    )
    nh = NonHolonomicWithoutObstaclesTable(cfg)
    nh.build_offline()
    goal = Pose(0.0, 0.0, 0.0)
    assert nh(goal, goal) == 0.0, "Heuristic at goal should be zero"
    far = Pose(100.0, 0.0, 0.0)
    val = nh(far, goal)
    assert abs(val - 100.0) < 1e-6, f"Heuristic at far point should approximate Euclidean distance: {val}"

# heuristic test to verify that the GVD-based heuristic produces a finite field
def test_compute_gvd_distance_returns_finite_field():
    occ = np.zeros((30, 30), dtype=bool)
    occ[10:20, 15] = True
    grid = GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11)
    og = OccupancyGrid(occ, grid)
    dO = compute_distance_to_obstacles_m(og.occ, resolution=1.0)
    dV = compute_gvd_distance_m(dO, resolution=1.0)
    assert dV.shape == dO.shape
    assert np.all(np.isfinite(dV))
    assert float(np.max(dV)) > 0.0

def test_compute_gvd_distance_from_occ():
    occ = np.zeros((30, 30), dtype=bool)
    occ[10:20, 15] = True
    grid = GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11)
    og = OccupancyGrid(occ, grid)
    # using dO just to compare the results
    dV = compute_gvd_distance_m(og.occ, resolution=1.0)
    assert dV.shape == occ.shape
    assert np.all(np.isfinite(dV))
    assert float(np.max(dV)) > 0.0

def test_nonholonomic_table_respects_nh_kappa_bins_override():
    """ test to check that nonholonomic table respects the kappa bins override (important for preventing memory blow-up for high-res grids) """
    from dolgov_cbmp.settings.config import HeuristicParams
    heuristics = HeuristicParams(
        nh_kappa_bins=5,
        nh_table_xy_radius=4.0,
        nh_table_xy_res=1.0,
        nh_table_theta_res=math.radians(20.0),
    )
    cfg = PlannerConfig(
        # grid=GridSpec(resolution=1.0, theta_bins=36, kappa_bins=11),
        vehicle=VehicleParams(),
        heuristics=heuristics,
    )
    nh = NonHolonomicWithoutObstaclesTable(cfg)
    nh.build_offline()
    assert nh._meta is not None
    INDEX_KAPPA_BINS = 6
    assert int(nh._meta[INDEX_KAPPA_BINS]) == 5