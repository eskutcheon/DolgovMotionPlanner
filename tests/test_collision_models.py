
import numpy as np

from dolgov_cbmp.utils import make_rectangle_footprint_offsets, pose_is_free
from dolgov_cbmp.structs import GridSpec, Pose
from dolgov_cbmp.settings import VehicleParams
from dolgov_cbmp.models import OccupancyGrid


def test_world_grid_roundtrip():
    grid = GridSpec(resolution=1.0, theta_bins=72, origin_xy=(0.0, 0.0))
    occ = np.zeros((10, 10), dtype=bool)
    og = OccupancyGrid(occ, grid)
    # cell center
    x, y = og.grid_to_world(3, 4)
    ix, iy = og.world_to_grid(x, y)
    assert (ix, iy) == (3, 4)


def test_out_of_bounds_is_occupied():
    grid = GridSpec(resolution=1.0, theta_bins=72, origin_xy=(0.0, 0.0))
    occ = np.zeros((5, 5), dtype=bool)
    og = OccupancyGrid(occ, grid)
    assert og.is_occupied(5, 0) is True
    assert og.is_occupied(0, 5) is True


def test_make_rectangle_footprint_offsets_shape_and_deterministic():
    V = VehicleParams()
    pts = make_rectangle_footprint_offsets(V.wheelbase, V.width, V.front_overhang, V.rear_overhang, sample_step=1.0)
    assert pts.ndim == 2
    assert pts.shape[1] == 2
    # deterministic order / contiguous
    assert pts.flags['C_CONTIGUOUS']


def test_pose_is_free_reference_point_only():
    grid = GridSpec(resolution=1.0, theta_bins=72)
    occ = np.zeros((10, 10), dtype=bool)
    occ[4, 3] = True
    og = OccupancyGrid(occ, grid)
    p_free = Pose(1.0, 1.0, 0.0)
    p_hit = Pose(3.1, 4.2, 0.0)
    assert pose_is_free(p_free, og, None) is True
    assert pose_is_free(p_hit, og, None) is False


def test_pose_is_free_with_rectangle_sampling_hits_obstacle():
    V = VehicleParams(width=2.0, front_overhang=1.0, rear_overhang=1.0, wheelbase=2.0)
    offsets = make_rectangle_footprint_offsets(V.wheelbase, V.width, V.front_overhang, V.rear_overhang, sample_step=0.5)
    grid = GridSpec(resolution=1.0, theta_bins=72)
    occ = np.zeros((20, 20), dtype=bool)
    og = OccupancyGrid(occ, grid)
    p = Pose(10.0, 10.0, 0.0)
    # Sample a footprint point that is "most forward" in +x for theta=0 for obstacle near the front of the vehicle centered at (10,10)
    front = offsets[np.argmax(offsets[:, 0])]
    ox, oy = float(front[0]), float(front[1])
    ix, iy = og.world_to_grid(p.x + ox, p.y + oy)
    occ[iy, ix] = True  # place obstacle exactly under a sampled footprint point
    assert pose_is_free(p, og, offsets) is False

