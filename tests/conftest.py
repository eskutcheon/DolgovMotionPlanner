
import sys
from pathlib import Path
import numpy as np
import pytest


# ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def grid_spec():
    from structs import GridSpec
    return GridSpec(resolution=1.0, theta_bins=72, origin_xy=(0.0, 0.0), kappa_bins=36)


@pytest.fixture
def vehicle_params():
    from structs import VehicleParams
    # defaults are fine for most tests
    return VehicleParams()


@pytest.fixture
def planner_config(grid_spec, vehicle_params):
    from structs import PlannerConfig
    return PlannerConfig(
        grid=grid_spec,
        vehicle=vehicle_params,
        step_size=1.0,
        n_substeps=5,
        # steering_samples=7,
        kappa_rate_samples=10,
        allow_reverse=True,
        # keep the nonholonomic table smaller for tests
        nonholonomic_table_xy_radius=8.0,
        nonholonomic_table_xy_res=1.0,
        nonholonomic_table_theta_res=np.deg2rad(10.0),
        footprint_sample_step=None,
    )


@pytest.fixture
def empty_grid(grid_spec):
    from models import OccupancyGrid
    occ = np.zeros((60, 60), dtype=bool)
    return OccupancyGrid(occ, grid_spec)


@pytest.fixture
def grid_with_wall(grid_spec):
    """ A grid with a vertical wall and a gap. """
    from models import OccupancyGrid
    occ = np.zeros((60, 60), dtype=bool)
    # Wall at x=30 with a gap at y in [28,32]
    occ[:, 30] = True
    occ[28:33, 30] = False
    return OccupancyGrid(occ, grid_spec)


# @pytest.fixture
# def grid_with_obstacles(grid_spec):
#     """ A grid with random obstacles. """
#     from models import OccupancyGrid
#     from utils import generate_random_maze_grid
#     occ = generate_random_maze_grid(80, 80, obstacle_prob=0.1, seed=42)
#     return OccupancyGrid(occ, grid_spec)

@pytest.fixture
def maze_grid_file():
    return np.random.choice(list((Path(__file__).parent / "grids").glob("*.npz")))


@pytest.fixture
def states_from_maze(grid_spec, maze_grid_file):
    """ A grid with predefined obstacles for deterministic tests. """
    from models import OccupancyGrid
    # occ = np.zeros((60, 60), dtype=bool)
    # # Add some obstacles
    # occ[10:15, 10:15] = True
    # occ[20:25, 40:45] = True
    # occ[35:40, 20:25] = True
    # occ[45:50, 50:55] = True
    # return OccupancyGrid(occ, grid_spec)
    occ_grid, start, goal = OccupancyGrid.grid_from_file(maze_grid_file, grid_spec)
    return occ_grid, start, goal


@pytest.fixture
def start_pose():
    from structs import Pose
    return Pose(5.0, 5.0, 0.0)


@pytest.fixture
def goal_pose():
    from structs import Pose
    return Pose(50.0, 50.0, np.deg2rad(90.0))


@pytest.fixture
def goal_spec(goal_pose):
    from structs import GoalSpec
    return GoalSpec(goal_pose, pos_tol=2.0, theta_tol=np.deg2rad(20.0))


@pytest.fixture
def cpp_available() -> bool:
    try:
        from cpp_kernels import CPP_AVAILABLE
        return bool(CPP_AVAILABLE)
    except Exception:
        return False