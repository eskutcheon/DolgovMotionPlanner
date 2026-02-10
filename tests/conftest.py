
import sys
from typing import Tuple, List, Optional, Any
from pathlib import Path
import numpy as np
import pytest


# ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def grid_spec():
    from src.structs import GridSpec
    return GridSpec(resolution=1.0, theta_bins=36, origin_xy=(0.0, 0.0), kappa_bins=11)


@pytest.fixture
def vehicle_params():
    from src.structs import VehicleParams
    # defaults are fine for most tests
    return VehicleParams()


@pytest.fixture
def planner_config(grid_spec, vehicle_params):
    from src.structs import PlannerConfig
    return PlannerConfig(
        grid=grid_spec,
        vehicle=vehicle_params,
        step_size=1.0,
        n_substeps=5,
        # steering_samples=7,
        kappa_rate_samples=3,
        allow_reverse=True,
        # keep the nonholonomic table smaller for tests
        nh_table_xy_radius=8.0,
        nh_table_xy_res=1.0,
        nh_table_theta_res=np.deg2rad(10.0),
        footprint_sample_step=None,
    )


@pytest.fixture
def empty_grid(grid_spec):
    from src.models import OccupancyGrid
    occ = np.zeros((60, 60), dtype=bool)
    return OccupancyGrid(occ, grid_spec)


@pytest.fixture
def grid_with_wall(grid_spec):
    """ A grid with a vertical wall and a gap. """
    from src.models import OccupancyGrid
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


def get_random_maze_file():
    return np.random.choice(list((Path(__file__).parent / "grids").glob("*.npz")))

@pytest.fixture
def easy_maze_file():
    """ A fixture that provides a path to a random maze file from the grids directory. """
    return r"tests/grids/5x5_square_res100.npz"
    # return Path(r"tests/grids/5x5_slanted2_res100.npz")



@pytest.fixture
def maze_grid_and_poses(grid_spec, easy_maze_file) -> Tuple[Any, List[float], List[float]]:
    """ A grid with predefined obstacles for deterministic tests. """
    from src.models import OccupancyGrid
    maze_file = get_random_maze_file()
    # maze_file = easy_maze_file
    print("Loading maze grid from file:", maze_file)
    occ_grid, start, goal = OccupancyGrid.grid_from_file(maze_file, grid_spec) #, pad_cells=2)
    return occ_grid, start, goal


@pytest.fixture
def start_pose():
    from src.structs import Pose
    return Pose(5.0, 5.0, 0.0)


@pytest.fixture
def goal_pose():
    from src.structs import Pose
    return Pose(50.0, 50.0, np.deg2rad(90.0))


@pytest.fixture
def goal_spec(goal_pose):
    from src.structs import GoalSpec
    return GoalSpec(goal_pose, pos_tol=2.0, theta_tol=np.deg2rad(20.0))


@pytest.fixture
def cpp_available() -> bool:
    try:
        from src.cpp_kernels import CPP_AVAILABLE
        return bool(CPP_AVAILABLE)
    except Exception:
        return False