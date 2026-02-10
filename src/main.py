"""
    initial Hybrid A* implementation inspired by:
        D. Dolgov et al., "Practical Search Techniques in Path Planning for Autonomous Driving" (AAAI 2008).

    Key ideas implemented:
        - Hybrid-state A*: discretize (x,y,theta) but store continuous pose per visited discrete node
        - Two heuristics:
            (1) non-holonomic-without-obstacles - pre-computable offline, queried in goal frame
            (2) holonomic-with-obstacles via 2D dynamic programming / Dijkstra on the grid
    Use h = max(h_nonhol, h_holonomic) as in the paper
        - Optional analytic expansion hook (e.g., Reeds-Shepp), collision-checked against the map
"""

import math
import numpy as np
# local module imports
from src.structs import GridSpec, VehicleParams, Pose, GoalSpec, PlannerConfig #, PlannerStats, PlannerTick, TickCallback
from src.models import OccupancyGrid
from src.planners import planner_factory



def plan_example(backend: str = "python") -> None:
    # Toy map (mostly empty) with a thin vertical obstacle.
    occ = np.zeros((200, 200), dtype=bool)
    occ[80:120, 100] = True
    grid = GridSpec(resolution=0.5, theta_bins=36, origin_xy=(0.0, 0.0), kappa_bins=11)
    og = OccupancyGrid(occ, grid)
    print("grid created: ", grid)
    cfg = PlannerConfig(grid=grid, vehicle=VehicleParams())
    print("planner config: ", cfg)
    planner = planner_factory(og, cfg, backend=backend)
    print("planner created with backend:", backend)
    start = Pose(5.0, 5.0, math.radians(0.0), 0.0)
    goal = GoalSpec(Pose(80.0, 80.0, math.radians(90.0), 0.0))
    path, stats = planner.plan(start, goal, max_expansions=100_000)
    print("planning completed.")
    print(
        f"backend={backend} | path poses={len(path)} | expanded={stats.expanded} | "
        f"pushed={stats.pushed} | elapsed={stats.elapsed_s():.3f}s"
    )


if __name__ == "__main__":
    plan_example()
