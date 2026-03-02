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

# import math
import numpy as np
# local module imports
# from src.structs import GridSpec, VehicleParams, Pose, GoalSpec, PlannerConfig
from src.settings import parse_planning_inputs
from src.models import OccupancyGrid
from src.planners import planner_factory



def plan_example() -> None:
    args = parse_planning_inputs()
    # Toy map (mostly empty) with a thin vertical obstacle.
    occ = np.zeros((200, 200), dtype=bool)
    occ[80:120, 100] = True
    og = OccupancyGrid(occ, args.planner_config.grid)
    print("grid created:", args.planner_config.grid)
    # print("planner config:", args.planner_config)
    print("start:", args.start)
    print("goal:", args.goal)
    planner = planner_factory(og, args.planner_config, backend=args.backend)
    print("planner created with backend:", args.backend)
    path, stats = planner.plan(args.start, args.goal, max_expansions=args.max_expansions)
    print("planning completed.")
    print(
        f"backend={args.backend} | path poses={len(path)} | expanded={stats.expanded} | "
        f"pushed={stats.pushed} | elapsed={stats.elapsed_s():.3f}s"
    )


if __name__ == "__main__":
    plan_example()
