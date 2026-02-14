""" Thin wrapper around the optional `hybrid_core` C++ extension.
    - This file is intentionally small so the rest of the codebase doesn't have pybind11-specific types all over the place
    - The extension is expected to expose:
        - run_search(...): full Hybrid A* search kernel
        - expand_primitives(...): optional debugging helper
    - If the extension isn't built, importers can fall back to the pure-Python backend.
"""

from typing import List, Optional, Tuple
import numpy as np
# local module imports
from src.structs import Pose, GoalSpec, PlannerConfig, PlannerStats
from src.models import OccupancyGrid

try:
    import hybrid_core  # type: ignore
    CPP_AVAILABLE = True
except Exception:  # pragma: no cover
    CPP_AVAILABLE = False
    hybrid_core = None  # type: ignore


def run_search_cpp(
    *,
    occ_grid: OccupancyGrid,
    cfg: PlannerConfig,
    start: Pose,
    goal: GoalSpec,
    steer_set: np.ndarray,
    h2d_dist: np.ndarray,
    nonhol_table: np.ndarray,
    nonhol_meta: Tuple[float, float, float, int, int],
    rho: Optional[np.ndarray],
    footprint_offsets: Optional[np.ndarray],
    max_expansions: int,
) -> Tuple[List[Pose], PlannerStats]:
    """ Run the full Hybrid A* kernel in C++. Returns a (path, stats) pair """
    if not CPP_AVAILABLE:
        raise RuntimeError("C++ backend not available (failed to import hybrid_core)")
    # Ensure contiguous arrays for safe buffer access in C++.
    occ_u8 = np.ascontiguousarray(occ_grid.occ.astype(np.uint8, copy=False))
    steer = np.ascontiguousarray(steer_set, dtype=np.float64)
    h2d = np.ascontiguousarray(h2d_dist, dtype=np.float64)
    nh = np.ascontiguousarray(nonhol_table, dtype=np.float64)
    if rho is None:
        rho_in = np.zeros((1, 1), dtype=np.float64)  # sentinel
        use_rho = False
    else:
        rho_in = np.ascontiguousarray(rho, dtype=np.float64)
        use_rho = True
    if footprint_offsets is None:
        fp = np.zeros((0, 2), dtype=np.float64)  # sentinel
        use_fp = False
    else:
        fp = np.ascontiguousarray(footprint_offsets, dtype=np.float64)
        use_fp = True
    ox, oy = occ_grid.grid.origin_xy
    res = float(occ_grid.grid.resolution)
    theta_bins = int(occ_grid.grid.theta_bins)
    nh_R, nh_res, nh_dth, nh_nxy, nh_theta_bins = nonhol_meta
    out = hybrid_core.run_search(
        float(start.x), float(start.y), float(start.theta),
        float(goal.pose.x), float(goal.pose.y), float(goal.pose.theta),
        float(goal.pos_tol), float(goal.theta_tol),
        int(max_expansions),
        occ_u8,
        float(ox), float(oy), float(res), int(theta_bins),
        steer,
        bool(cfg.allow_reverse),
        float(cfg.step_size), int(cfg.n_substeps),
        float(cfg.vehicle.wheelbase),
        float(cfg.weights.reverse_penalty),
        float(cfg.weights.switch_dir_penalty),
        # heuristics
        h2d,
        nh,
        float(nh_R), float(nh_res), float(nh_dth), int(nh_nxy), int(nh_theta_bins),
        # optional edge shaping
        bool(use_rho),
        rho_in,
        float(cfg.weights.voronoi_weight),
        # footprint
        bool(use_fp),
        fp,
    )

    path_arr = np.asarray(out["path"], dtype=np.float64)
    path: List[Pose] = [Pose(float(x), float(y), float(th)) for x, y, th in path_arr]

    s = out["stats"]
    stats = PlannerStats(
        expanded=int(s.get("expanded", 0)),
        pushed=int(s.get("pushed", 0)),
        collision_checks=int(s.get("collision_checks", 0)),
        analytic_attempts=int(s.get("analytic_attempts", 0)),
        analytic_successes=int(s.get("analytic_successes", 0)),
        start_time_s=0.0,
        end_time_s=0.0,
    )

    return path, stats
