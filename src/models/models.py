# src/models/models.py
from typing import List, Optional, Tuple, Sequence, Dict
import math
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=100000)
from pathlib import Path

from src.structs import GridSpec, VehicleParams, Pose, DiscreteKey
from src.utils import TAU, wrap_angle, pose_is_free, pose_is_free_cached_cells, kappa_to_bin, theta_to_bin



# Occupancy + distance field
class OccupancyGrid:
    """ Binary occupancy grid in row-major (iy, ix) order - `occ[y, x] == True` means obstacle.
        Important for multi-threading later: treat `occ` as read-only during planning
    """
    def __init__(self, occupied: np.ndarray, grid: GridSpec):
        if occupied.dtype not in (bool, np.bool_):
            occupied = occupied.astype(bool)
        self.occ: np.ndarray = occupied
        self.grid: GridSpec = grid
        self.height, self.width = occupied.shape

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height

    def pose_from_cell(self, ix: int, iy: int, theta: float = 0.0, kappa: float = 0.0) -> Pose:
        """ helper method to create a Pose at the cell center (ix,iy) """
        x, y = self.grid_to_world(ix, iy)
        return Pose(x=x, y=y, theta=float(theta), kappa=float(kappa))

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        ox, oy = self.grid.origin_xy
        ix = int(math.floor((x - ox) / self.grid.resolution))
        iy = int(math.floor((y - oy) / self.grid.resolution))
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """ Return center of cell (ix, iy) in world coordinates """
        ox, oy = self.grid.origin_xy
        x = ox + (ix + 0.5) * self.grid.resolution
        y = oy + (iy + 0.5) * self.grid.resolution
        return x, y

    def is_occupied(self, ix: int, iy: int) -> bool:
        if not self.in_bounds(ix, iy):
            return True  # treat out-of-bounds as obstacles
        return bool(self.occ[iy, ix])

    def _sanity_check_poses(self, sx: int, sy: int, gx: int, gy: int, err_msg: str) -> None:
        if not self.in_bounds(sx, sy) or self.is_occupied(sx, sy):
            raise ValueError(f"Start cell {err_msg}")
        if not self.in_bounds(gx, gy) or self.is_occupied(gx, gy):
            raise ValueError(f"Goal cell {err_msg}")

    @staticmethod
    def pad_grid(occ: np.ndarray, grid: GridSpec, pad_cells: int = 1) -> Tuple[np.ndarray, GridSpec]:
        """ Pad the occupancy grid by a certain number of cells using reflection padding. """
        occ = np.pad(occ, pad_width=int(pad_cells), mode='reflect') #mode="constant", constant_values=True)
        ox, oy = grid.origin_xy
        r = float(grid.resolution)
        gridspec_kwargs = grid.to_dict()
        gridspec_kwargs['origin_xy'] = (ox - pad_cells * r, oy - pad_cells * r)
        grid = GridSpec(**gridspec_kwargs)
        print("Applied padding of", pad_cells, "cells; new shape:", occ.shape)
        return occ, grid

    @staticmethod
    def grid_from_file(
        path: str | Path, grid: GridSpec, poses_kind: str = "auto", pad_cells: int = 0
    ) -> Tuple['OccupancyGrid', List[float], List[float]]:
        """ Load a binary occupancy grid and start/goal indices/coordinates in the grid from a .npz file
            Expected keys:
            - occupancy: HxW bool array
            - poses: shape (2,2), either [[ix,iy],[ix,iy]] or [[x,y],[x,y]]
            - optional: poses_kind: "grid" or "world"
        """
        if not isinstance(path, Path):
            path = Path(path)
        grid_dict: Dict[str, np.ndarray] = np.load(path)
        occ = grid_dict['occupancy'].astype(bool, copy=False)
        assert occ.ndim == 2, "Occupancy grid must be 2D"
        poses = grid_dict['poses']
        assert poses.shape == (2, 2), "Poses array must have shape (2,2)"
        print("\nLoaded occupancy grid from", path, f"with shape {occ.shape} and dtype {occ.dtype}")
        # Resolve pose convention
        assert poses_kind in ("auto", "grid", "world"), f"Invalid poses_kind={poses_kind!r}"
        if poses_kind == "auto":
            # if poses are integer-like and inside bounds, treat as grid indices (cell centers)
            int_like = np.array_equal(poses, np.round(poses))
            in_bounds = (
                np.all(poses[:, 0] >= 0) and np.all(poses[:, 1] >= 0) and
                np.all(poses[:, 0] < occ.shape[1]) and np.all(poses[:, 1] < occ.shape[0])
            )
            poses_kind = "grid" if (int_like and in_bounds) else "world"
        # Optional padding: if we add padding, we MUST shift the origin accordingly
        if pad_cells > 0:
            occ, grid = OccupancyGrid.pad_grid(occ, grid, pad_cells=pad_cells)
        # create OccupancyGrid and set start/goal accordingly
        occ_grid = OccupancyGrid(occ, grid)
        start_xy: List[float]
        goal_xy: List[float]
        if poses_kind == "grid":
            sx, sy, gx, gy = [int(round(float(poses[i, j]))) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
            occ_grid._sanity_check_poses(sx, sy, gx, gy, "is out of bounds or occupied")
            start_xy, goal_xy = occ_grid.grid_to_world(sx, sy), occ_grid.grid_to_world(gx, gy)
        else:
            sx, sy, gx, gy = [float(poses[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
            s_ix, s_iy = occ_grid.world_to_grid(sx, sy)
            g_ix, g_iy = occ_grid.world_to_grid(gx, gy)
            occ_grid._sanity_check_poses(s_ix, s_iy, g_ix, g_iy, "world coordinates map to out-of-bounds or occupied cell")
            start_xy, goal_xy = [sx, sy], [gx, gy]
        return occ_grid, start_xy, goal_xy


    def view_grid(self, path: Optional[List[Pose]] = None):
        """ print grid with two different markers for free space and obstacles; optionally overlay a path """
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        RED = '\033[91m'
        grid_display = np.full(self.occ.shape, '.', dtype=str)
        grid_display[self.occ] = '#'
        grid_display = grid_display.tolist()
        if path is not None:
            for p in path:
                ix, iy = self.world_to_grid(p.x, p.y)
                #? NOTE: doesn't count out-of-bounds entries as collisions, unlike elsewhere in the code
                if 0 <= ix < self.occ.shape[1] and 0 <= iy < self.occ.shape[0]:
                    # mark collisions in red and other path poses in yellow
                    marker = RED + 'X' + RESET if self.occ[iy, ix] else YELLOW + 'o' + RESET
                    grid_display[iy][ix] = marker
        # print(grid_display)
        for row in grid_display:
            print("".join(row))
        print() # newline after grid for readability

    def adjust_start_pose_for_clearance(self, start: Pose) -> Pose:
        """ if starting point is in obstacle, jitter to nearest neighbors until free """
        x, y = start.x, start.y
        ix, iy = self.world_to_grid(x, y)
        tried = set()
        while self.is_occupied(ix, iy):
            # if starting point is in obstacle, try nearest neighbors until free
            directions = [(0,1), (1,0), (0,-1), (-1,0)]
            dir_indices = np.random.permutation(len(directions))
            for idx in dir_indices:
                dx, dy = directions[idx]
                if (ix + dx, iy + dy) in tried:
                    continue
                ix += dx
                iy += dy
                tried.add((ix, iy))
                break
        new_x, new_y = self.grid_to_world(ix, iy)
        return Pose(new_x, new_y, start.theta, start.kappa)


# Voronoi Field (paper Eq. 1)
class VoronoiField:
    """ $rho_V(x,y)$ shaped by distance-to-obstacles (and optionally GVD distance).
        This follows the same structure as Dolgov et al. (AAAI 2008):
            - $dO(x,y)$: distance to nearest obstacle
            - $dV(x,y)$: distance to generalized Voronoi diagram (optional)
        If $dV$ is not available, we use the simple proxy $dV := dO$
    """

    def __init__(self, dO_m: np.ndarray, alpha: float, dO_max: float, dV_m: Optional[np.ndarray] = None):
        # clamp to avoid division-by-zero; also means that when very close to obstacles, the potential will be near 1.0 as expected
        self.dO = np.maximum(dO_m.astype(np.float64, copy=False), 1e-9)
        self.dV = np.maximum(dV_m.astype(np.float64, copy=False), 1e-9) if dV_m is not None else None
        self.alpha = float(alpha)
        self.dO_max = float(dO_max)

    @property
    def rho(self) -> np.ndarray:
        """ Vectorized potential in [0,1] on the grid.
            NOTE: If $dV$ is not available, proxy it with $dV := dO$ (keeps a weak "skeleton-ish" scaling effect)
        """
        alpha = self.alpha
        dO_max = self.dO_max
        dV = self.dV if self.dV is not None else self.dO
        # Voronoi field (scaled by obstacle distance and GVD distance) w/ standard convention rho=0 for dO >= dO_max
        rho = np.zeros_like(self.dO, dtype=np.float64)
        mask = self.dO <= dO_max
        if np.any(mask):
            t1 = alpha / (alpha + self.dO)
            t2 = dV / (self.dO + dV)
            t3 = ((self.dO - dO_max) ** 2) / (dO_max ** 2) #? NOTE: (drives to 0 at dO=dO_max)
            # broadcasted multiplication - only compute where dO <= dO_max since otherwise rho=0
            rho[mask] = (t1 * t2 * t3)[mask]
        # # potential modeled after Eq. (1) in the 2008 paper - clipped to [0,1]
        # a = np.clip(dO / dO_max, 0.0, 1.0)
        # b = 1.0 - np.clip(dV / dO_max, 0.0, 1.0)
        # rho = (a**alpha) * b
        # clamp numeric noise
        return np.clip(rho, 0.0, 1.0).astype(np.float64, copy=False)


#? NOTE: both heuristic model classes have repeated use of some of the same discretization methods and may as well use a shared Indexer class
class Indexer:
    """ Discretizes (x,y,theta,dir) and provides a flat index for best-g arrays """
    def __init__(self, grid: OccupancyGrid, kappa_bins: Optional[int] = None, kappa_max: Optional[float] = None):
        self.grid = grid
        self.W = int(grid.width)
        self.H = int(grid.height)
        self.theta_bins = int(grid.grid.theta_bins)
        self.dtheta = TAU / float(self.theta_bins)
        self.kappa_bins = int(kappa_bins or grid.grid.kappa_bins)
        self.kappa_max = float(kappa_max or grid.grid.kappa_max)
        self.kappa_min = -self.kappa_max        # placeholder; should be set from PlannerConfig
        self.dkappa = (self.kappa_max - self.kappa_min) / self.kappa_bins

    def pose_to_key(self, pose: Pose, direction: int) -> DiscreteKey:
        """ Discretize a continuous pose into a DiscreteKey
            Args:
                pose: continuous pose in world frame
                direction: last motion mode (+1 forward, -1 reverse) used to arrive at this pose
        """
        ix, iy = self.grid.world_to_grid(pose.x, pose.y)
        itheta = theta_to_bin(pose.theta, self.theta_bins, self.dtheta)
        ikappa = kappa_to_bin(pose.kappa, self.kappa_min, self.kappa_max, self.dkappa, self.kappa_bins)
        d = 1 if direction >= 0 else -1
        return DiscreteKey(ix=ix, iy=iy, itheta=itheta, ikappa=ikappa, direction=d)

    def key_to_flat(self, key: DiscreteKey) -> int:
        """ return flat packed index for best-g arrays """
        dir_idx = 0 if int(key.direction) >= 0 else 1
        return int((((key.iy * self.W + key.ix) * self.theta_bins + key.itheta) * self.kappa_bins + key.ikappa) * 2 + dir_idx)

    def get_total_states(self) -> int:
        """ total number of discrete states in the grid (for dense best-g) """
        return self.W * self.H * self.theta_bins * self.kappa_bins * 2 # factor of 2 for direction


# ----------------------------
# Motion model (constant curvature bicycle)
# ----------------------------

class BicycleModel:
    def __init__(self, vehicle: VehicleParams):
        self.L = float(vehicle.wheelbase)

    # propagation approach uses curvature instead of steering angle
    def propagate(self, pose: Pose, u: float, direction: int, ds: float, kappa_max: float) -> Pose:
        r""" Propagate the bicycle model for distance $ds$ with curvature-rate $u$ and direction (+1 forward, -1 reverse)
            new system parameters:
                $u  =  d\kappa / ds$
                $d\theta / ds  =  \sigma \kappa$
                $dx / ds  =  \sigma \cos(\theta)$
                $dy / ds  =  \sigma \sin(\theta)$
        """
        sigma = 1.0 if direction >= 0 else -1.0
        x0, y0, th0 = pose.x, pose.y, pose.theta
        # kappa = math.tan(float(steer)) / self.L
        k0 = float(pose.kappa)
        k1 = max(-kappa_max, min(kappa_max, k0 + float(u) * float(ds)))
        # midpoint integration (stable and cheap)
        # TODO: explore other numerical integration methods like trapezoidal, RK4, or modern quadrature methods later
        km = 0.5 * (k0 + k1)
        thm = th0 + 0.5 * sigma * km * ds
        x1 = x0 + sigma * ds * math.cos(thm)
        y1 = y0 + sigma * ds * math.sin(thm)
        th1 = wrap_angle(th0 + sigma * km * ds)
        return Pose(x=x1, y=y1, theta=th1, kappa=k1)

    # @staticmethod
    # def propagate_const_kappa(pose: Pose, kappa: float, direction: int, ds: float, kappa_max = None) -> Pose:
    #     """ Propagate the bicycle model for distance $ds$ with constant curvature $kappa$ and direction (+1 forward, -1 reverse) """
    #     sigma = 1.0 if direction >= 0 else -1.0
    #     x0, y0, th0 = pose.x, pose.y, pose.theta
    #     # midpoint integration (same style as BicycleModel.propagate)
    #     thm = th0 + 0.5 * sigma * kappa * ds
    #     x1 = x0 + sigma * ds * math.cos(thm)
    #     y1 = y0 + sigma * ds * math.sin(thm)
    #     th1 = wrap_angle(th0 + sigma * kappa * ds)
    #     return Pose(x=x1, y=y1, theta=th1, kappa=kappa)

    # for step selection and collision checking - for now it needs access to the footprint cache and distance field on the planner
    @staticmethod
    def pose_is_free_fast(
        pose: Pose,
        grid: OccupancyGrid,
        footprint_offsets: Optional[np.ndarray],
        dO: float,
        gate_radius_m: float = 0.0,
        exact_check_margin_m: float = 0.0,
        footprint_cache: Optional[Sequence[np.ndarray]] = None,
        theta_bins: int = 0,
    ) -> bool:
        """ Use distance-to-obstacle gating + cached footprint when possible; fall back to exact footprint near obstacles
            - if (dO >= gate_radius + exact_margin) => accept (reference point has ample clearance)
            - if (gate_radius <= dO < gate_radius + exact_margin) and cache available => cached footprint cells
            - if (dO < gate_radius) or cache missing => exact footprint sampling
            NOTE: If gate_radius_m <= 0, gating is disabled and we always do the exact check.
        """
        # if gate radius checking is disabled, always do exact checks
        if gate_radius_m <= 0.0:
            return pose_is_free(pose, grid, footprint_offsets)
        gate = float(gate_radius_m)
        margin = max(0.0, float(exact_check_margin_m))
        # wide clearance band that's safe by a conservative circumscribed-radius bound; helps to skip expensive footprint checks
        if dO >= gate + margin:
            return True
        # Medium clearance band: use conservative cached cells if available
        if footprint_cache is not None and theta_bins > 0 and dO >= gate:
            it = theta_to_bin(pose.theta, theta_bins)
            return pose_is_free_cached_cells(pose, grid, footprint_cache[it])
        # Tight band: do the exact footprint check
        return pose_is_free(pose, grid, footprint_offsets)

    def rollout(
        self,
        pose: Pose,
        u: float,
        direction: int,
        ds: float,
        n_substeps: int,
        kappa_max: float,
        grid: OccupancyGrid,
        footprint_offsets: Optional[np.ndarray],
        dO_m: Optional[np.ndarray] = None,
        gate_radius_m: float = 0.0,
        exact_check_margin_m: float = 0.0,
        footprint_cache: Optional[Sequence[np.ndarray]] = None,
        theta_bins: int = 0,
        rho: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[Pose, float]]: # ) -> Optional[Pose]:
        r""" Propagate + collision-check along the edge; returns (endpoint, $\int \rho ds$) if collision-free else None """
        step = float(ds) / float(n_substeps)
        cur = pose
        rho_int = 0.0
        for _ in range(int(n_substeps)):
            cur = self.propagate(cur, u, direction, step, kappa_max = kappa_max)
            # compute grid cell indices
            ix, iy = grid.world_to_grid(cur.x, cur.y)
            if not grid.in_bounds(ix, iy):
                return None
        clearance = float(dO_m[iy, ix])
        # conservative distance-transform gating + cached footprint band
        if dO_m is None or gate_radius_m <= 0.0:
            if not pose_is_free(cur, grid, footprint_offsets):
                return None
        else:
            if not BicycleModel.pose_is_free_fast(
                cur, grid, footprint_offsets, clearance,
                gate_radius_m=float(gate_radius_m),
                footprint_cache=footprint_cache,
                theta_bins=int(theta_bins),
            ):
                return None
            # integrate Voronoi cost if available
            if rho is not None:
                rho_int += float(rho[iy, ix]) * step
        return cur, float(rho_int)

