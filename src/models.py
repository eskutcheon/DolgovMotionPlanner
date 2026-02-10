
from typing import List, Optional, Tuple, Sequence, Dict
import heapq
import math
import numpy as np
from pathlib import Path

from src.structs import GridSpec, VehicleParams, Pose, DiscreteKey, PlannerConfig
from src.utils import TAU, SQRT2, wrap_angle, wrap_angle_2pi, pose_is_free, pose_is_free_cached_cells



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
        print("Loaded occupancy grid from", path, f"with shape {occ.shape} and dtype {occ.dtype}")
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
            # np.pad(occ, pad_width=pad_cells, mode='reflect')
            occ = np.pad(occ, pad_width=int(pad_cells), mode='reflect') #mode="constant", constant_values=True)
            ox, oy = grid.origin_xy
            r = float(grid.resolution)
            gridspec_kwargs = grid.to_dict()
            gridspec_kwargs['origin_xy'] = (ox - pad_cells * r, oy - pad_cells * r)
            grid = GridSpec(**gridspec_kwargs)
            print("Applied padding of", pad_cells, "cells; new shape:", occ.shape)
        # create OccupancyGrid and set start/goal accordingly
        occ_grid = OccupancyGrid(occ, grid)
        # occ_grid.view_grid()    #! DEBUGGING - remove later
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
                if 0 <= ix < self.occ.shape[1] and 0 <= iy < self.occ.shape[0]:
                    # mark collisions in red and other path poses in yellow
                    marker = RED + 'X' + RESET if self.occ[iy, ix] else YELLOW + 'o' + RESET
                    grid_display[iy][ix] = marker
        # print(grid_display)
        for row in grid_display:
            print("".join(row))



def adjust_start_pose_for_clearance(
    start: Pose,
    grid: OccupancyGrid,
) -> Pose:
    """ if starting point is in obstacle, jitter to nearest neighbors until free """
    x, y = start.x, start.y
    ix, iy = grid.world_to_grid(x, y)
    tried = set()
    while grid.is_occupied(ix, iy):
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
    new_x, new_y = grid.grid_to_world(ix, iy)
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
        self.dO = dO_m.astype(np.float64, copy=False)
        self.dV = dV_m.astype(np.float64, copy=False) if dV_m is not None else None
        # self.params = params
        self.alpha = float(alpha)
        self.dO_max = float(dO_max)

    # TODO: consider decorating this function with @property to cache the result
    def rho(self) -> np.ndarray:
        """ Vectorized potential in [0,1] on the grid.
            NOTE: If $dV$ is not available, proxy it with $dV := dO$ (keeps a weak "skeleton-ish" scaling effect)
        """
        alpha = float(self.alpha)
        dO_max = float(self.dO_max)
        dO = self.dO
        dV = self.dV if self.dV is not None else dO
        # Eq. (1)-style potential, clipped to [0,1]
        a = np.clip(dO / dO_max, 0.0, 1.0)
        b = 1.0 - np.clip(dV / dO_max, 0.0, 1.0)
        rho = (a**alpha) * b
        # clamp numeric noise
        return np.clip(rho, 0.0, 1.0).astype(np.float64, copy=False)




# Discretization utilities
# TODO: move to utils?
def kappa_to_bin(kappa: float, kappa_min: float, kappa_max: float, dkappa: float, kappa_bins: int) -> int:
    """ Map curvature to bin index """
    k = max(kappa_min, min(kappa_max, kappa))
    bin_idx = int(math.floor((k - kappa_min) / dkappa))
    return min(max(bin_idx, 0), kappa_bins - 1)


#? NOTE: both heuristic model classes have repeated use of some of the same discretization methods and may as well use a shared Indexer class


class Indexer:
    """ Discretizes (x,y,theta,dir) and provides a flat index for best-g arrays """
    def __init__(self, grid: OccupancyGrid, *, kappa_bins: Optional[int] = None, kappa_max: Optional[float] = None):
        self.grid = grid
        self.W = int(grid.width)
        self.H = int(grid.height)
        self.theta_bins = int(grid.grid.theta_bins)
        self.dtheta = TAU / float(self.theta_bins)
        #& UPDATE: adding curvature parameters for the new discretization functions
        self.kappa_bins = int(kappa_bins or grid.grid.kappa_bins)
        self.kappa_max = float(kappa_max or grid.grid.kappa_max)
        self.kappa_min = -self.kappa_max        # placeholder; should be set from PlannerConfig
        self.dkappa = (self.kappa_max - self.kappa_min) / self.kappa_bins

    def _theta_to_bin(self, theta: float) -> int:
        """ Map heading to bin index """
        t = wrap_angle_2pi(theta)
        return int(math.floor(t / self.dtheta)) % self.theta_bins

    #& UPDATE: modifying functions below to use curvature, not direction
    #&#############################################################################################

    #? NOTE: DiscreteKey could be removed in favor of explicit tuples
    def pose_to_key(self, pose: Pose) -> DiscreteKey:
        ix, iy = self.grid.world_to_grid(pose.x, pose.y)
        itheta = self._theta_to_bin(pose.theta)
        ikappa = kappa_to_bin(pose.kappa, self.kappa_min, self.kappa_max, self.dkappa, self.kappa_bins)
        return DiscreteKey(ix=ix, iy=iy, itheta=itheta, ikappa=ikappa) #direction=1 if direction >= 0 else -1)

    def key_to_flat(self, key: DiscreteKey) -> int:
        """ return flat packed index for best-g arrays """
        return int((((key.iy * self.W + key.ix) * self.theta_bins + key.itheta) * self.kappa_bins + key.ikappa))

    #&#############################################################################################


# ----------------------------
# Motion model (constant curvature bicycle)
# ----------------------------

class BicycleModel:
    def __init__(self, vehicle: VehicleParams):
        self.L = float(vehicle.wheelbase)

    #& UPDATE: modified propagation approach to use curvature instead of steering angle
    def propagate(self, pose: Pose, u: float, direction: int, ds: float, *, kappa_max: float) -> Pose:
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


    def rollout(
        self,
        pose: Pose,
        # steer: float,
        u: float,
        direction: int,
        ds: float,
        n_substeps: int,
        kappa_max: float,
        grid: OccupancyGrid,
        footprint_offsets: Optional[np.ndarray],
        *,
        dO_m: Optional[np.ndarray] = None,
        gate_radius_m: float = 0.0,
        exact_check_margin_m: float = 0.0,
        footprint_cache: Optional[Sequence[np.ndarray]] = None,
        theta_bins: int = 0,
        rho: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[Pose, float]]: # ) -> Optional[Pose]:
        r""" Propagate + collision-check along the edge; returns (endpoint, $\int \rho ds$) if collision-free else None. """
        #& UPDATE: added curvature parameters to the function signature above, which now returns a tuple of (Pose, float) or None
        step = float(ds) / float(n_substeps)
        cur = pose
        dtheta = TAU / float(theta_bins) if theta_bins > 0 else 0.0
        rho_int = 0.0
        for _ in range(int(n_substeps)):
            cur = self.propagate(cur, u, direction, step, kappa_max = kappa_max)
            # compute grid cell indices
            ix, iy = grid.world_to_grid(cur.x, cur.y)
            if not grid.in_bounds(ix, iy):
                return None
            # TODO: would prefer to use memoization through functools rather than explicitly handling footprint_cache here
            # conservative distance-transform checkpoint: if reference point has enough clearance, accept immediately
            if (dO_m is None) or (gate_radius_m <= 0.0) or (float(dO_m[iy, ix]) < float(gate_radius_m)):
                # Use cached footprint when not extremely tight; fall back to exact footprint near obstacles.
                if (
                    footprint_cache is not None
                    and theta_bins > 0
                    and dO_m is not None
                    and float(dO_m[iy, ix]) >= float(gate_radius_m) + float(exact_check_margin_m)
                ):
                    it = int(math.floor(wrap_angle_2pi(cur.theta) / dtheta)) % int(theta_bins)
                    if not pose_is_free_cached_cells(cur, grid, footprint_cache[it]):
                        return None
                else:
                    if not pose_is_free(cur, grid, footprint_offsets):
                        return None
            # integrate Voronoi cost if available
            if rho is not None:
                rho_int += float(rho[iy, ix]) * step
        return cur, float(rho_int)



# ----------------------------
# Heuristics from the paper
# ----------------------------

class HolonomicWithObstacles2D:
    """ 2D Dijkstra cost-to-go map on the grid, ignoring non-holonomy; paper uses this to detect U-shaped obstacles/dead-ends """
    def __init__(self, grid: OccupancyGrid, cost_per_cell: Optional[np.ndarray] = None,
                 dO_m: Optional[np.ndarray] = None, min_clearance_m: float = 0.0):
        self.grid = grid
        self.cost_per_cell = cost_per_cell      # optional additional per-cell cost (e.g., Voronoi rho)
        # self._dist: Optional[np.ndarray] = None # computed per goal
        self._dist: Optional[np.ndarray] = None      # [h,w]
        self._done: Optional[np.ndarray] = None      # [h,w] settled flags
        self._pq: List[Tuple[float, int, int]] = []  # heap for lazy Dijkstra
        self._nbrs = None
        self._dO = dO_m
        self._min_clear = float(min_clearance_m)


    def compute(self, goal: Pose) -> None:
        w, h = self.grid.width, self.grid.height
        dist = np.full((h, w), np.inf, dtype=np.float64) # NOTE: needs to stay float64 for heapq comparisons
        done = np.zeros((h, w), dtype=bool)
        gx, gy = self.grid.world_to_grid(goal.x, goal.y)
        # goal in obstacle => heuristic is unusable; keep inf and fallback at query-time
        if not self.grid.in_bounds(gx, gy) or self.grid.is_occupied(gx, gy):
            self._dist = dist
            self._done = done
            self._pq = []
            return
        c1 = float(self.grid.grid.resolution)
        c2 = c1 * SQRT2
        self._nbrs = ((-1, 0, c1), (1, 0, c1), (0, -1, c1), (0, 1, c1), (-1, -1, c2), (-1, 1, c2), (1, -1, c2), (1, 1, c2))
        pq: List[Tuple[float, int, int]] = []
        dist[gy, gx] = 0.0
        heapq.heappush(pq, (0.0, gx, gy))
        #& UPDATE: `compute` now just initializes the model while `__call__` internally calls `_ensure_settled` to loop over pq and lazily settle nodes
        self._dist = dist
        self._done = done
        self._pq = pq


    def _ensure_settled(self, tx: int, ty: int) -> None:
        if self._dist is None or self._done is None or self._nbrs is None:
            # print("HolonomicWithObstacles2D: compute() must be called before _ensure_settled()")
            return
        if not self.grid.in_bounds(tx, ty) or self._done[ty, tx]:
            # print("HolonomicWithObstacles2D: target cell already settled or out of bounds: ", tx, ty)
            return
        while self._pq:
            d, x, y = heapq.heappop(self._pq)
            if self._done[y, x] or d > self._dist[y, x] + 1e-6:
                continue
            self._done[y, x] = True
            for dx, dy, step in self._nbrs:
                nx, ny = x + dx, y + dy
                # TODO: create small helpers for this and other long conditional chains
                if self.grid.is_occupied(nx, ny) or (self._dO is not None and self._min_clear > 0.0 and float(self._dO[ny, nx]) < self._min_clear):
                    continue
                extra = float(self.cost_per_cell[ny, nx]) if self.cost_per_cell is not None else 0.0
                nd = d + step * (1.0 + extra)
                if nd < self._dist[ny, nx]:
                    self._dist[ny, nx] = nd
                    heapq.heappush(self._pq, (nd, nx, ny))
            # after relaxing neighbors, check if we settled the target cell (must come after or relaxation never happens when the first query is the goal)
            if x == tx and y == ty:
                return

    def __call__(self, pose: Pose) -> float:
        if self._dist is None:
            # print("HolonomicWithObstacles2D: compute() must be called before __call__()")
            return float("inf") # 0.0
        ix, iy = self.grid.world_to_grid(pose.x, pose.y)
        if not self.grid.in_bounds(ix, iy):
            # print("HolonomicWithObstacles2D: query out of bounds: ", ix, iy)
            return float("inf")
        self._ensure_settled(ix, iy)
        if self._done is not None and not self._done[iy, ix]:
            # print("HolonomicWithObstacles2D: cell not settled after _ensure_settled(): ", ix, iy)
            return float("inf")
        # print("values of self._dist that are non-inf: ", self._dist[self._dist < float("inf")])
        return float(self._dist[iy, ix])

    def cpp_view(self) -> np.ndarray:
        if self._dist is None:
            raise RuntimeError("compute() must be called before cpp_view()")
        return np.ascontiguousarray(self._dist, dtype=np.float64)



# TODO: replace bits and pieces of these classes with new Spec classes instead of passing so much
class NonHolonomicWithoutObstaclesTable:
    """ Goal-local heuristic table over (x, y, theta), ignoring obstacles - implements Dijkstra over a small goal-centered grid in goal frame
        - paper computes shortest path to (0,0,0) in a neighborhood, offline
    """
    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self._table: Optional[np.ndarray] = None  # [iy, ix, itheta, ikappa]
        # (R, res, theta_bins, dth, nxy, kappa_bins, kappa_max, dkappa)
        self._meta: Optional[Tuple[float, float, int, float, int, int, float, float]] = None

    def build_offline(self) -> None:
        R = float(self.cfg.nh_table_xy_radius)
        res = float(self.cfg.nh_table_xy_res)
        dth = float(self.cfg.nh_table_theta_res)
        # local table dims
        nxy = int(math.ceil((2.0 * R) / res))
        if nxy % 2 == 0:
            nxy += 1
        theta_bins = int(round(TAU / dth))
        dth = TAU / float(theta_bins)
        kappa_bins = int(self.cfg.grid.kappa_bins)
        kappa_max = float(self.cfg.kappa_max)
        dkappa = (2.0 * kappa_max) / float(kappa_bins)

        def idx_to_xy(ix: int, iy: int) -> Tuple[float, float]:
            """ index <-> coordinate in goal frame """
            x = (ix - nxy // 2) * res
            y = (iy - nxy // 2) * res
            return x, y

        table = np.full((nxy, nxy, theta_bins, kappa_bins), np.inf, dtype=np.float64)
        # Motion primitives (same as search): constant steering for one step_size in LOCAL metric
        #? NOTE: ds should align to res for table consistency; we use res here.
        model = BicycleModel(self.cfg.vehicle)
        #& UPDATE: changed use of steering set to curvature rate set (using new _u_set in place of _steer_set)
        # steer_set = self._steer_set()
        u_set = self._u_set()
        # Goal at center cell, theta=0, kappa=0
        gx = nxy // 2
        gy = nxy // 2
        gt = 0
        gk = int(round(kappa_max / dkappa))
        table[gy, gx, gt, gk] = 0.0
        pq: List[Tuple[float, int, int, int, int]] = [(0.0, gx, gy, gt, gk)]
        heapq.heapify(pq)
        # Run Dijkstra outward by applying REVERSE dynamics. Instead of inverting bicycle exactly, we approximate by applying forward
            # primitives from each state and relaxing neighbors (works since costs are symmetric-ish for the obstacle-free heuristic)
        while pq:
            d, ix, iy, it, ik = heapq.heappop(pq)
            if d != table[iy, ix, it, ik]:
                continue
            x, y = idx_to_xy(ix, iy)
            th = wrap_angle(it * dth)
            kappa = -kappa_max + (ik + 0.5) * dkappa
            cur = Pose(x=x, y=y, theta=th, kappa=kappa)
            # expand neighbors and relax costs
            for direction in (+1, -1): # include reverse in heuristic table
                for u in u_set:
                    nxt = model.propagate(cur, u, direction, res, kappa_max = kappa_max)
                    # map nxt to table indices
                    nix = int(round(nxt.x / res)) + nxy // 2
                    niy = int(round(nxt.y / res)) + nxy // 2
                    if not (0 <= nix < nxy and 0 <= niy < nxy):
                        continue
                    # TODO: replace with some global theta_to_bin method later
                    nit = int(math.floor(wrap_angle_2pi(nxt.theta) / dth)) % theta_bins
                    nik = kappa_to_bin(nxt.kappa, -kappa_max, kappa_max, dkappa, kappa_bins)
                    # small reverse penalty in heuristic can be set to 0 for admissibility as in this case
                    nd = d + res
                    if nd < table[niy, nix, nit, nik]:
                        table[niy, nix, nit, nik] = nd
                        heapq.heappush(pq, (nd, nix, niy, nit, nik))
        self._table = table
        self._meta = (R, res, theta_bins, dth, nxy, kappa_bins, kappa_max, dkappa)


    def _u_set(self) -> List[float]:
        m = int(self.cfg.kappa_rate_samples)
        u_max = float(self.cfg.kappa_rate_max)
        if m <= 1:
            return [0.0]
        # return symmetric samples including 0
        return np.linspace(-u_max, u_max, m, dtype=np.float64).tolist()


    def cpp_view(self) -> Tuple[np.ndarray, float, float, float, int, int]:
        """ Return contiguous table and metadata for the C++ kernel """
        if self._table is None or self._meta is None:
            raise RuntimeError("build_offline() must be called before cpp_view()")
        # R, res, theta_bins, dth, nxy = self._meta
        #!!! FIXME: C++ backend not updated to use curvature dimension yet
        R, res, theta_bins, dth, nxy, kappa_bins, kappa_max, dkappa = self._meta
        table = np.ascontiguousarray(self._table, dtype=np.float64)
        return table, float(R), float(res), float(dth), int(nxy), int(theta_bins)

    def __call__(self, pose: Pose, goal: Pose) -> float:
        # use Euclidean fallback if table not built yet
        if self._table is None or self._meta is None:
            dx, dy = pose.x - goal.x, pose.y - goal.y
            return math.hypot(dx, dy)
        R, res, theta_bins, dth, nxy, kappa_bins, kappa_max, dkappa = self._meta
        dx = pose.x - goal.x
        dy = pose.y - goal.y
        # UPDATE: removed numpy overhead in favor of explicit rotation matrix multiplication with vector $(\Delta x, \Delta y)$
        cg = math.cos(goal.theta)
        sg = math.sin(goal.theta)
        xl = cg * dx + sg * dy
        yl = -sg * dx + cg * dy
        if abs(xl) > R or abs(yl) > R:
            return math.hypot(xl, yl)
        ix = int(round(xl / res)) + nxy // 2
        iy = int(round(yl / res)) + nxy // 2
        thl = wrap_angle(pose.theta - goal.theta)
        it = int(math.floor(wrap_angle_2pi(thl) / dth)) % theta_bins
        ik = kappa_to_bin(float(pose.kappa - goal.kappa), -kappa_max, kappa_max, dkappa, kappa_bins)
        val = float(self._table[iy, ix, it, ik])
        if not math.isfinite(val):
            return math.hypot(xl, yl)
        return val