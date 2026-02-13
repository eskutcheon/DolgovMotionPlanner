# src/models/heuristics.py
from typing import List, Optional, Tuple
import heapq
import math
import numpy as np

from .models import OccupancyGrid, BicycleModel
from src.structs import Pose, PlannerConfig
from src.utils import TAU, SQRT2, wrap_angle, wrap_angle_2pi, kappa_to_bin




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
        kappa_bins = int(self.cfg.nh_kappa_bins or self.cfg.grid.kappa_bins)
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
        # gk = int(round(kappa_max / dkappa))
        gk = kappa_to_bin(0.0, -kappa_max, kappa_max, dkappa, kappa_bins)
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