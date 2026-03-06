# src/dolgov_cbmp/models/heuristics.py
from typing import List, Dict, Optional, Tuple, Any
import heapq
import math
import numpy as np

from dolgov_cbmp.models import OccupancyGrid #, BicycleModel
from dolgov_cbmp.structs import Pose, PlannerConfig
from dolgov_cbmp.utils import TAU, SQRT2, wrap_angle, kappa_to_bin, theta_to_bin



class HolonomicWithObstacles2D:
    """ 2D Dijkstra cost-to-go map on the grid, ignoring non-holonomy; paper uses this to detect U-shaped obstacles/dead-ends """
    def __init__(
        self,
        grid: OccupancyGrid,
        cost_per_cell: Optional[np.ndarray] = None,
        dO_m: Optional[np.ndarray] = None,
        min_clearance_m: float = 0.0,
        soft_clearance_m: float = 0.0,
        soft_clearance_weight: float = 0.0,
    ):
        self.grid = grid
        self.cost_per_cell = cost_per_cell      # optional additional per-cell cost (e.g., Voronoi rho)
        self._dist: Optional[np.ndarray] = None      # [H,W] - distance computed per goal
        self._done: Optional[np.ndarray] = None      # [H,W] settled flags
        self._pq: List[Tuple[float, int, int]] = []  # heap for lazy Dijkstra
        self._nbrs = None
        self._dO = dO_m
        self._min_clear = float(min_clearance_m)
        self._soft_clear_m = float(soft_clearance_m)
        self._soft_clear_w = float(soft_clearance_weight)
        self._use_soft_clear = self._dO is not None and self._soft_clear_m > 0.0 and self._soft_clear_w > 0.0
        self._prune_hard_clear = self._dO is not None and self._min_clear > 0.0


    def compute(self, goal: Pose) -> None:
        H, W = self.grid.height, self.grid.width
        dist = np.full((H,W), np.inf, dtype=np.float64) # NOTE: needs to stay float64 for heapq comparisons
        done = np.zeros((H,W), dtype=bool)
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
                # optional hard clearance pruning using dO
                if self.grid.is_occupied(nx, ny) or (self._prune_hard_clear and float(self._dO[ny, nx]) < self._min_clear):
                    #? WARNING: passing the vehicle circumscribed radius here will usually over-prune the heuristic
                    continue
                # base per-cell shaping cost (e.g., Voronoi field) if provided - maybe allow negatives for attractor field functionality
                extra = float(self.cost_per_cell[ny, nx]) if self.cost_per_cell is not None else 0.0
                # optional soft clearance bias (no pruning) - meant to be intentionally cheap
                if self._use_soft_clear:
                    dO = float(self._dO[ny, nx])
                    # quadratic ramp in [0, 1] for dO in [0, soft_clearance_m]
                    r_squared = max(0.0, (self._soft_clear_m - dO) / self._soft_clear_m) ** 2
                    extra += self._soft_clear_w * r_squared
                # Dijkstra relaxation step
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
    # Process-wide cache: non-holonomic table is goal-independent, so we reuse it across planners created w/ the same discretization settings
    _OFFLINE_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, Tuple[float, float, int, float, int, bool, int, float, float]]] = {}

    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self._table: Optional[np.ndarray] = None  # [iy, ix, itheta, ikappa]
        # meta: (R, res, theta_bins, dth, nxy, use_kappa_dim, kappa_bins, kappa_max, dkappa)
        self._meta: Optional[Tuple[float, float, int, float, int, bool, int, float, float]] = None
        # cache goal-dependent trig to avoid redoing cos/sin on every heuristic query
        self._goal_cache_key: Optional[Tuple[float, float, float, float]] = None
        self._goal_cache_rot: Optional[Tuple[float, float]] = None

    @staticmethod
    def _cache_key(
        R: float,
        res: float,
        theta_bins: int,
        dth: float,
        use_kappa_dim: bool,
        kappa_bins: int,
        kappa_max: float,
        dkappa: float,
        u_set: np.ndarray,
    ) -> Tuple[Any, ...]:
        u_key = tuple(float(x) for x in u_set)
        return (R, res, theta_bins, dth, use_kappa_dim, kappa_bins, kappa_max, dkappa, u_key)


    def _run_dijkstra(
        self,
        table: np.ndarray,
        nxy: int,
        res: float,
        theta_bins: int,
        dth: float,
        use_kappa_dim: bool,
        kappa_bins: int,
        kappa_max: float,
        dkappa: float,
        action_set: np.ndarray,
        gx: int,
        gy: int,
        gt: int,
        gk: Optional[int],
    ) -> None:
        """ Tight Dijkstra search loop with reverse dynamics (propagating from goal outward using negative step size) to fill heuristic table with min path costs
            Speedups:
                - avoids allocating Pose objects + calling BicycleModel.propagate in the inner loop
                - avoids per-node list creation when relaxing neighbors (precompute neighbor states as tuples of indices and costs)
                - precomputes x/y/theta/kappa lookup tables
        """
        center = nxy // 2
        inv_res = 1.0 / float(res)
        eps = 1e-6
        # small action sets; materialize as Python floats once (faster than iterating numpy scalars)
        actions: List[float] = action_set.tolist()
        # precompute coordinate and heading lookup tables
        xy_lookup = (np.arange(nxy, dtype=np.float64) - float(center)) * float(res)
        th_lookup = np.arange(theta_bins, dtype=np.float64) * float(dth)
        k_lookup: Optional[np.ndarray] = None
        # inv_dkappa: Optional[float] = None
        pq_seed = (gx, gy, gt, gk) if use_kappa_dim else (gx, gy, gt)
        if use_kappa_dim:
            assert gk is not None
            k_lookup = (-float(kappa_max) + (np.arange(kappa_bins, dtype=np.float64) + 0.5) * float(dkappa))
        for i in range(theta_bins):
            th_lookup[i] = wrap_angle(float(th_lookup[i]))
        # heap state: (d, tie, ix, iy, it, ik) if use_kappa_dim else (d, tie, ix, iy, it)
        pq: List[Tuple[float, int, int, int, int, Optional[int]]] = [(0.0, 0, *pq_seed)]
        tie = 0
        while pq:
            pq_key = heapq.heappop(pq)
            d, _, ix, iy, it = pq_key[:5]
            table_idx = list(pq_key[2:])
            table_idx[0], table_idx[1] = table_idx[1], table_idx[0]  # swap ix/iy for correct table indexing
            if d > float(table[*table_idx]) + eps:
                continue
            x0 = float(xy_lookup[ix])
            y0 = float(xy_lookup[iy])
            th0 = float(th_lookup[it])
            k0: Optional[float] = float(k_lookup[pq_key[5]]) if use_kappa_dim else None
            for sigma in (1.0, -1.0):
                for u in actions: # may be either curvature or curvature rate depending on the table type
                    # inline BicycleModel.propagate midpoint integration
                    k1: Optional[float] = None
                    if use_kappa_dim:
                        # inline BicycleModel.propagate midpoint integration for curvature-aware table
                        k1 = k0 + u * res
                        k1 = max(-kappa_max, min(k1, kappa_max))  # clamp to [-kappa_max, kappa_max]
                    # use (constant) kappa action directly when not using curvature dimension
                    km = 0.5 * (k0 + k1) if use_kappa_dim else u
                    thm = th0 + 0.5 * sigma * km * res
                    x1 = x0 + sigma * res * math.cos(thm)
                    y1 = y0 + sigma * res * math.sin(thm)
                    th1 = wrap_angle(th0 + sigma * km * float(res))
                    nix = int(round(x1 * inv_res)) + center
                    niy = int(round(y1 * inv_res)) + center
                    if (nix < 0) or (nix >= nxy) or (niy < 0) or (niy >= nxy):
                        continue
                    nit = theta_to_bin(th1, theta_bins, dth)
                    nik: Optional[int] = kappa_to_bin(k1, -kappa_max, kappa_max, dkappa, kappa_bins) if use_kappa_dim else None
                    nxt_idx = (niy, nix, nit, nik) if use_kappa_dim else (niy, nix, nit)
                    nd = d + res
                    if nd + 1e-9 < float(table[nxt_idx]):
                        table[nxt_idx] = nd
                        tie += 1
                        nxt_key = (nix, niy, nit, nik) if use_kappa_dim else (nix, niy, nit)
                        heapq.heappush(pq, (nd, tie, *nxt_key))


    def build_offline(self) -> None:
        """ Build goal-local heuristic table over (x, y, theta) or (x, y, theta, kappa) by running Dijkstra from the goal state outward using REVERSE dynamics
            - If cfg.heuristics.nh_kappa_bins is None (default), we match the *paper* heuristic more closely:
                state = (x, y, theta)
                action = choose curvature (steering) directly each step
                - intentionally ignores curvature-rate continuity and curvature state dimension, making the heuristic optimistic
                    (good for A* guidance) and much cheaper to precompute
            - Elif cfg.heuristics.nh_kappa_bins is set, we build a larger table that additionally discretizes curvature.
        """
        R = float(self.cfg.heuristics.nh_table_xy_radius)
        res = float(self.cfg.heuristics.nh_table_xy_res)
        dth = float(self.cfg.heuristics.nh_table_theta_res)
        # local table dims
        nxy = int(math.ceil((2.0 * R) / res))
        if nxy % 2 == 0:
            nxy += 1
        theta_bins = int(round(TAU / dth))
        dth = TAU / float(theta_bins)
        # kappa_bins = int(self.cfg.heuristics.nh_kappa_bins or self.cfg.grid.kappa_bins)
        kappa_max = float(self.cfg.curvature.kappa_max)
        use_kappa_dim = self.cfg.heuristics.nh_kappa_bins is not None
        # model = BicycleModel(self.cfg.vehicle)
        # Goal at center cell, theta=0
        gx, gy, gt = nxy // 2, nxy // 2, 0
        # relaxation_args = [nxy, dth]
        # nh_state = [0.0, gx, gy, gt]
        table_dims = [nxy, nxy, theta_bins]
        kappa_bins: int = 0
        dkappa: float = 0.0
        gk: Optional[int] = None
        if use_kappa_dim:
            # Optional curvature-aware table (x, y, theta, kappa)
            kappa_bins = int(self.cfg.heuristics.nh_kappa_bins)
            dkappa = (2.0 * kappa_max) / float(kappa_bins)
            gk = kappa_to_bin(0.0, -kappa_max, kappa_max, dkappa, kappa_bins)
            table_dims.append(kappa_bins)
            # relaxation_args += [dkappa]
            # nh_state.append(gk)
        action_set = self._action_set(use_kappa_dim)
        # attempt to reuse a previously computed table (primarily useful for unit tests that repeatedly rebuilds planners)
        key = self._cache_key(
            R=R,
            res=res,
            theta_bins=theta_bins,
            dth=dth,
            use_kappa_dim=use_kappa_dim,
            kappa_bins=kappa_bins,
            kappa_max=kappa_max,
            dkappa=dkappa,
            u_set=action_set,
        )
        cached = self._OFFLINE_CACHE.get(key)
        if cached is not None:
            self._table, self._meta = cached
            return
        table = np.full(table_dims, np.inf, dtype=np.float64)
        if use_kappa_dim:
            assert gk is not None
            table[gy, gx, gt, gk] = 0.0
        else:
            table[gy, gx, gt] = 0.0
        # run the fast Dijkstra search and then freeze the table (it should be read-only at runtime)
        self._run_dijkstra(
            table=table, nxy=nxy,
            res=res, theta_bins=theta_bins, dth=dth, action_set=action_set,
            use_kappa_dim=use_kappa_dim, kappa_bins=kappa_bins, kappa_max=kappa_max, dkappa=dkappa,
            gx=gx, gy=gy, gt=gt, gk=gk,
        )
        meta = (R, res, theta_bins, dth, nxy, use_kappa_dim, kappa_bins, kappa_max, dkappa)
        # make the cached table immutable (guard against accidental writes from other code paths)
        table = np.ascontiguousarray(table, dtype=np.float64)
        table.setflags(write=False)
        self._table = table
        self._meta = meta
        self._OFFLINE_CACHE[key] = (table, meta)


    def _action_set(self, use_kappa_dim: bool = False) -> np.ndarray:
        if use_kappa_dim:
            kappa_max = float(self.cfg.curvature.kappa_max)
            return np.array([-kappa_max, 0.0, kappa_max], dtype=np.float64)
        m = int(self.cfg.curvature.kappa_rate_samples)
        u_max = float(self.cfg.curvature.kappa_rate_max)
        if m <= 1:
            return np.array([0.0])
        # return symmetric samples including 0
        return np.linspace(-u_max, u_max, m, dtype=np.float64)


    def cpp_view(self) -> Tuple[np.ndarray, float, float, float, int, int]:
        """ Return contiguous table and metadata for the C++ kernel """
        if self._table is None or self._meta is None:
            raise RuntimeError("build_offline() must be called before cpp_view()")
        # R, res, theta_bins, dth, nxy = self._meta
        #!!! FIXME: C++ backend not updated to use curvature dimension yet
        R, res, theta_bins, dth, nxy, use_kappa_dim, _, _, _ = self._meta
        table = np.ascontiguousarray(self._table, dtype=np.float64)
        return table, float(R), float(res), float(dth), int(nxy), int(theta_bins)


    def __call__(self, pose: Pose, goal: Pose) -> float:
        # use Euclidean fallback if table not built yet
        if self._table is None or self._meta is None:
            dx, dy = pose.x - goal.x, pose.y - goal.y
            return math.hypot(dx, dy)
        R, res, theta_bins, dth, nxy, use_kappa_dim, kappa_bins, kappa_max, dkappa = self._meta
        dx = pose.x - goal.x
        dy = pose.y - goal.y
        # rotate (dx,dy) into goal-local frame (cache cos/sin per-goal for speed)
        gkey = (float(goal.x), float(goal.y), float(goal.theta), float(goal.kappa))
        if self._goal_cache_key != gkey:
            self._goal_cache_key = gkey
            self._goal_cache_rot = (math.cos(float(goal.theta)), math.sin(float(goal.theta)))
        assert self._goal_cache_rot is not None
        cg, sg = self._goal_cache_rot
        xl = cg * dx + sg * dy
        yl = -sg * dx + cg * dy
        if abs(xl) > R or abs(yl) > R:
            return math.hypot(xl, yl)
        ix = int(round(xl / res)) + nxy // 2
        iy = int(round(yl / res)) + nxy // 2
        thl = wrap_angle(pose.theta - goal.theta)
        # curvature-aware lookup (optional)
        it = theta_to_bin(thl, theta_bins, dth)
        if (not use_kappa_dim) or self._table.ndim == 3:
            val = float(self._table[iy, ix, it])
            if not math.isfinite(val):
                return math.hypot(xl, yl)
            return val
        ik = kappa_to_bin(float(pose.kappa - goal.kappa), -kappa_max, kappa_max, dkappa, kappa_bins)
        val = float(self._table[iy, ix, it, ik])
        if not math.isfinite(val):
            return math.hypot(xl, yl)
        return val