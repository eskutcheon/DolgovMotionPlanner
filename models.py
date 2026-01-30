
from typing import List, Optional, Tuple, Sequence, Protocol
import heapq
import math
import numpy as np

from structs import GridSpec, VehicleParams, Pose, VoronoiParams, DiscreteKey, PlannerConfig
from utils import TAU, wrap_angle, wrap_angle_2pi




# ----------------------------
# Occupancy + distance field
# ----------------------------

class OccupancyGrid:
    """ Binary occupancy grid in row-major (iy, ix) order - `occ[y, x] == True` means obstacle.
        Important for multi-threading later: treat `occ` as read-only during planning
    """
    def __init__(self, occupied: np.ndarray, grid: GridSpec):
        if occupied.dtype != np.bool_:
            occupied = occupied.astype(np.bool_)
        self.occ: np.ndarray = occupied
        self.grid: GridSpec = grid
        self.height, self.width = occupied.shape

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height

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




# ----------------------------
# Collision checking
# ----------------------------

class CollisionChecker(Protocol):
    def edge_is_free(self, poses: Sequence[Pose]) -> bool: ...


class ReferencePointCollisionChecker:
    """ Baseline: checks only the center point against occupancy.
        NOTE: fast, but not footprint-accurate - used for benchmarking baselines
    """
    def __init__(self, grid: OccupancyGrid):
        self.grid = grid

    def edge_is_free(self, poses: Sequence[Pose]) -> bool:
        for p in poses:
            ix, iy = self.grid.world_to_grid(p.x, p.y)
            if self.grid.is_occupied(ix, iy):
                return False
        return True



def compute_distance_to_obstacles_m(occ: np.ndarray, resolution: float) -> np.ndarray:
    """ dO(x,y): Euclidean distance to nearest obstacle in meters.
        fast path uses SciPy if available; fallback uses an approximate chamfer distance
    """

    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
        free = ~occ
        dist = distance_transform_edt(free).astype(np.float64) * resolution
        return dist
    except Exception:
        # Approximate fallback: multi-pass chamfer-like distance on grid
        # (Not as accurate as EDT; intended only as a functional baseline.)
        h, w = occ.shape
        INF = 1e9
        dist = np.full((h, w), INF, dtype=np.float64)
        dist[occ] = 0.0
        # two-pass 8-neighbor chamfer with costs (1, sqrt(2))
        c1 = resolution
        c2 = resolution * math.sqrt(2.0)

        def min_pass(x_iter, y_iter, neighbors):
            for y in y_iter:
                for x in x_iter:
                    # skip any obstacles
                    if dist[y, x] == 0.0:
                        continue
                    best = dist[y, x]
                    # check each neighbor cell (within boundaries) for possible improvement
                    for dx, dy, cost in neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            best = min(best, dist[ny, nx] + cost)
                    dist[y, x] = best

        # forward then backward passes
        fwd_neighbors = ((-1, 0, c1), (0, -1, c1), (-1, -1, c2), (1, -1, c2))
        min_pass(range(w), range(h), fwd_neighbors)
        bwd_neighbors = ((1, 0, c1), (0, 1, c1), (1, 1, c2), (-1, 1, c2))
        min_pass(range(w - 1, -1, -1), range(h - 1, -1, -1), bwd_neighbors)
        return dist


# ----------------------------
# Voronoi Field (paper Eq. 1)
# ----------------------------


class VoronoiField:
    """ $rho_V(x,y)$ shaped by distance-to-obstacles (and optionally GVD distance).
        This follows the same structure as Dolgov et al. (AAAI 2008):
            - $dO(x,y)$: distance to nearest obstacle
            - $dV(x,y)$: distance to generalized Voronoi diagram (optional)
        If $dV$ is not available, we use the simple proxy $dV := dO$
    """

    def __init__(self, dO_m: np.ndarray, params: VoronoiParams, dV_m: Optional[np.ndarray] = None):
        self.dO = dO_m.astype(np.float64, copy=False)
        self.dV = dV_m.astype(np.float64, copy=False) if dV_m is not None else None
        self.params = params

    def rho(self) -> np.ndarray:
        """ Vectorized potential in [0,1] on the grid.
            NOTE: If $dV$ is not available, proxy it with $dV := dO$ (keeps a weak "skeleton-ish" scaling effect)
        """
        alpha = float(self.params.alpha)
        dO_max = float(self.params.dO_max)
        dO = self.dO
        dV = self.dV if self.dV is not None else dO
        # Eq. (1)-style potential, clipped to [0,1]
        a = np.clip(dO / dO_max, 0.0, 1.0)
        b = 1.0 - np.clip(dV / dO_max, 0.0, 1.0)
        rho = (a**alpha) * b
        # clamp numeric noise
        return np.clip(rho, 0.0, 1.0).astype(np.float64, copy=False)


# ----------------------------
# Collision: rectangle footprint sampled in vehicle frame
# ----------------------------

def make_rectangle_footprint_offsets(vehicle: VehicleParams, sample_step: float) -> np.ndarray:
    """ Return an (N,2) array of (dx,dy) offsets in the vehicle frame.
        Frame convention:
            - origin at rear axle center
            - +x forward, +y left
            - rectangle spans:
                x in [-rear_overhang, wheelbase + front_overhang]
                y in [-width/2, +width/2]

        We sample the *interior* of the rectangle on a regular lattice. This is simple and robust on occupancy grids.
    """
    step = float(sample_step)
    if step <= 0.0:
        raise ValueError("sample_step must be > 0")
    # rectangle bounds
    x0 = -float(vehicle.rear_overhang)
    x1 = float(vehicle.wheelbase + vehicle.front_overhang)
    y0 = -0.5 * float(vehicle.width)
    y1 = +0.5 * float(vehicle.width)
    # generate grid of points to sample
    xs = np.arange(x0, x1 + 1e-9, step, dtype=np.float64)
    ys = np.arange(y0, y1 + 1e-9, step, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    return np.ascontiguousarray(pts, dtype=np.float64)


def pose_is_free(p: Pose, grid: OccupancyGrid, footprint_offsets: Optional[np.ndarray]) -> bool:
    """ Check collision of a pose against the occupancy grid.
        If `footprint_offsets` is None, checks only the reference point.
        Otherwise, checks all transformed offsets.
    """
    ox, oy = grid.grid.origin_xy
    res = float(grid.grid.resolution)
    # fast path: reference point only
    if footprint_offsets is None:
        ix = int(math.floor((p.x - ox) / res))
        iy = int(math.floor((p.y - oy) / res))
        return not grid.is_occupied(ix, iy)
    c = math.cos(p.theta)
    s = math.sin(p.theta)
    occ = grid.occ
    H, W = occ.shape
    # Tight loop: avoid calling back into grid methods.
    for dx, dy in footprint_offsets:
        x = p.x + c * float(dx) - s * float(dy)
        y = p.y + s * float(dx) + c * float(dy)
        ix = int(math.floor((x - ox) / res))
        iy = int(math.floor((y - oy) / res))
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            return False
        if occ[iy, ix]:
            return False
    return True


# ----------------------------
# Discretization utilities
# ----------------------------

class Indexer:
    """ Discretizes (x,y,theta,dir) and provides a flat index for best-g arrays """

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid
        self.W = int(grid.width)
        self.H = int(grid.height)
        self.theta_bins = int(grid.grid.theta_bins)
        self.dtheta = TAU / float(self.theta_bins)

    def theta_to_bin(self, theta: float) -> int:
        t = wrap_angle_2pi(theta)
        k = int(math.floor(t / self.dtheta)) % self.theta_bins
        return k

    @staticmethod
    def dir_to_index(direction: int) -> int:
        return 0 if direction >= 0 else 1

    def pose_to_key(self, pose: Pose, direction: int) -> DiscreteKey:
        ix, iy = self.grid.world_to_grid(pose.x, pose.y)
        itheta = self.theta_to_bin(pose.theta)
        return DiscreteKey(ix=ix, iy=iy, itheta=itheta, direction=1 if direction >= 0 else -1)

    def key_to_flat(self, key: DiscreteKey) -> int:
        return self.flat_index(key.ix, key.iy, key.itheta, key.direction)

    def flat_index(self, ix: int, iy: int, itheta: int, direction: int) -> int:
        idir = self.dir_to_index(direction)
        # Layout matches the C++ kernel: [H,W,theta_bins,2]
        return int((((iy * self.W + ix) * self.theta_bins + itheta) * 2 + idir))


# ----------------------------
# Motion model (constant curvature bicycle)
# ----------------------------

class BicycleModel:
    def __init__(self, vehicle: VehicleParams):
        self.L = float(vehicle.wheelbase)

    def propagate(self, pose: Pose, steer: float, direction: int, ds: float) -> Pose:
        """ Constant steer over distance ds (signed by direction) """
        sigma = 1.0 if direction >= 0 else -1.0
        kappa = math.tan(float(steer)) / self.L
        x0, y0, th0 = pose.x, pose.y, pose.theta
        # if kappa is near 0, it's a straight line; else circular arc
        if abs(kappa) < 1e-9:
            x1 = x0 + sigma * ds * math.cos(th0)
            y1 = y0 + sigma * ds * math.sin(th0)
            th1 = th0
        else:
            dth = sigma * ds * kappa
            th1 = wrap_angle(th0 + dth)
            R = 1.0 / kappa
            x1 = x0 + sigma * (math.sin(th1) - math.sin(th0)) * R
            y1 = y0 - sigma * (math.cos(th1) - math.cos(th0)) * R
        return Pose(x=x1, y=y1, theta=th1)

    def rollout_end_if_free(
        self,
        pose: Pose,
        steer: float,
        direction: int,
        ds: float,
        n_substeps: int,
        grid: OccupancyGrid,
        footprint_offsets: Optional[np.ndarray],
    ) -> Optional[Pose]:
        """ Propagate and collision-check along the edge. Returns the endpoint Pose if collision-free, else None. """
        step = float(ds) / float(n_substeps)
        cur = pose
        for _ in range(int(n_substeps)):
            cur = self.propagate(cur, steer, direction, step)
            if not pose_is_free(cur, grid, footprint_offsets):
                return None
        return cur

    def rollout(self, pose: Pose, steer: float, direction: int, ds: float, n: int) -> List[Pose]:
        """ Return intermediate poses including endpoint (debug/visualization) """
        out: List[Pose] = []
        step = float(ds) / float(n)
        cur = pose
        for _ in range(int(n)):
            cur = self.propagate(cur, steer, direction, step)
            out.append(cur)
        return out


# ----------------------------
# Heuristics from the paper
# ----------------------------

class HolonomicWithObstacles2D:
    """ 2D Dijkstra cost-to-go map on the grid, ignoring non-holonomy; paper uses this to detect U-shaped obstacles/dead-ends """
    def __init__(self, grid: OccupancyGrid, cost_per_cell: Optional[np.ndarray] = None):
        self.grid = grid
        self.cost_per_cell = cost_per_cell      # optional additional per-cell cost (e.g., Voronoi rho)
        self._dist: Optional[np.ndarray] = None # computed per goal

    def compute(self, goal: Pose) -> None:
        w, h = self.grid.width, self.grid.height
        dist = np.full((h, w), np.inf, dtype=np.float64)
        gx, gy = self.grid.world_to_grid(goal.x, goal.y)
        # goal in obstacle => heuristic is unusable; keep inf and fallback at query-time
        if not self.grid.in_bounds(gx, gy) or self.grid.is_occupied(gx, gy):
            self._dist = dist
            return
        c1 = float(self.grid.grid.resolution)
        c2 = c1 * math.sqrt(2.0)
        nbrs = ((-1, 0, c1), (1, 0, c1), (0, -1, c1), (0, 1, c1), (-1, -1, c2), (-1, 1, c2), (1, -1, c2), (1, 1, c2))
        pq: List[Tuple[float, int, int]] = []
        dist[gy, gx] = 0.0
        heapq.heappush(pq, (0.0, gx, gy))
        while pq:
            d, x, y = heapq.heappop(pq)
            if d != dist[y, x]:
                continue
            for dx, dy, step in nbrs:
                nx, ny = x + dx, y + dy
                # if not self.grid.in_bounds(nx, ny) or self.grid.is_occupied(nx, ny):
                if self.grid.is_occupied(nx, ny): # `is_occupied` also checks bounds and treats out-of-bounds cells as obstacles
                    continue
                extra = 0.0
                if self.cost_per_cell is not None:
                    extra = float(self.cost_per_cell[ny, nx])
                nd = d + step * (1.0 + extra)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, nx, ny))
        self._dist = dist

    def cpp_view(self) -> np.ndarray:
        if self._dist is None:
            raise RuntimeError("compute() must be called before cpp_view()")
        return np.ascontiguousarray(self._dist, dtype=np.float64)

    def __call__(self, pose: Pose) -> float:
        if self._dist is None:
            return 0.0
        ix, iy = self.grid.world_to_grid(pose.x, pose.y)
        if not self.grid.in_bounds(ix, iy):
            return float("inf")
        return float(self._dist[iy, ix])


class NonHolonomicWithoutObstaclesTable:
    """ Goal-local heuristic table over (x, y, theta), ignoring obstacles - implements Dijkstra over a small goal-centered grid in goal frame
        - paper computes shortest path to (0,0,0) in a neighborhood, offline
    """
    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self._table: Optional[np.ndarray] = None  # [iy, ix, itheta]
        # PREVIOUSLY: (radius, res, theta_bins_local)
        self._meta: Optional[Tuple[float, float, int, float, int]] = None  # (R, res, theta_bins, dth, nxy)

    def build_offline(self) -> None:
        R = float(self.cfg.nonholonomic_table_xy_radius)
        res = float(self.cfg.nonholonomic_table_xy_res)
        dth = float(self.cfg.nonholonomic_table_theta_res)
        # local table dims
        nxy = int(math.ceil((2.0 * R) / res))
        if nxy % 2 == 0:
            nxy += 1
        theta_bins = int(round(TAU / dth))
        dth = TAU / float(theta_bins)

        def idx_to_xy(ix: int, iy: int) -> Tuple[float, float]:
            """ index <-> coordinate in goal frame """
            x = (ix - nxy // 2) * res
            y = (iy - nxy // 2) * res
            return x, y

        table = np.full((nxy, nxy, theta_bins), np.inf, dtype=np.float64)
        # Motion primitives (same as search): constant steering for one step_size in LOCAL metric
        #? NOTE: ds should align to res for table consistency; we use res here.
        model = BicycleModel(self.cfg.vehicle)
        steer_set = self._steer_set()
        # Goal at center cell, theta=0
        gx = nxy // 2
        gy = nxy // 2
        gk = 0
        table[gy, gx, gk] = 0.0
        pq: List[Tuple[float, int, int, int]] = [(0.0, gx, gy, gk)]
        heapq.heapify(pq)
        # Run Dijkstra outward by applying REVERSE dynamics. Instead of inverting bicycle exactly, we approximate by applying forward
            # primitives from each state and relaxing neighbors (works since costs are symmetric-ish for the obstacle-free heuristic)
        while pq:
            d, ix, iy, it = heapq.heappop(pq)
            if d != table[iy, ix, it]:
                continue
            x, y = idx_to_xy(ix, iy)
            th = wrap_angle(it * dth)
            cur = Pose(x=x, y=y, theta=th)
            # expand neighbors and relax costs
            for direction in (+1, -1): # include reverse in heuristic table
                for steer in steer_set:
                    nxt = model.propagate(cur, steer, direction, res)
                    nix = int(round(nxt.x / res)) + nxy // 2
                    niy = int(round(nxt.y / res)) + nxy // 2
                    if not (0 <= nix < nxy and 0 <= niy < nxy):
                        continue
                    nit = int(math.floor(wrap_angle_2pi(nxt.theta) / dth)) % theta_bins
                    # small reverse penalty in heuristic can be set to 0 for admissibility as in this case
                    nd = d + res
                    if nd < table[niy, nix, nit]:
                        table[niy, nix, nit] = nd
                        heapq.heappush(pq, (nd, nix, niy, nit))
        self._table = table
        self._meta = (R, res, theta_bins, dth, nxy)

    def _steer_set(self) -> List[float]:
        m = int(self.cfg.steering_samples)
        max_steer = float(self.cfg.vehicle.max_steer)
        # return symmetric samples including 0
        return np.linspace(-max_steer, max_steer, m, dtype=np.float64).tolist()

    def cpp_view(self) -> Tuple[np.ndarray, float, float, float, int, int]:
        """ Return contiguous table and metadata for the C++ kernel """
        if self._table is None or self._meta is None:
            raise RuntimeError("build_offline() must be called before cpp_view()")
        R, res, theta_bins, dth, nxy = self._meta
        table = np.ascontiguousarray(self._table, dtype=np.float64)
        return table, float(R), float(res), float(dth), int(nxy), int(theta_bins)

    def __call__(self, pose: Pose, goal: Pose) -> float:
        # use Euclidean fallback if table not built yet
        if self._table is None or self._meta is None:
            dx, dy = pose.x - goal.x, pose.y - goal.y
            return math.hypot(dx, dy)
        R, res, theta_bins, dth, nxy = self._meta
        dx = pose.x - goal.x
        dy = pose.y - goal.y
        # Rotate into goal frame: R(-theta_g) * (pose - goal)
        # RgT = rot2d(-goal.theta)
        # v = np.array([pose.x - goal.x, pose.y - goal.y], dtype=np.float64)
        # xl, yl = RgT @ v
        # UPDATE: removed numpy overhead in favor of explicit rotation matrix application
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
        val = float(self._table[iy, ix, it])
        if not math.isfinite(val):
            return math.hypot(xl, yl)
        return val