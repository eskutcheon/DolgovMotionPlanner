

from typing import TYPE_CHECKING, Optional, Tuple, List
import math
import numpy as np

if TYPE_CHECKING:
    from structs import VehicleParams
    from models import OccupancyGrid, Pose



TAU = 2.0 * math.pi
SQRT2 = math.sqrt(2.0)


def wrap_angle(theta: float) -> float:
    """ wrap angle back to interval [-pi, pi) """
    x = (theta + math.pi) % TAU - math.pi
    return x

def wrap_angle_2pi(theta: float) -> float:
    """ wrap angle back to interval [0, 2pi) """
    return theta % TAU

def rot2d(theta: float) -> np.ndarray:
    """ 2D rotation matrix for angle theta (in radians) """
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def clamp(x: float, lo: float, hi: float) -> float:
    """ general utility for other float clamping """
    return lo if x < lo else hi if x > hi else x


def adjust_start_pose_for_clearance(
    start: 'Pose',
    grid: 'OccupancyGrid',
) -> 'Pose':
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
    return 'Pose'(new_x, new_y, start.theta, start.kappa)



def generate_random_maze_grid(width: int, height: int, obstacle_prob: float, seed: Optional[int] = None) -> np.ndarray:
    """ Generate a random occupancy grid where chosen cells and a few neighbors are marked as obstacles """

    rng = np.random.default_rng(seed)
    occ = np.zeros((height, width), dtype=bool)
    obstacle_prob /= int(math.sqrt(width * height)) # adjust prob to avoid overfilling

    def set_neighbor_as_obstacle(x: int, y: int, num_neighbors: int):
        """ recursive call to set neighboring cells as obstacles up to the number specified """
        if num_neighbors <= 0:
            return
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_indices = rng.permutation(len(directions))
        for idx in dir_indices:
            dx, dy = directions[idx]
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not occ[ny, nx]:
                occ[ny, nx] = True
                set_neighbor_as_obstacle(nx, ny, num_neighbors - 1)

    for y in range(height):
        for x in range(width):
            if rng.random() < obstacle_prob:
                occ[y, x] = True
                # also mark immediate neighbors to create larger obstacles
                num_nbrs = rng.integers(0, 11)
                set_neighbor_as_obstacle(x, y, num_nbrs)
    np.set_printoptions(threshold=100000)
    print(occ.astype(np.uint8))
    return occ



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

        # two-pass 8-neighbor chamfer with costs (1, sqrt(2))
        c1 = resolution
        c2 = resolution * math.sqrt(2.0)
        # forward then backward passes
        fwd_neighbors = ((-1, 0, c1), (0, -1, c1), (-1, -1, c2), (1, -1, c2))
        min_pass(range(w), range(h), fwd_neighbors)
        bwd_neighbors = ((1, 0, c1), (0, 1, c1), (1, 1, c2), (-1, 1, c2))
        min_pass(range(w - 1, -1, -1), range(h - 1, -1, -1), bwd_neighbors)
        return dist



# ----------------------------
# Collision: rectangle footprint sampled in vehicle frame
# ----------------------------

def make_rectangle_footprint_offsets(vehicle: 'VehicleParams', sample_step: float) -> np.ndarray:
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

#& NEW
def rectangle_circumscribed_radius(vehicle: 'VehicleParams') -> float:
    """ Conservative radius (meters) of the rectangular footprint around the rear-axle origin """
    x_front = float(vehicle.wheelbase + vehicle.front_overhang)
    x_rear = float(vehicle.rear_overhang)
    y = 0.5 * float(vehicle.width)
    return max(math.hypot(x_front, y), math.hypot(x_rear, y))


def pose_is_free(p: 'Pose', grid: 'OccupancyGrid', footprint_offsets: Optional[np.ndarray]) -> bool:
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



#& NEW
def build_orientation_binned_footprint_cache(
    footprint_offsets_m: np.ndarray,
    resolution_m: float,
    theta_bins: int,
    *,
    dilate_cells: int = 1,
) -> List[np.ndarray]:
    """ Precompute conservative integer grid-cell offsets for each theta bin.
        The cache is deliberately *slightly conservative* (via dilation) so it never misses collisions.
    """
    res = float(resolution_m)
    dtheta = TAU / float(theta_bins)
    caches: List[np.ndarray] = []
    off = np.ascontiguousarray(footprint_offsets_m, dtype=np.float64)
    for it in range(int(theta_bins)):
        th = (it + 0.5) * dtheta  # bin center
        c = math.cos(th)
        s = math.sin(th)
        xs = c * off[:, 0] - s * off[:, 1]
        ys = s * off[:, 0] + c * off[:, 1]
        dix = np.rint(xs / res).astype(np.int32, copy=False)
        diy = np.rint(ys / res).astype(np.int32, copy=False)
        # create set of unique (dx, dy) offsets
        cells = set(zip(dix.tolist(), diy.tolist()))
        if dilate_cells > 0:
            base = list(cells)
            for dx, dy in base:
                for ox in range(-dilate_cells, dilate_cells + 1):
                    for oy in range(-dilate_cells, dilate_cells + 1):
                        cells.add((dx + ox, dy + oy))
        # store as contiguous array
        arr = np.array(sorted(cells), dtype=np.int32)
        caches.append(np.ascontiguousarray(arr))
    return caches


def pose_is_free_cached_cells(p: 'Pose', grid: 'OccupancyGrid', cell_offsets: np.ndarray) -> bool:
    """ Fast conservative collision check using precomputed (dix, diy) offsets for a theta bin """
    ox, oy = grid.grid.origin_xy
    res = float(grid.grid.resolution)
    ix0 = int(math.floor((p.x - ox) / res))
    iy0 = int(math.floor((p.y - oy) / res))
    occ = grid.occ
    H, W = occ.shape
    for dx, dy in cell_offsets:
        ix = ix0 + int(dx)
        iy = iy0 + int(dy)
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            return False
        if occ[iy, ix]:
            return False
    return True