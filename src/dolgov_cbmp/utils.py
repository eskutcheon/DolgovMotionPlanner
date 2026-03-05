# src/dolgov_cbmp/utils.py
from typing import TYPE_CHECKING, Optional, Tuple, List
import math
import numpy as np

if TYPE_CHECKING:
    from dolgov_cbmp.models import OccupancyGrid


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

def theta_to_bin(theta: float, theta_bins: int, dtheta: Optional[float] = None) -> int:
    """ Map angle to bin index """
    if theta_bins <= 0:
        raise ValueError("theta_bins must be > 0")
    dtheta = dtheta or (TAU / float(theta_bins)) # default bin width = 2*pi / theta_bins
    t = wrap_angle_2pi(theta)
    return int(math.floor(t / dtheta)) % int(theta_bins)
    # return min(max(bin_idx, 0), theta_bins - 1)

def kappa_to_bin(kappa: float, kappa_min: float, kappa_max: float, dkappa: float, kappa_bins: int) -> int:
    """ Map curvature to bin index """
    k = max(kappa_min, min(kappa_max, kappa))
    bin_idx = int(math.floor((k - kappa_min) / dkappa))
    return min(max(bin_idx, 0), kappa_bins - 1)


def goal_reached(p: Tuple[float, float, float], goal: Tuple[float, float, float], pos_tol: float, theta_tol: float) -> bool:
    """ Check if pose p is within the position and orientation tolerances of the goal pose. """
    dx = p[0] - goal[0]
    dy = p[1] - goal[1]
    if dx * dx + dy * dy > pos_tol * pos_tol:
        return False
    dth = wrap_angle(p[2] - goal[2])
    return abs(dth) <= theta_tol

def generate_random_maze_grid(width: int, height: int, obstacle_prob: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
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


# --------------------------------------------------------
# Distance fields: $dO(x,y)$ and $dV(x,y)$
# --------------------------------------------------------

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
        c1, c2 = resolution, resolution * SQRT2
        # forward then backward passes
        fwd_neighbors = ((-1, 0, c1), (0, -1, c1), (-1, -1, c2), (1, -1, c2))
        min_pass(range(w), range(h), fwd_neighbors)
        bwd_neighbors = ((1, 0, c1), (0, 1, c1), (1, 1, c2), (-1, 1, c2))
        min_pass(range(w - 1, -1, -1), range(h - 1, -1, -1), bwd_neighbors)
        return dist

def _compute_gvd_distance_scipy(occ: np.ndarray, resolution: float) -> np.ndarray:
    r""" (Scipy version) approximate distance-to-GVD by extracting a ridge mask from dO and running a distance transform to that ridge """
    from scipy.ndimage import distance_transform_edt  # type: ignore
    occ = occ.astype(bool, copy=False)
    free = ~occ
    # indices of shape (2, H, W) giving nearest obstacle cell for each free cell
    _, indices = distance_transform_edt(free, return_indices=True) # only need nearest-obstacle indices - distances aren't used directly
    H, W = occ.shape
    nearest_idx = indices[0] * W + indices[1]  # flatten to 1D index for each cell
    ridge = np.zeros((H, W), dtype=bool)
    nbr_offsets = (-1,0,1)
    # mark ridge cells where the nearest obstacle changes between 8 neighboring free cells to approximate the GVD boundary
    for dy in nbr_offsets:
        for dx in nbr_offsets:
            if dx == 0 and dy == 0: # exclude center cell
                continue
            shifted_idx = np.roll(nearest_idx, shift=(dy, dx), axis=(0, 1))
            ridge |= (shifted_idx != nearest_idx) & free
    # remove wrap-around artifacts by clearing the borders of the ridge mask
    for i in (0, -1):
        ridge[i, :] = False
        ridge[:, i] = False
    dist_to_ridge = distance_transform_edt(~ridge).astype(np.float64) * resolution
    return dist_to_ridge

def _compute_gvd_distance_from_dO(dO_m: np.ndarray, resolution: float) -> np.ndarray:
    # fallback: extract ridge from dO and compute distance to that ridge
    dO = dO_m.astype(np.float64, copy=False)
    H, W = dO.shape
    ridge = np.zeros((H, W), dtype=bool)
    # skip map border for simplicity (borders are poor GVD indicators anyway)
    for iy in range(1, H - 1):
        for ix in range(1, W - 1):
            c = float(dO[iy, ix])
            if c <= 0.0:
                continue
            nbrs = dO[(iy - 1):(iy + 2), (ix - 1):(ix + 2)] #.ravel() # includes center cell, but that doesn't affect the max
            # strict local maxima or broad plateau maxima (within tiny epsilon)
            mx = float(np.max(nbrs))
            if c >= mx - 1e-9: # if current cell is roughly a local maximum in dO, mark as ridge cell
                # require at least two near-max neighbors to avoid isolated spikes
                if int(np.sum(nbrs >= (mx - 1e-6))) >= 3:
                    ridge[iy, ix] = True
    # fallback for sparse/no-obstacle maps - avoids all-zero ridge by seeding the map centerline
    if not np.any(ridge):
        ridge[H // 2, :] = True
        ridge[:, W // 2] = True
    # distance transform to nearest ridge cell
    return compute_distance_to_obstacles_m(ridge, resolution)

def compute_gvd_distance_m(dO_or_occ: np.ndarray, resolution: float) -> np.ndarray:
    r"""
        $dV(x,y)$: distance (meters) to the generalized Voronoi diagram (GVD) in a grid map
        Accepts either:
            - occ  : boolean occupancy grid (True = obstacle)
            - dO_m : distance-to-obstacles field in meters (0 on obstacles)
        NOTES:
        - a practical GVD proxy is the set of free cells separating distinct nearest-obstacle regions, approximated by
            1. computing the nearest obstacle cell for every free cell (via EDT return_indices),
            2. marking free cells whose neighbors have a different nearest-obstacle identity as GVD boundary,
            3. running an EDT to that boundary to get $dV(x,y)$ in meters
        - if SciPy is unavailable, we fall back to the older ridge-on-dO heuristic (less accurate in tight mazes)
    """
    arr = np.asarray(dO_or_occ)
    # Occupancy case: prefer the more faithful SciPy-based boundary extraction when possible.
    if arr.dtype in (bool, np.bool, np.bool_):
        occ = arr.astype(bool, copy=False)
        try:
            return _compute_gvd_distance_scipy(occ, resolution)
        except Exception:
            # fallback: extract ridge from dO and compute distance to that ridge
            dO = compute_distance_to_obstacles_m(occ, resolution).astype(np.float64, copy=False)
            return _compute_gvd_distance_from_dO(dO, resolution)
    # Distance-field case (this is what planner_base currently passes in).
    dO = arr.astype(np.float64, copy=False)
    return _compute_gvd_distance_from_dO(dO, resolution)


# --------------------------------------------------------
# Collision: rectangle footprint sampled in vehicle frame
# --------------------------------------------------------

def make_rectangle_footprint_offsets(wheelbase: float, width: float, front_overhang: float, rear_overhang: float, sample_step: float) -> np.ndarray:
    """ Return an (N,2) array of (dx,dy) offsets in the vehicle frame.
        Frame convention:
            - origin at rear axle center
            - +x forward, +y left
            - rectangle spans:
                x in [-rear_overhang, wheelbase + front_overhang]
                y in [-width/2, +width/2]
        We sample the *interior* points of the rectangle on a regular lattice - should be simple and robust on occupancy grids
    """
    step = float(sample_step)
    if step <= 0.0:
        raise ValueError("sample_step must be > 0")
    # rectangle bounds
    x0 = -float(rear_overhang)              # start of the rectangle footprint behind the rear axle
    x1 = float(wheelbase + front_overhang)  # end of the rectangle footprint in front of the rear axle
    y0 = -0.5 * float(width)
    y1 = +0.5 * float(width)
    # generate grid of points to sample
    xs = np.arange(x0, x1 + 1e-9, step, dtype=np.float64)
    ys = np.arange(y0, y1 + 1e-9, step, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    return np.ascontiguousarray(pts, dtype=np.float64)


def rectangle_circumscribed_radius(wheelbase, width, front_overhang, rear_overhang) -> float:
    """ Conservative radius (meters) of the smallest circle centered at the rear axle origin, fully containing the footprint rectangle"""
    #? NOTE: does a purely geometric "worst‑case radius of the footprint" around the rear‑axle origin, independent of steering angle or kinematic constraints
    #   conservative but ignores how the body actually swings during a turn, so it can be both over‑conservative and still miss some off‑tracking behaviors
    x_front = float(wheelbase + front_overhang)
    y = 0.5 * float(width)
    return math.hypot(x_front, y)



def pose_is_free(p: 'Pose', grid: 'OccupancyGrid', footprint_offsets: Optional[np.ndarray]) -> bool:
    """ Check collision of a pose against the occupancy grid.
        If `footprint_offsets` is None, checks only the reference point.
        Otherwise, checks all transformed offsets.
    """
    #! FIXME: doesn't currently include the goal tolerance as a free region around the goal pose
        # need to consider that primarily for _validate_exact_path in the planners (which checks whether the path is collision-free all the way to the goal pose)
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



def build_orientation_binned_footprint_cache(
    footprint_offsets_m: np.ndarray,
    resolution_m: float,
    theta_bins: int,
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


def get_occupied_rectangles(occ: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """ return merged occupied rectangles as (ix0, iy0, ix1, iy1) w/ right endpoints excluded """
    rects: List[Tuple[int, int, int, int]] = []
    rows, cols = occ.shape
    # run_len[y, x] = number of consecutive True cells to the right, starting at (y, x)
    run_len = np.zeros_like(occ, dtype=np.int32)
    # compute run lengths by scanning right-to-left in a vectorized manner
    run_len[:, -1] = occ[:, -1].astype(np.int32)
    for x in range(cols - 2, -1, -1):
        run_len[:, x] = np.where(occ[:, x], run_len[:, x + 1] + 1, 0)
    # treat run_len as a histogram of widths for each row, finding maximal rectangles starting at each (y, x) with height >= 1
    for y in range(rows):
        # use a stack to compute maximal rectangles in this histogram
        stack = []  # list of (x_start, width)
        for x in range(cols + 1):
            w = run_len[y, x] if x < cols else 0
            start = x
            while stack and stack[-1][1] > w:
                x_start, w_prev = stack.pop()
                # For each rectangle height=1 at row y then expand downward by looking at min widths in subsequent rows
                max_h = _max_height_for_width(run_len, y, x_start, w_prev)
                if max_h > 0:
                    rects.append((x_start, y, x_start + w_prev, y + max_h))
                start = x_start
            if w > 0:
                stack.append((start, w))
    return rects

def _max_height_for_width(run_len: np.ndarray, y: int, x: int, width: int) -> int:
    # find max height from row y downward, where run_len[row, x] >= width
    col = run_len[y:, x]
    ok = col >= width # Boolean mask of rows where the run length supports this width
    # return first row where it fails, indicating the height
    if not ok[0]:
        return 0
    fail = np.argmax(~ok)
    return fail if fail > 0 else len(ok)