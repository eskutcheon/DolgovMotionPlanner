# src/dolgov_cbmp/models/reeds_shepp.py

""" Reeds-Shepp / Dubins-style local connectors for Hybrid A* analytic expansions
    - optional module (for now) without hard dependencies on external RS libraries
        - try to use them if available at runtime, but fall back to a pure-Python implementation otherwise
    - returned path is a list of Pose in world coordinates, with estimated curvature (kappa) from heading changes
        - NOTE: curvature is clamped to +/- 1/turning_radius
    - instantaneous curvature jumps are allowed since downstream smoothing/refinement can enforce curvature continuity
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq
import math
import numpy as np
# local imports
from dolgov_cbmp.structs import Pose
from dolgov_cbmp.utils import wrap_angle


def reeds_shepp_shot(
    start: Pose,
    goal: Pose,
    turning_radius: float,
    step_size: float,
    allow_reverse: bool,
) -> Optional[List[Pose]]:
    """ Compute a collision-unaware connector path from start -> goal
        - If an external Reeds-Shepp solver is available, we use it. Otherwise, we fall back to a small
            obstacle-free A* in SE(2) using {L, S, R} primitives in forward/reverse.
        Returns:
            List[Pose] including the start pose as the first element and (approximately) the goal pose as the last.
    """
    rho = float(turning_radius)
    if rho <= 0.0:
        return None
    ds = max(1e-3, float(step_size))
    q0 = (float(start.x), float(start.y), float(start.theta))
    q1 = (float(goal.x), float(goal.y), float(goal.theta))
    # attempt to use external library bindings (optional)
    try:  # pragma: no cover
        import reeds_shepp  # type: ignore
        # commonly-used Python binding is named `reeds_shepp` and exposes `path_sample(q0, q1, rho, step)`
        if hasattr(reeds_shepp, "path_sample"):
            pts = reeds_shepp.path_sample(q0, q1, rho, ds)
            poses = _poses_from_samples(pts, rho)
            return _force_terminal_goal(poses, goal)
        # some wrappers expose a callable class with `sample_many`
        if hasattr(reeds_shepp, "ReedsSheppPath"):
            path = reeds_shepp.ReedsSheppPath(q0, q1, rho)
            pts = path.sample_many(ds)
            poses = _poses_from_samples(pts, rho)
            return _force_terminal_goal(poses, goal)
    except Exception:
        pass
    # pure Python fallback (local A* with curvature-bounded primitives)
    return _rs_fallback_astar(start, goal, rho, ds, allow_reverse)



def _force_terminal_goal(path: List[Pose], goal: Pose) -> List[Pose]:
    if not path:
        return [goal]
    # ensure final pose matches the goal position/heading exactly
    #   kappa left as-is since callers may overwrite the terminal kappa basin, and downstream smoothing can enforce curvature continuity
    last = path[-1]
    if (abs(last.x - goal.x) + abs(last.y - goal.y) + abs(wrap_angle(last.theta - goal.theta))) > 1e-6:
        path = list(path) + [Pose(x=goal.x, y=goal.y, theta=goal.theta, kappa=last.kappa)]
    return path


def _poses_from_samples(samples, turning_radius: float) -> List[Pose]:
    """ Convert a list of (x, y, theta) samples into Pose with estimated curvature """
    pts = list(samples) if samples is not None else []
    if not pts:
        return []
    xs = np.array([float(p[0]) for p in pts], dtype=np.float64)
    ys = np.array([float(p[1]) for p in pts], dtype=np.float64)
    th = np.array([float(p[2]) for p in pts], dtype=np.float64)
    k_max = 1.0 / float(turning_radius)
    kappa = np.zeros_like(th)
    # Estimate curvature from heading changes over arc length.
    for i in range(len(th) - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        ds = max(1e-6, math.hypot(dx, dy))
        dth = wrap_angle(float(th[i + 1] - th[i]))
        kappa[i] = max(-k_max, min(k_max, dth / ds))
    if len(th) > 1:
        kappa[-1] = kappa[-2]
    return [Pose(x=float(xs[i]), y=float(ys[i]), theta=float(th[i]), kappa=float(kappa[i])) for i in range(len(th))]



# Fallback A* approach

# TODO: may just want to handle tuples - there's not much benefit to _NodeKey beyond readability, and it adds overhead
@dataclass(frozen=True)
class _NodeKey:
    ix: int
    iy: int
    it: int

    def __lt__(self, other: '_NodeKey') -> bool:
        return (self.ix, self.iy, self.it) < (other.ix, other.iy, other.it)


def _rs_fallback_astar(start: Pose, goal: Pose, rho: float, ds: float, allow_reverse: bool) -> Optional[List[Pose]]:
    """ Obstacle-free A* in SE(2) using curvature-bounded primitives
        Fallback for when an optimal RS solver is not available - only meant for short-range analytic expansions
    """
    # Discretization (kept coarse to stay fast)
    xy_res = ds
    theta_bins = 72  # 5-degree bins
    dtheta = 2.0 * math.pi / float(theta_bins)

    def to_key(p: Pose) -> _NodeKey:
        ix = int(round(p.x / xy_res))
        iy = int(round(p.y / xy_res))
        it = int(math.floor(((p.theta % (2.0 * math.pi)) / dtheta))) % theta_bins
        return _NodeKey(ix, iy, it)

    def h(p: Pose) -> float:
        # simple admissible-ish heuristic: Euclidean distance + small heading term
        dx = p.x - goal.x
        dy = p.y - goal.y
        return math.hypot(dx, dy) + 0.2 * rho * abs(wrap_angle(p.theta - goal.theta))

    k = 1.0 / rho
    # primitive set: (segment_kappa, direction)
    dirs = (+1, -1) if allow_reverse else (+1,)
    prims: List[Tuple[float, int]] = []
    for direction in dirs:
        prims.append((0.0, direction))
        prims.append((+k, direction))
        prims.append((-k, direction))

    def step_pose(p: Pose, seg_kappa: float, direction: int) -> Pose:
        # midpoint integration with constant curvature over ds
        # TODO: test other integration methods here - this is simple but may be inaccurate for large ds or high curvature
        sigma = 1.0 if direction >= 0 else -1.0
        th0 = float(p.theta)
        km = float(seg_kappa)
        thm = th0 + 0.5 * sigma * km * ds
        x1 = float(p.x) + sigma * ds * math.cos(thm)
        y1 = float(p.y) + sigma * ds * math.sin(thm)
        th1 = (th0 + sigma * km * ds + math.pi) % (2.0 * math.pi) - math.pi
        return Pose(x=x1, y=y1, theta=th1, kappa=seg_kappa)

    start_k = to_key(start)
    goal_k = to_key(goal)
    open_heap: List[Tuple[float, float, _NodeKey]] = []
    g_best: Dict[_NodeKey, float] = {start_k: 0.0}
    parent: Dict[_NodeKey, Tuple[_NodeKey, Pose]] = {}
    pose_at: Dict[_NodeKey, Pose] = {start_k: start}
    heapq.heappush(open_heap, (h(start), 0.0, start_k))
    # Bound expansions to keep this a "shot" (not a full planner)
    max_exp = 8000
    while open_heap and max_exp > 0:
        max_exp -= 1
        f, g, key = heapq.heappop(open_heap)
        if g > g_best.get(key, float("inf")) + 1e-9:
            continue
        cur_pose = pose_at[key]
        # Termination: close enough in (x,y,theta)
        if key == goal_k:
            return _reconstruct_path(parent, pose_at, key)
        for seg_kappa, direction in prims:
            nxt = step_pose(cur_pose, seg_kappa, direction)
            nk = to_key(nxt)
            ng = g + ds
            if ng + 1e-9 < g_best.get(nk, float("inf")):
                g_best[nk] = ng
                parent[nk] = (key, nxt)
                pose_at[nk] = nxt
                heapq.heappush(open_heap, (ng + h(nxt), ng, nk))
    # If we didn't land exactly on the goal key, pick the closest reached key.
    if not g_best:
        return None
    best_key = min(g_best.keys(), key=lambda kk: _key_distance(kk, goal_k, xy_res, dtheta))
    path = _reconstruct_path(parent, pose_at, best_key)
    return _force_terminal_goal(path, goal)


def _key_distance(a: _NodeKey, b: _NodeKey, xy_res: float, dtheta: float) -> float:
    dx = (a.ix - b.ix) * xy_res
    dy = (a.iy - b.iy) * xy_res
    dth = abs((a.it - b.it) * dtheta)
    # TODO: remove hard-coded multiplier on heading term - meant to keep the heuristic in the same ballpark as the xy distance
    return math.hypot(dx, dy) + 0.2 * dth


def _reconstruct_path(parent: Dict[_NodeKey, Tuple[_NodeKey, Pose]], pose_at: Dict[_NodeKey, Pose], last: _NodeKey) -> List[Pose]:
    out: List[Pose] = []
    k = last
    # out = [pose_at[k] for k, _ in parent.items() if k in pose_at]
    out.append(pose_at[k])
    while k in parent:
        k_prev, pose = parent[k]
        k = k_prev
        out.append(pose_at[k])
    out.reverse()
    return out