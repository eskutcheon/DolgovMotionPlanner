# src/models/refiner.py

# Nonlinear optimization-based path smoothing (high-level flow)
# -------------------------------------------------------------------------------------------------
# Goal: improve a piecewise-linear / discrete planner path by moving interior (x,y) vertices to
#   minimize an objective while keeping endpoints fixed and staying collision-free
# Post-processing steps:
#   1. lightweight local relaxation smoothing
#   2. [OPTIONAL] nonlinear objective-based refinement with collision-aware anchoring
# -------------------------------------------------------------------------------------------------

from typing import List, Optional, Tuple
import math
import numpy as np
from scipy.optimize import minimize
# local module imports
from .models import OccupancyGrid
from src.structs import Pose, PathSmootherParams
from src.utils import TAU, SQRT2, wrap_angle, pose_is_free


class PathRefiner:
    """ Path post-processor for lightweight smoothing and objective-based anchored refinement """

    def __init__(self, occ_map: OccupancyGrid, cfg: PathSmootherParams, dO_m: np.ndarray, kappa_max: float,
                 footprint_offsets: Optional[np.ndarray], rho: Optional[np.ndarray] = None):
        """
            Args:
                occ_map: Occupancy grid used for bounds checks and world <=> grid transforms
                cfg: Smoothing/optimization weights and solver parameters
                dO_m: Obstacle distance field (meters), indexed as [y, x]
                kappa_max: Maximum allowed curvature magnitude used when recomputing kappa
                footprint_offsets: Robot footprint sample offsets (passed to collision checks)
                rho: Optional Voronoi (or corridor) potential field, indexed as [y, x]
        """
        self.map = occ_map
        self.cfg = cfg
        self.kappa_max = kappa_max
        self.dO = dO_m
        self.rho = rho
        self.footprint_offsets = footprint_offsets
        self.alpha = cfg.smoothing_alpha
        # precompute grid-field gradients once to make optimization evaluations cheap and avoid repeated finite differences in the objective loop
        res = float(self.map.grid.resolution)
        self._dO_dy, self._dO_dx = np.gradient(self.dO, res)
        self._rho_dx: Optional[np.ndarray] = None
        self._rho_dy: Optional[np.ndarray] = None
        if self.rho is not None:
            rdy, rdx = np.gradient(self.rho, res)
            self._rho_dx = rdx
            self._rho_dy = rdy

    @staticmethod
    def _compute_heading_curvature(path_xy: np.ndarray, kappa_max: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Compute heading and curvature for a polyline.
            Args:
                path_xy: Array of shape (N, 2) containing waypoint positions in world coordinates
                kappa_max: Clamp for curvature magnitude
            Returns:
                (theta, kappa): headings and curvatures, each of length N
        """
        n = int(path_xy.shape[0])
        theta = np.zeros(n, dtype=np.float64)
        kappa = np.zeros(n, dtype=np.float64)
        # compute headings via finite differences; for zero-length segments, repeat previous heading
        for i in range(1, n):
            dx = float(path_xy[i, 0] - path_xy[i - 1, 0])
            dy = float(path_xy[i, 1] - path_xy[i - 1, 1])
            theta[i] = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else theta[i - 1]
        theta[0] = theta[1] if n > 1 else 0.0
        # approximate curvature as heading change over arc length, $\kappa \approx \frac{\Delta \theta}{\Delta s}$
        for i in range(1, n - 1):
            ds = max(1e-4, 0.5 * (np.hypot(*(path_xy[i] - path_xy[i - 1])) + np.hypot(*(path_xy[i + 1] - path_xy[i]))))
            kappa[i] = max(-kappa_max, min(kappa_max, wrap_angle(float(theta[i + 1] - theta[i])) / ds))
        if n > 2:
            kappa[0], kappa[-1] = kappa[1], kappa[-2]
        return theta, kappa

    def _xy_to_path(self, xy: np.ndarray, ref_path: List[Pose]) -> List[Pose]:
        """ Convert optimized XY waypoints back to Pose list
            Args:
                xy: Array of shape (N, 2) with (x, y) in world coordinates
                ref_path: Reference Pose list (used to preserve endpoint theta/kappa)
            Returns:
                New Pose list with updated positions, recomputed theta/kappa, and endpoints preserved
        """
        # rebuild theta/kappa consistently from the optimized geometry; ensures the path remains kinematically feasible
        theta, kappa = self._compute_heading_curvature(xy, float(self.kappa_max))
        out = [Pose(x=float(xy[i, 0]), y=float(xy[i, 1]), theta=float(theta[i]), kappa=float(kappa[i])) for i in range(len(ref_path))]
        # keep endpoint orientation/curvature unchanged (anchors in pose-space)
        out[0] = Pose(x=float(xy[0, 0]), y=float(xy[0, 1]), theta=ref_path[0].theta, kappa=ref_path[0].kappa)
        out[-1] = Pose(x=float(xy[-1, 0]), y=float(xy[-1, 1]), theta=ref_path[-1].theta, kappa=ref_path[-1].kappa)
        return out

    def _objective_value(self, xy: np.ndarray) -> float:
        """ scalar objective value for a given path geometry - used for optimization """
        val, _ = self._objective_value_and_grad(xy)
        return val

    def _penalize_curvature_change(self, grad: np.ndarray, xy: np.ndarray) -> float:
        r""" curvature-continuity proxy: penalize changes in curvature (3rd finite difference)
            - proxy for curvature continuity that's cheaper to compute than exact curvature derivatives, since it only depends on vertex positions, not headings
            - computed as the sum of squared third-order differences: $\sum_i |p_{i-2} - 3p_{i-1} + 3p_i - p_{i+1}|^2$
            - gradient is added to the overall objective gradient to encourage smoother curvature profiles
        """
        w_cr = float(self.cfg.objective_w_curvature_rate)
        if w_cr <= 0.0 or len(xy) < 4:
            return 0.0
        # Compute third order differences: $p_{i-2} - 3p_{i-1} + 3p_i - p_{i+1}$
        dd3 = xy[:-3] - 3.0 * xy[1:-2] + 3.0 * xy[2:-1] - xy[3:]
        val = float(np.sum(dd3 * dd3))
        # Gradient for $|dd3|^2$ accumulates into the 4 points participating in each 3rd-diff stencil.
        grad[:-3] += 2.0 * w_cr * dd3
        grad[1:-2] -= 6.0 * w_cr * dd3
        grad[2:-1] += 6.0 * w_cr * dd3
        grad[3:] -= 2.0 * w_cr * dd3
        return val

    def _enforce_smoothness(self, grad: np.ndarray, xy: np.ndarray) -> float:
        r""" smoothness proxy: penalize second finite difference (discrete curvature) to suppress geometric wiggles
            - computed as the sum of squared second-order differences: $\sum_i |p_{i-1} - 2p_i + p_{i+1}|^2$
            - gradient is added to the overall objective gradient to encourage smoother paths
        """
        w_sm = float(self.cfg.objective_w_smooth)
        if w_sm <= 0.0 or len(xy) < 3:
            return 0.0
        # compute second order differences: $\Delta^{2} p_{i}  =  p_{i-1} - 2p_i + p_{i+1}$
        dd = xy[:-2] - 2.0 * xy[1:-1] + xy[2:]
        val = w_sm * float(np.sum(dd * dd)) #! might not need this multiplier
        # Gradient for $|dd|^2$ accumulates into the 3 points contributing to each diff stencil
        grad[:-2] += 2.0 * w_sm * dd
        grad[1:-1] -= 4.0 * w_sm * dd
        grad[2:] += 2.0 * w_sm * dd
        return val


    def _objective_value_and_grad(self, xy: np.ndarray) -> Tuple[float, np.ndarray]:
        r""" Objective and analytic gradient for nonlinear-optimized path refinement
            Args:
                xy: Array of shape (N, 2) containing candidate waypoint positions
            Returns:
                (val, grad) where grad w/ shape (N, 2) is the gradient of the objective w.r.t. each waypoint's (x, y) position
            Notes:
                The objective is a weighted sum of:
                - length: $\sum_i |p_{i+1} - p_i|$
                - smoothness/curvature proxy: $\sum_i |p_{i-1} - 2p_i + p_{i+1}|^2$
                - obstacle clearance penalty when $d_O < d_{safe}$
                - optional Voronoi/corridor potential $\sum_i \rho(p_i)$
        """
        val = 0.0
        grad = np.zeros_like(xy)
        eps = 1e-8
        # length term - $\sum_i |p_{i+1} - p_i|$ (with stable gradient via unit tangents)
        seg = xy[1:] - xy[:-1]
        seg_len = np.linalg.norm(seg, axis=1) + eps
        wl = float(self.cfg.objective_w_length)
        val += wl * float(np.sum(seg_len))
        unit = seg / seg_len[:, None]
        grad[:-1] -= wl * unit
        grad[1:] += wl * unit
        #& UPDATE - removed combined weighting of smoothness and curvature so that now smoothness is enforced with the 2nd diff and curvature change is enforced with the 3rd diff
        # smoothness term w/ 2nd order differences - suppress geometric oscillations
        val += self._enforce_smoothness(grad, xy)
        # curvature continuity proxy (3rd differences) - cheaper than exact curvature derivatives, encourages smoother curvature profiles
        val += self._penalize_curvature_change(grad, xy) #? NOTE: does slow down the planner a fair bit apparently
        # obstacle and Voronoi terms (sampled on grid with precomputed gradients) - turns spatial penalties into cheap pointwise lookups plus vector adds
        if len(xy) > 2:
            xi = np.floor((xy[1:-1, 0] - float(self.map.grid.origin_xy[0])) / float(self.map.grid.resolution)).astype(np.int64)
            yi = np.floor((xy[1:-1, 1] - float(self.map.grid.origin_xy[1])) / float(self.map.grid.resolution)).astype(np.int64)
            inb = (xi >= 0) & (yi >= 0) & (xi < self.map.width) & (yi < self.map.height)
            # penalize out-of-bounds states (keeps optimizer inside the map)
            if np.any(~inb):
                val += 1000.0 * float(np.sum(~inb))
            if np.any(inb):
                xib = xi[inb]
                yib = yi[inb]
                # Obstacle clearance penalty for $d_O < d_{safe}$ : $ w_O (d_{safe} - d_O)^2$
                dO = self.dO[yib, xib]
                margin = float(self.cfg.objective_smoothing_safe_distance_m) - dO
                active = margin > 0.0
                if np.any(active):
                    m = margin[active]
                    val += float(self.cfg.objective_w_obstacle) * float(np.sum(m * m))
                    local = np.where(inb)[0][active] + 1
                    grad[local, 0] += -2.0 * float(self.cfg.objective_w_obstacle) * m * self._dO_dx[yib[active], xib[active]]
                    grad[local, 1] += -2.0 * float(self.cfg.objective_w_obstacle) * m * self._dO_dy[yib[active], xib[active]]
                # optional corridor shaping via Voronoi potential $\rho$ if present
                if self.rho is not None and self._rho_dx is not None and self._rho_dy is not None:
                    rv = self.rho[yib, xib]
                    wv = float(self.cfg.objective_w_voronoi)
                    val += wv * float(np.sum(rv))
                    local = np.where(inb)[0] + 1
                    grad[local, 0] += wv * self._rho_dx[yib, xib]
                    grad[local, 1] += wv * self._rho_dy[yib, xib]
        # enforce fixed endpoints (position anchors) by zeroing gradients
        grad[0] = 0.0
        grad[-1] = 0.0
        return float(val), grad

    # def _optimize_xy(self, xy_init: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    #     """ Run a simple finite-difference gradient descent on the free (non-anchored) vertices """
    #     xy = np.array(xy_init, dtype=np.float64, copy=True)
    #     eps = max(1e-3, float(self.cfg.objective_smoothing_fd_eps))
    #     lr0 = max(1e-4, float(self.cfg.objective_smoothing_lr))
    #     for _ in range(max(1, int(self.cfg.objective_smoothing_iters))):
    #         base = self._objective_value(xy)
    #         grad = np.zeros_like(xy)
    #         for i in range(1, len(xy) - 1):
    #             if anchors[i]:
    #                 continue
    #             for a in (0, 1):
    #                 xy[i, a] += eps; p = self._objective_value(xy)
    #                 xy[i, a] -= 2.0 * eps; m = self._objective_value(xy)
    #                 xy[i, a] += eps
    #                 grad[i, a] = (p - m) / (2.0 * eps)
    #         if float(np.linalg.norm(grad[1:-1])) < 1e-6:
    #             break
    #         step = lr0
    #         accepted = False
    #         for _ in range(6):
    #             cand = xy - step * grad
    #             cand[anchors] = xy_init[anchors]
    #             cand[0], cand[-1] = xy_init[0], xy_init[-1]
    #             if self._objective_value(cand) <= base:
    #                 xy = cand
    #                 accepted = True
    #                 break
    #             step *= 0.5
    #         if not accepted:
    #             break
    #     return xy

    def _optimize_xy(self, xy_init: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """ Run a gradient-based solver on the free (non-anchored) vertices
            Args:
                xy_init: Initial waypoints, shape (N, 2)
                anchors: Boolean mask of length N; True means "hold this vertex fixed"
            Returns:
                Optimized waypoints array, shape (N, 2)
        """
        xy_base = np.array(xy_init, dtype=np.float64, copy=True)
        # define which vertices are allowed to move (exclude anchors and endpoints)
        free = (~anchors).copy()
        free[0] = False
        free[-1] = False
        free_idx = np.where(free)[0]
        if free_idx.size == 0:
            return xy_base
        z0 = xy_base[free_idx].reshape(-1) # flatten to 1D for the optimizer

        def fg(z: np.ndarray) -> Tuple[float, np.ndarray]:
            """ objective/gradient callback that unpacks z into an array for evaluation, then returns objective and gradient w.r.t. free vertices """
            xy = np.array(xy_base, copy=True)
            xy[free_idx] = z.reshape(-1, 2)
            f, g = self._objective_value_and_grad(xy)
            return f, g[free_idx].reshape(-1)

        # run L-BFGS-B (fast for smooth medium-sized problems; uses analytic gradients)
        res = minimize(
            lambda z: fg(z)[0],
            z0,
            jac=lambda z: fg(z)[1],
            method="L-BFGS-B",
            options={
                "maxiter": int(self.cfg.objective_solver_maxiter),
                "ftol": float(self.cfg.objective_solver_tol),
            },
        )
        if not res.success and res.x is None:
            return xy_base
        xy_base[free_idx] = np.asarray(res.x, dtype=np.float64).reshape(-1, 2)
        return xy_base


    def _objective_refine_path(self, path: List[Pose]) -> List[Pose]:
        """ Main loop for objective-based path refinement with collision-aware anchoring and optional point reduction for long paths """
        xy0_full = np.array([[p.x, p.y] for p in path], dtype=np.float64)
        # performance guard: optimize a reduced set of vertices for long paths, then lift back by interpolation
        n_max = max(10, int(self.cfg.objective_max_points))
        if len(path) > n_max:
            idx = np.unique(np.linspace(0, len(path) - 1, n_max, dtype=int))
            base_path = [path[int(i)] for i in idx]
            xy0 = xy0_full[idx]
        else:
            idx = None
            base_path = path
            xy0 = xy0_full
        # initialize anchors endpoints
        anchors_base = np.zeros(len(base_path), dtype=bool)
        anchors_base[0] = True
        anchors_base[-1] = True
        # iteratively refine, do collision checks, then add anchors near collisions
        for _ in range(max(1, int(self.cfg.smoothing_anchor_rounds))):
            xy_opt = self._optimize_xy(xy0, anchors_base)
            if idx is not None: # interpolate back to full resolution
                t_full = np.linspace(0.0, 1.0, len(path))
                t_red = np.linspace(0.0, 1.0, len(base_path))
                x_full = np.interp(t_full, t_red, xy_opt[:, 0])
                y_full = np.interp(t_full, t_red, xy_opt[:, 1])
                cand = self._xy_to_path(np.column_stack((x_full, y_full)), path)
            else:
                cand = self._xy_to_path(xy_opt, path)
            # validate feasibility (collision-free under footprint)
            coll_idx = [i for i, p in enumerate(cand) if not pose_is_free(p, self.map, self.footprint_offsets)]
            if not coll_idx:
                return cand
            # add anchors around colliding indices to prevent the optimizer from repeatedly pushing vertices into obstacles
            if idx is not None:
                # anchor nearest reduced samples corresponding to colliding full-resolution states
                scale = (len(base_path) - 1) / max(1, len(path) - 1)
                red_coll = np.clip(np.round(np.asarray(coll_idx, dtype=np.float64) * scale).astype(int), 0, len(base_path) - 1)
                for rc in red_coll:
                    anchors_base[max(0, rc - 1):min(len(anchors_base), rc + 2)] = True
            else:
                for ci in coll_idx:
                    anchors_base[max(0, ci - 1):min(len(anchors_base), ci + 2)] = True
        return list(path) # return the original path as a fallback


    def smooth_path(self, path: List[Pose]) -> List[Pose]:
        """ called by planners - run lightweight interpolation smoothing then optional objective refinement """
        out = list(path) # make a copy to modify in place
        # cheap local relaxation (windowed averaging) with collision acceptance
        passes = max(1, int(self.cfg.smoothing_passes))
        w = max(2, int(self.cfg.smoothing_window))
        for _ in range(passes):
            changed = False
            for i in range(1, len(out) - 1):
                lo = max(0, i - w)
                hi = min(len(out) - 1, i + w)
                p_prev, p_next = out[lo], out[hi]
                # blend current state toward its neighborhood average (small step for stability).
                cx = (1.0 - self.alpha) * out[i].x + self.alpha * 0.5 * (p_prev.x + p_next.x)
                cy = (1.0 - self.alpha) * out[i].y + self.alpha * 0.5 * (p_prev.y + p_next.y)
                # smooth heading via sin/cos averaging to handle wrap-around
                sth = np.sin(out[i].theta) + self.alpha * (np.sin(p_prev.theta) + np.sin(p_next.theta))
                cth = np.cos(out[i].theta) + self.alpha * (np.cos(p_prev.theta) + np.cos(p_next.theta))
                th = float(np.arctan2(sth, cth))
                # smooth and clamp curvature via direct averaging
                #? NOTE: could also do this via finite differences on the smoothed geometry after optimization for more exact curvature smoothing
                kappa = float((1.0 - self.alpha) * out[i].kappa + self.alpha * 0.5 * (p_prev.kappa + p_next.kappa))
                kappa = max(-float(self.kappa_max), min(float(self.kappa_max), kappa))
                candidate = Pose(x=float(cx), y=float(cy), theta=th, kappa=kappa)
                if pose_is_free(candidate, self.map, self.footprint_offsets):
                    out[i] = candidate
                    changed = True
            if not changed: # early exit if no changes were made
                break
        # optional nonlinear refinement (anchored + collision-aware) using the full nonlinear optimization approach
        if bool(self.cfg.use_objective_smoother) and len(out) >= 5:
            out = self._objective_refine_path(out)
        return out