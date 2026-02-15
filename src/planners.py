# src/planners.py
# TODO: will be splitting this file into several new files in a new `planners` module later

from typing import Dict, List, Optional, Tuple, Literal, Union, TypeAlias, Callable
import heapq
import time
import numpy as np
# local module imports
from src.structs import Pose, GoalSpec, PlannerStats, HybridNode, PlannerConfig, DiscreteKey, PlannerTick
from src.models import *
from src.utils import (
    SQRT2, wrap_angle, pose_is_free, compute_distance_to_obstacles_m, make_rectangle_footprint_offsets,
    rectangle_circumscribed_radius, build_orientation_binned_footprint_cache, compute_gvd_distance_m
)

try:
    from src.cpp_kernels import run_search_cpp, CPP_AVAILABLE
except Exception:  # pragma: no cover
    CPP_AVAILABLE = False
    run_search_cpp = None  # type: ignore


# planner tick callback type alias for telemetry integration - accepts a PlannerTick object containing the current search state and statistics, and returns None
TickCallback: TypeAlias = Callable[[PlannerTick], None]
# type alias for the best-g structure, which can be either a dense numpy array or a sparse dict depending on the configuration
BestGType: TypeAlias = Union[np.ndarray, Dict[Tuple[int, int], float]]
# stores the current frontier of the analytic beam search, sorted by a terminal score that combines distance to goal with heuristic guidance
SearchFrontierType: TypeAlias = List[Tuple[float, Pose, List[Pose], int, float, float]]



class PlannerEventStream:
    """ small event collector for planner stats and optional tick snapshots """
    def __init__(self, stats: PlannerStats, start_time_s: float, callback: Optional[TickCallback], stride: int):
        self.stats = stats
        self.start_time_s = float(start_time_s)
        self.callback = callback
        self.stride = max(1, int(stride))
        self.explored_since_tick: List[Pose] = []
        self.collisions_since_tick: List[Pose] = []
        self.pruned_trajectories_since_tick: List[List[Pose]] = []
        self.latest_analytic_shot: List[Pose] = []

    def on_expand(self, pose: Pose, open_size: int) -> None:
        self.stats.expanded += 1
        self.stats.goal_checks += 1
        self.stats.max_open_size = max(self.stats.max_open_size, int(open_size))
        self.explored_since_tick.append(pose)

    def on_push(self) -> None:
        self.stats.pushed += 1

    def on_rollout_attempt(self, n_substeps: int) -> None:
        self.stats.collision_checks += int(n_substeps)

    def on_rollout_failure(self, pose: Pose) -> None:
        self.stats.failed_rollouts += 1
        self.collisions_since_tick.append(pose)

    def on_pruned_trajectory(self, path: List[Pose]) -> None:
        if len(path) >= 2:
            self.pruned_trajectories_since_tick.append(path)

    def on_analytic_shot(self, path: Optional[List[Pose]]) -> None:
        self.latest_analytic_shot = list(path) if path else []

    def emit_tick(self, force: bool, best_node: HybridNode, cur_pose: Pose, trajectory: List[Pose], open_size: int) -> None:
        if any((
            self.callback is None,
            (not force and (self.stats.expanded % self.stride) != 0),
            (force and not (self.explored_since_tick or self.collisions_since_tick))
        )):
            return
        self.callback(
            PlannerTick(
                iteration=self.stats.expanded,
                time_s=time.perf_counter() - self.start_time_s,
                expanded=self.stats.expanded,
                pushed=self.stats.pushed,
                open_size=int(open_size),
                collision_checks=self.stats.collision_checks,
                failed_rollouts=self.stats.failed_rollouts,
                best_f=float(best_node.f),
                best_g=float(best_node.g),
                pose=cur_pose,
                best_pose=best_node.pose,
                trajectory=trajectory,
                explored_poses=list(self.explored_since_tick),
                collision_poses=list(self.collisions_since_tick),
                pruned_trajectories=list(self.pruned_trajectories_since_tick),
                analytic_shot=list(self.latest_analytic_shot),
            )
        )
        self.explored_since_tick.clear()
        self.collisions_since_tick.clear()
        self.pruned_trajectories_since_tick.clear()
        # self.latest_analytic_shot.clear()
        self.stats.ticks_emitted += 1








class HybridAStarPlannerBase:
    """ Hybrid A* planner (Dolgov et al. style)
        Reference Hybrid A*:
            - heap open-set
            - continuous pose stored in node (Hybrid A* "continuous state in discrete nodes")

        Multi-thread preparation:
            - All persistent data on the planner is read-only after __init__.
            - Per-call state (open set, best-g arrays, goal-dependent heuristics) is allocated in `plan()` to allow concurrency
            - The C++ kernel releases the GIL during the search loop so Python threads can run concurrently.
    """

    def __init__(
        self,
        occ_grid: OccupancyGrid,
        config: PlannerConfig,
        use_rectangle_footprint: bool = True,
        use_voronoi_edge_cost: bool = True,
    ):
        self.map = occ_grid
        self.cfg = config
        self.indexer = Indexer(occ_grid, kappa_bins=occ_grid.grid.kappa_bins, kappa_max=occ_grid.grid.kappa_max)
        self.model = BicycleModel(config.vehicle)
        Vehicle = config.vehicle
        # Curvature-rate controls $u = \frac{d\kappa}{ds}$
        m = int(config.kappa_rate_samples)
        u_max = float(config.kappa_rate_max)
        # curvature samples used to generate edges during search ($u \in \{-\kappa_{max}, 0, +\kappa_{max}\}$)
        self.u_set = np.linspace(-u_max, u_max, m, dtype=np.float64) if m > 1 else np.array([0.0], dtype=np.float64)
        # Footprint offsets for collision checking.
        self.footprint_offsets: Optional[np.ndarray]
        if use_rectangle_footprint:
            # using 0.25 multiplier to get denser sampling than old default (0.5*resolution)
            step = config.footprint_sample_step or 0.25 * float(config.grid.resolution)
            self.footprint_offsets = make_rectangle_footprint_offsets(
                Vehicle.wheelbase, Vehicle.width, Vehicle.front_overhang, Vehicle.rear_overhang, step
            )
        else:
            self.footprint_offsets = None
        # Map-dependent fields (goal-independent) can be cached safely.
        self._dO = compute_distance_to_obstacles_m(self.map.occ, float(self.map.grid.resolution))
        self._dV: Optional[np.ndarray] = None
        self._rho: Optional[np.ndarray] = None
        if use_voronoi_edge_cost and float(config.weights.voronoi_weight) > 0.0:
            self._dV = compute_gvd_distance_m(self._dO, float(self.map.grid.resolution))
            self._rho = VoronoiField(self._dO, self.cfg.voronoi_alpha, self.cfg.voronoi_dO_max, dV_m=self._dV).rho
        #& UPDATE: new path smoothing object does actual nonlinear optimization for refinement
        self.refiner = PathRefiner(self.map, self.cfg.smoother, self._dO, self.cfg.kappa_max, footprint_offsets=self.footprint_offsets, rho=self._rho)
        # Conservative collision gate radius (distance-transform) + cached footprint per theta bin
        res = float(self.map.grid.resolution)
        # print("SANITY CHECK: map resolution: ", res)
        radius = rectangle_circumscribed_radius(Vehicle.wheelbase, Vehicle.width, Vehicle.front_overhang, Vehicle.rear_overhang)
        self._gate_radius_m = radius + 0.5 * res * SQRT2
        self._exact_margin_m = float(res)
        self._footprint_cache: Optional[List[np.ndarray]] = None
        if self.footprint_offsets is not None:
            self._footprint_cache = build_orientation_binned_footprint_cache(self.footprint_offsets, res, int(self.cfg.grid.theta_bins), dilate_cells=1)
        # Non-holonomic goal-local heuristic table is goal-independent and can be cached
        self._nonhol = NonHolonomicWithoutObstaclesTable(config)
        self._nonhol.build_offline()


    def plan(
        self, start: Pose, goal: GoalSpec, max_expansions: int = 200_000,
        tick_callback: Optional[TickCallback] = None, tick_stride: int = 100,
    ) -> Tuple[List[Pose], PlannerStats]:
        raise NotImplementedError("HybridAStarPlannerBase is an abstract base class; subclasses should implement plan()")


    def _heuristic(self, pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D) -> float:
        # Paper uses max(h_holonomic, h_nonholonomic)
        h_hol = h2d(pose)
        h_nh = self._nonhol(pose, goal.pose)
        return max(float(h_hol), float(h_nh))

    def _build_goal_heuristics(self, goal: GoalSpec) -> HolonomicWithObstacles2D:
        # h2d = HolonomicWithObstacles2D(self.map, cost_per_cell=self._rho, dO_m=self._dO, min_clearance_m=self._gate_radius_m)
        # for mazes/corridors, don't prune cells by circumscribed radius here; let the continuous collision checker handle feasibility
            #? NOTE: test with `test_python_backend_can_pass_through_gap`
        h2d = HolonomicWithObstacles2D(self.map, cost_per_cell=self._rho, dO_m=self._dO, min_clearance_m=0.0)
        h2d.compute(goal.pose)
        return h2d

    def _goal_reached(self, pose: Pose, goal: GoalSpec) -> bool:
        dx = pose.x - goal.pose.x
        dy = pose.y - goal.pose.y
        # checking squared Euclidean distance against squared position tolerance
        if (dx * dx + dy * dy) > goal.pos_tol**2:
            return False
        dth = wrap_angle(pose.theta - goal.pose.theta)
        # early exit on theta tolerance, so final check is for kappa
        if abs(dth) > goal.theta_tol:
            return False
        dkappa = pose.kappa - goal.pose.kappa
        return abs(dkappa) <= goal.kappa_tol

    def _edge_cost(self, ds: float, direction: int, prev_direction: int, u: float, prev_u: float, rho_int: float) -> float:
        W = self.cfg.weights
        c = ds
        if direction < 0:
            c *= W.reverse_penalty
        if direction != prev_direction:
            c += float(W.switch_dir_penalty)
        # curvature-rate regularization (smooth steering evolution)
        c += float(W.kappa_rate_weight) * float(u * u) * float(ds)
        c += float(W.kappa_rate_change_weight) * abs(float(u - prev_u))
        # integrated Voronoi cost along the edge ($ \rho \in \[0,1\] $) if enabled
        if rho_int > 0.0 and float(W.voronoi_weight) > 0.0:
            c += float(W.voronoi_weight) * float(rho_int)
        return c

    @staticmethod
    def _reconstruct(nodes: List[HybridNode], goal_id: int) -> List[Pose]:
        path: List[Pose] = []
        nid = int(goal_id)
        while nid >= 0:
            path.append(nodes[nid].pose)
            nid = nodes[nid].parent_id
        path.reverse()
        return path

    def _validate_path_exact(self, path: List[Pose], verbose: bool = False) -> bool:
        """ exact pose-by-pose footprint validation - starting at the goal and working backwards to the start (for better debugging of failure cases) """
        if verbose:
            all_coll = [p for p in path if not pose_is_free(p, self.map, self.footprint_offsets)]
            # print(f"Validating path with {len(path)} poses, {len(all_coll)} in collision, starting from goal:")
            for coll in all_coll[::-1]:  # print in reverse order (from start to goal)
                ix, iy = self.map.world_to_grid(coll.x, coll.y)
                print("Collision at pose: ", coll, " - grid indices: ", (iy, ix), "dO at cell: ", self._dO[iy, ix])
            return len(all_coll) == 0
        for p in path:
            if not pose_is_free(p, self.map, self.footprint_offsets):
                return False
        return True

    def _should_try_analytic(self, pose: Pose, goal: Pose, expanded: int) -> bool:
        # if self.cfg.analytic_every_n <= 0 or expanded % self.cfg.analytic_every_n != 0:
        analytic = self.cfg.analytic
        if analytic.every_n <= 0:
            return False
        dx = pose.x - goal.x
        dy = pose.y - goal.y
        # test if goal is within the Euclidean distance threshold for an analytic attempt to be worthwhile (e.g., Reeds-Shepp shot)
        if (dx * dx + dy * dy) > (analytic.max_distance ** 2):
            return False
        if analytic.use_adaptive_schedule:
            # near goal -> tighter interval (more frequent attempts), far from goal -> looser interval
            d_max = max(1e-6, float(analytic.max_distance))
            d_goal = float(np.hypot(dx, dy))
            ratio = min(max(d_goal / d_max, 0.0), 1.0)
            power = max(0.5, float(analytic.distance_power))
            min_i = max(1, int(analytic.min_interval))
            max_i = max(min_i, int(analytic.max_interval))
            interval = int(round(min_i + (max_i - min_i) * (ratio ** power)))
            return (expanded % max(1, interval)) == 0
        return (expanded % int(analytic.every_n)) == 0


    def _select_step(self, pose: Pose) -> float:
        """ Variable-resolution step (longer arcs in wide free space on Voronoi regions) """
        step_cfg = self.cfg.step_policy
        ds_min = self.cfg.step_size
        if not step_cfg.use_variable_step:
            return ds_min
        ds_max = float(step_cfg.step_size_max)
        beta = float(step_cfg.variable_step_beta)
        ix, iy = self.map.world_to_grid(pose.x, pose.y)
        if not self.map.in_bounds(ix, iy):
            return ds_min
        dO = float(self._dO[iy, ix])
        # Approximate dV with dO if GVD distance unavailable: $ds \approx \beta * (d_O + d_V) \approx 2 * \beta * d_O$
        dV = float(self._dV[iy, ix]) if self._dV is not None else dO
        ds = beta * (dO + dV) # equal to $2 \beta * dO$ if dV unavailable
        #& UPDATE: Additional improvement - reduce step in high-curvature plans to improve local maneuver quality
        kappa_ratio = min(1.0, abs(float(pose.kappa)) / max(1e-6, float(self.cfg.kappa_max)))
        ds /= (1.0 + float(step_cfg.curvature_slowdown_gain) * kappa_ratio)
        ds = min(max(ds, ds_min), ds_max) # clamp to [ds_min, ds_max]
        return ds

    def _pose_is_free_fast(self, pose: Pose) -> bool:
        """ Use gate + cached footprint when possible; fall back to exact footprint near obstacles """
        ix, iy = self.map.world_to_grid(pose.x, pose.y)
        if not self.map.in_bounds(ix, iy):
            return False
        if float(self._dO[iy, ix]) >= float(self._gate_radius_m):
            return True
        if self._footprint_cache is not None and float(self._dO[iy, ix]) >= float(self._gate_radius_m) + float(self._exact_margin_m):
            # Conservative cached check is safe here - (exact theta bin selection occurs inside rollout; using exact for safety)
            return True
        # tight / ambiguous: do exact footprint check
        is_free = pose_is_free(pose, self.map, self.footprint_offsets)
        return is_free


    def _terminal_score(self, pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D) -> float:
        dp = float(np.hypot(pose.x - goal.pose.x, pose.y - goal.pose.y))
        dth = abs(wrap_angle(pose.theta - goal.pose.theta))
        dk = abs(float(pose.kappa - goal.pose.kappa))
        score = sum(w * val for w, val in zip(self.cfg.connector.terminal_score_weights, (dp, dth, dk)))
        return score + 0.1 * self._heuristic(pose, goal, h2d)


    def _rollout_kinematic_model(
        self,
        pose: Pose,
        u: float,
        direction: int,
        ds: float,
        rho: Optional[np.ndarray],
        events: Optional[PlannerEventStream],
    ) -> Optional[Tuple[Pose, float]]:
        if events is not None:
            events.on_rollout_attempt(self.cfg.n_substeps)
        result = self.model.rollout(
            pose, u, direction, ds, self.cfg.n_substeps,
            self.cfg.kappa_max, self.map, self.footprint_offsets,
            dO_m=self._dO, gate_radius_m=self._gate_radius_m, exact_check_margin_m=self._exact_margin_m,
            footprint_cache=self._footprint_cache, theta_bins=self.cfg.grid.theta_bins,
            rho=rho,
        )
        if result is None and events is not None:
            events.on_rollout_failure(pose)
        return result


    #! FIXME: not in line with the stats and planner tick classes
    def _try_goal_shot(self, start_pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D, events: Optional[PlannerEventStream] = None) -> Optional[List[Pose]]:
        r""" Fast 'analytic-like' attempt: greedily roll out a short sequence of $(\sigma,u)$ to reach the goal tolerance """
        horizon = max(4, int(self.cfg.connector.connector_horizon))
        beam_width = max(2, int(self.cfg.connector.connector_beam_width))
        directions = (+1, -1) if self.cfg.allow_reverse else (+1,)
        # keep track of the path taken to reach each node in the beam for easy reconstruction if we find a valid connection to the goal
        # entries: (terminal_score, pose, path, prev_dir, prev_u, local_cost)
        #? NOTE: can't make this a heap since the Pose class doesn't have the dunder methods for comparison - might keep a heap just for tscores and index them that way
        beam: SearchFrontierType = [(self._terminal_score(start_pose, goal, h2d), start_pose, [], +1, 0.0, 0.0)]
        best_path: Optional[List[Pose]] = None
        best_score = float("inf")
        # bounded horizon search with a simple cost that combines terminal distance to goal with integrated Voronoi cost & curvature regularization
        for _ in range(horizon):
            candidates: SearchFrontierType = []
            seen_keys = set()
            for _, cur, path, prev_dir, prev_u, local_cost in beam:
                if self._goal_reached(cur, goal):
                    return path
                goal_dist = float(np.hypot(cur.x - goal.pose.x, cur.y - goal.pose.y))
                ds = min(self._select_step(cur), max(float(self.cfg.step_size), goal_dist))
                for direction in directions:
                    for u in self.u_set:
                        rollout_result = self._rollout_kinematic_model(cur, u, direction, ds, rho=None, events=events)
                        if rollout_result is None:
                            continue
                        nxt, rho_int = rollout_result
                        key = self.indexer.pose_to_key(nxt)
                        nxt_key = (*key.as_tuple(), direction)
                        if nxt_key in seen_keys:
                            continue
                        seen_keys.add(nxt_key)
                        cost = local_cost + self._edge_cost(ds, direction, prev_dir, u, prev_u, rho_int)
                        tscore = cost + self._terminal_score(nxt, goal, h2d)
                        nxt_path = path + [nxt]
                        candidates.append((tscore, nxt, nxt_path, direction, u, cost))
                        if tscore < best_score:
                            best_score, best_path = tscore, nxt_path
                if not candidates:
                    break
                beam = heapq.nsmallest(beam_width, candidates, key=lambda it: it[0])
        if best_path and self._goal_reached(best_path[-1], goal):
            return best_path
        return None


    def _smooth_path(self, path: List[Pose]) -> List[Pose]:
        """ Lightweight post-search smoothing pass on (x,y,theta,kappa) with collision safeguards """
        if not self.cfg.use_path_smoothing or len(path) < 5:
            return path
        return self.refiner.smooth_path(path)



def planner_factory(
    occ_grid: OccupancyGrid,
    config: PlannerConfig,
    backend: Literal["python", "cpp"] = "python",
    use_rectangle_footprint: bool = True,
    use_voronoi_edge_cost: bool = True,
) -> HybridAStarPlannerBase:
    """ Factory function to create a Hybrid A* planner with the requested backend. """
    if backend == "cpp":
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ backend requested but not available")
        return HybridAStarPlannerCpp(
            occ_grid,
            config,
            use_rectangle_footprint=use_rectangle_footprint,
            use_voronoi_edge_cost=use_voronoi_edge_cost,
        )
    elif backend == "python":
        return HybridAStarPlannerPython(
            occ_grid,
            config,
            use_rectangle_footprint=use_rectangle_footprint,
            use_voronoi_edge_cost=use_voronoi_edge_cost,
        )
    else:
        raise ValueError(f"Unknown planner backend: {backend}")



class HybridAStarPlannerPython(HybridAStarPlannerBase):
    # TODO: consider writing a wrapper class for best_g that abstracts away the dense vs sparse implementation details and provides get/set methods
        # would clean up the code a bit and encapsulate the logic better - could also potentially take over responsibilities of the `Indexer` class
    def _init_best_g(self) -> Tuple[bool, BestGType]:
        # determine whether to use dense best_g array or sparse dict based on grid size
        H, W = self.map.height, self.map.width
        theta_bins, kappa_bins = self.cfg.grid.theta_bins, self.cfg.grid.kappa_bins
        total_states = H * W * theta_bins * kappa_bins
        use_dense_best_g = total_states <= int(self.cfg.dense_best_g_max_states)
        # optionally split by last direction (+1/-1) - tiny overhead and better consistency with switch penalties
        direction_dim = 2 if bool(self.cfg.use_directional_dominance) else 1
        if use_dense_best_g:
            best_g = np.full((H, W, theta_bins, kappa_bins, direction_dim), np.inf, dtype=np.float32)
            return use_dense_best_g, best_g
        return use_dense_best_g, {}

    @staticmethod
    def _direction_bucket(direction: int) -> int:
        return 0 if int(direction) >= 0 else 1

    #? NOTE: DiscreteKey could easily be written out of this in favor of passing indices directly
    def _update_best_g(self, best_g: BestGType, use_dense_best_g: bool, key: 'DiscreteKey', g: float, direction: int):
        dir_idx = self._direction_bucket(direction) if bool(self.cfg.use_directional_dominance) else 0
        if use_dense_best_g:
            best_g[key.iy, key.ix, key.itheta, key.ikappa, dir_idx] = g
        else:
            best_g[(self.indexer.key_to_flat(key), dir_idx)] = g

    def _is_better_g(self, best_g: BestGType, use_dense_best_g: bool, key: 'DiscreteKey', g: float, direction: int) -> bool:
        dir_idx = self._direction_bucket(direction) if bool(self.cfg.use_directional_dominance) else 0
        if use_dense_best_g:
            return g <= float(best_g[key.iy, key.ix, key.itheta, key.ikappa, dir_idx]) + 1e-8
        else:
            flat = self.indexer.key_to_flat(key)
            return g <= float(best_g.get((flat, dir_idx), float("inf"))) + 1e-8


    def _attempt_analytic_connection(
        self, cur: HybridNode, goal: GoalSpec, h2d: HolonomicWithObstacles2D, nodes: List[HybridNode], nid: int,
        verbose=False, events: Optional[PlannerEventStream] = None,
    ) -> Tuple[Optional[List[Pose]], bool]:
        shot: Optional[List[Pose]] = self._try_goal_shot(cur.pose, goal, h2d, events=events)
        if shot is None:
            return None, False # return no path and success=False
        # splice shot onto reconstructed prefix
        prefix: List[Pose] = self._reconstruct(nodes, nid)
        full_path = self._smooth_path(prefix + shot)
        if full_path and not self._validate_path_exact(full_path, verbose=verbose):
            # shot found but rejected by validation step
            return full_path, False
        if len(full_path) == 0:
            print("[WARNING] Analytic shot succeeded but failed to reconstruct path; returning no path.")
        return full_path, True


    """ Hybrid A* planner as Python implementation of the search loop """
    def plan(
        self,
        start: Pose,
        goal: GoalSpec,
        max_expansions: int = 200_000,
        tick_callback: Optional[TickCallback] = None,
        tick_stride: int = 100,
    ) -> Tuple[List[Pose], PlannerStats]:
        stats = PlannerStats()
        stats.start_time_s = time.perf_counter()
        events = PlannerEventStream(stats, stats.start_time_s, tick_callback, tick_stride)
        # Goal-dependent holonomic-with-obstacles heuristic.
        h2d = self._build_goal_heuristics(goal)
        use_dense_best_g, best_g = self._init_best_g()
        open_heap: List[Tuple[float, float, int]] = []  # open set with keys (f, tie_key, node_id)
        nodes: List[HybridNode] = []
        start_key = self.indexer.pose_to_key(start)
        if not self.map.in_bounds(start_key.ix, start_key.iy) or not self._pose_is_free_fast(start):
            stats.end_time_s = time.perf_counter()
            print("\n[WARNING] Start pose is in collision or out of bounds; returning no path.")
            return [], stats
        h0 = self._heuristic(start, goal, h2d)
        n0 = HybridNode(key=start_key, pose=start, g=0.0, h=h0, f=h0, parent_id=-1, parent_action=(+1, 0.0))
        nodes.append(n0)
        # update best-g for the start node before pushing to open set
        self._update_best_g(best_g, use_dense_best_g, start_key, 0.0, +1)
        tie0 = -n0.g if bool(self.cfg.prefer_larger_g_tiebreak) else 0.0
        heapq.heappush(open_heap, (n0.f, tie0, 0))
        # stats.pushed += 1
        events.on_push()
        best_h_nid = 0
        best_h_val = n0.h
        while open_heap and stats.expanded < int(max_expansions):
            _, _, nid = heapq.heappop(open_heap)
            cur = nodes[nid]
            k = cur.key
            # dominance check: skip if no improvement
            if not self.map.in_bounds(k.ix, k.iy) or not self._is_better_g(best_g, use_dense_best_g, k, cur.g, cur.parent_action[0]):
                stats.dominated_skips += 1
                continue
            events.on_expand(cur.pose, len(open_heap))
            # if this node has the best h so far, save it for a potential last-chance analytic connection at the end (helps in sparse/open maps)
            if cur.h < best_h_val:
                best_h_val = float(cur.h)
                best_h_nid = int(nid)
            best_node = nodes[best_h_nid] # aliasing for better readability in tick callback
            cur_traj = self._reconstruct(nodes, nid) # to reuse in tick callback without reconstructing multiple times per expansion
            events.emit_tick(force=False, best_node=best_node, cur_pose=cur.pose, trajectory=cur_traj, open_size=len(open_heap))
            # goal check for early exit
            if self._goal_reached(cur.pose, goal):
                events.emit_tick(force=True, best_node=best_node, cur_pose=cur.pose, trajectory=cur_traj, open_size=len(open_heap))
                stats.end_time_s = time.perf_counter()
                path: List[Pose] = self._smooth_path(cur_traj)
                if path and not self._validate_path_exact(path, verbose=True):
                    print("\n[WARNING] Goal reached but final path failed exact collision check; returning no path.")
                    return [], stats
                if len(path) == 0:
                    print("\n[WARNING] Goal reached but failed to reconstruct path; returning no path.")
                return path, stats
            if self._should_try_analytic(cur.pose, goal.pose, stats.expanded):
                # TODO: feel like it might be worth building a queue of fields to increment and passing it to stats as keywords
                stats.analytic_attempts += 1
                shot_path, success = self._attempt_analytic_connection(cur, goal, h2d, nodes, nid, events=events)
                # if a shot was found, either restart loop and skip this node (if validation failed) or return path
                if shot_path is not None:
                    events.on_analytic_shot(shot_path)
                    if not success:
                        continue
                    stats.analytic_successes += 1
                    events.emit_tick(force=True, best_node=best_node, cur_pose=cur.pose, trajectory=cur_traj, open_size=len(open_heap))
                    stats.end_time_s = time.perf_counter()
                    return shot_path, stats
            prev_dir, prev_u = cur.parent_action
            # expand children via curvature-rate controls and direction
            directions = (+1, -1) if self.cfg.allow_reverse else (+1,)
            for direction in directions:
                ds = self._select_step(cur.pose)
                # for steer in self.steer_set:
                for u in self.u_set:
                    res = self._rollout_kinematic_model(cur.pose, u, direction, ds, rho=self._rho, events=events)
                    if res is None:
                        continue
                    nxt_pose, rho_int = res
                    nxt_key = self.indexer.pose_to_key(nxt_pose)
                    if not self.map.in_bounds(nxt_key.ix, nxt_key.iy):
                        stats.out_of_bounds_skips += 1
                        continue
                    # cost: length + direction penalties + optional Voronoi integral
                    g2 = cur.g + self._edge_cost(ds, direction, prev_dir, float(u), float(prev_u), float(rho_int))
                    # dominance check: only keep best g per discrete key
                    if not self._is_better_g(best_g, use_dense_best_g, nxt_key, g2, direction):
                        if events is not None:
                            events.on_pruned_trajectory([cur.pose, nxt_pose])
                        continue
                    h2 = self._heuristic(nxt_pose, goal, h2d)
                    n2 = HybridNode(key=nxt_key, pose=nxt_pose, parent_id=nid, g=g2, h=h2, f = g2 + h2, parent_action=(direction, u))
                    nodes.append(n2)
                    self._update_best_g(best_g, use_dense_best_g, nxt_key, g2, direction)
                    tie = -n2.g if bool(self.cfg.prefer_larger_g_tiebreak) else 0.0
                    heapq.heappush(open_heap, (n2.f, tie, len(nodes) - 1))
                    events.on_push()
        # last-chance bounded connector from best frontier node (helps sparse/open maps)
        if self.cfg.use_analytic_connector and 0 <= best_h_nid < len(nodes):
            shot_path, success = self._attempt_analytic_connection(nodes[best_h_nid], goal, h2d, nodes, best_h_nid, verbose=True, events=events)
            if shot_path is not None:
                events.on_analytic_shot(shot_path)
            if shot_path and success:
                events.emit_tick(
                    force=True,
                    best_node=nodes[best_h_nid],
                    cur_pose=nodes[best_h_nid].pose,
                    trajectory=self._reconstruct(nodes, best_h_nid),
                    open_size=len(open_heap)
                )
                stats.end_time_s = time.perf_counter()
                return shot_path, stats
        stats.end_time_s = time.perf_counter()
        print("\n[INFO] Planner failed to find a path within the expansion limit; returning no path.")
        return [], stats



#!!! FIXME: update for curvature inclusions
class HybridAStarPlannerCpp(HybridAStarPlannerBase):
    """ Hybrid A* planner with C++ backend """
    def __init__(
        self,
        occ_grid: OccupancyGrid,
        config: PlannerConfig,
        use_rectangle_footprint: bool = True,
        use_voronoi_edge_cost: bool = True,
    ):
        assert CPP_AVAILABLE, "C++ backend requested but not available"
        super().__init__(
            occ_grid,
            config,
            use_rectangle_footprint=use_rectangle_footprint,
            use_voronoi_edge_cost=use_voronoi_edge_cost,
        )


    def plan(
        self,
        start: Pose,
        goal: GoalSpec,
        max_expansions: int = 200_000,
        tick_callback: Optional[TickCallback] = None,
        tick_stride: int = 100,
    ) -> Tuple[List[Pose], PlannerStats]:
        if tick_callback is not None:
            RuntimeError("[ERROR] tick_callback is currently only supported in the Python planner loop.")
        stats = PlannerStats()
        stats.start_time_s = time.perf_counter()
        # Goal-dependent holonomic-with-obstacles heuristic.
        h2d = self._build_goal_heuristics(goal)
        # TODO: update C++ backend and kernels in cpp_kernels.py to handle curvature bins instead of direction
        nh_table, nh_R, nh_res, nh_dth, nh_nxy, nh_theta_bins = self._nonhol.cpp_view()
        path, s2 = run_search_cpp(
            occ_grid=self.map,
            cfg=self.cfg,
            start=start,
            goal=goal,
            #!! FIXME: deprecated steering_samples argument in C++ backend; replace with curvature-rate samples
            steer_set=self.steer_set,
            h2d_dist=h2d.cpp_view(),
            nonhol_table=nh_table,
            nonhol_meta=(nh_R, nh_res, nh_dth, nh_nxy, nh_theta_bins),
            rho=self._rho,
            footprint_offsets=self.footprint_offsets,
            max_expansions=max_expansions,
        )
        stats.expanded = s2.expanded
        stats.pushed = s2.pushed
        stats.collision_checks = s2.collision_checks
        stats.analytic_attempts = s2.analytic_attempts
        stats.analytic_successes = s2.analytic_successes
        stats.end_time_s = time.perf_counter()
        return path, stats
