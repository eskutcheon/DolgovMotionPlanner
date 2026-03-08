# src/dolgov_cbmp/planners/planner_base.py

from typing import List, Optional, Tuple, TypeAlias, Callable
import heapq
import time
import numpy as np
# local module imports
from dolgov_cbmp.structs import Pose, GoalSpec, PlannerStats, HybridNode, PlannerTick
from dolgov_cbmp.settings import PlannerConfig
from dolgov_cbmp.models import *
from dolgov_cbmp.utils import (
    SQRT2, wrap_angle, pose_is_free, compute_distance_to_obstacles_m, make_rectangle_footprint_offsets,
    rectangle_circumscribed_radius, build_orientation_binned_footprint_cache, compute_gvd_distance_m
)


# planner tick callback type alias for telemetry integration - accepts a PlannerTick object containing the current search state and statistics, and returns None
TickCallback: TypeAlias = Callable[[PlannerTick], None]
# stores the current frontier of the analytic beam search, sorted by a terminal score that combines distance to goal with heuristic guidance
SearchFrontierType: TypeAlias = List[Tuple[float, Pose, List[Pose], int, float, float]]


# TODO: maybe try to use a producer-consumer pattern so this can run on a separate thread while consuming actions from queue published to by the planner
    # key to topics by callback names (e.g., on_expand, on_rollout_failure) and have the planner publish relevant data to relevant topics
    # empty data stream would still have commands for simple actions like incrementing class variables
class PlannerEventStream:
    """ small event collector for planner stats and optional tick snapshots """
    def __init__(self, stats: PlannerStats, start_time_s: float, callback: Optional[TickCallback], stride: int):
        self.stats = stats
        self.start_time_s = float(start_time_s)
        self.callback = callback
        self.stride = max(1, int(stride))
        self.explored_since_tick: List[Pose] = []
        # TODO: improve storage efficiency by only preserving the indices of relevant explored edges and reconstructing the poses from the nodes list
        self.explored_edges_since_tick: List[Tuple[Pose, Pose]] = []
        self.collisions_since_tick: List[Pose] = []
        self.pruned_trajectories_since_tick: List[List[Pose]] = []
        self.latest_analytic_shot: List[Pose] = []

    def on_expand(self, pose: Pose, open_size: int) -> None:
        self.stats.expanded += 1
        self.stats.goal_checks += 1
        self.stats.max_open_size = max(self.stats.max_open_size, int(open_size))
        self.explored_since_tick.append(pose)

    def on_explored_edge(self, p0: Pose, p1: Pose) -> None:
        self.explored_edges_since_tick.append((p0, p1))

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

    def emit_tick(self, force: bool, cur_node: HybridNode, cur_pose: Pose, trajectory: List[Pose], open_size: int) -> None:
        if any((
            self.callback is None,
            (not force and (self.stats.expanded % self.stride) != 0),
            # might take this out as an unnecessary (for now) safeguard
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
                best_f=float(cur_node.f),
                best_g=float(cur_node.g),
                pose=cur_pose,
                best_pose=cur_node.pose,
                trajectory=trajectory,
                explored_poses=list(self.explored_since_tick),
                explored_edges=list(self.explored_edges_since_tick),
                collision_poses=list(self.collisions_since_tick),
                pruned_trajectories=list(self.pruned_trajectories_since_tick),
                analytic_shot=list(self.latest_analytic_shot),
            )
        )
        self.explored_since_tick.clear()
        self.collisions_since_tick.clear()
        self.explored_edges_since_tick.clear()
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
        # TODO: planning to keep kappa_bins as part of the grid spec to mirror theta_bins, but need to finish integration of the new WorldModel
        self.indexer = Indexer(occ_grid, kappa_bins=occ_grid.grid.kappa_bins, kappa_max=config.curvature.kappa_max)
        self.model = BicycleModel(config.vehicle)
        Vehicle = config.vehicle
        # Curvature-rate controls $u = \frac{d\kappa}{ds}$
        m = int(config.curvature.kappa_rate_samples)
        u_max = float(config.curvature.kappa_rate_max)
        # curvature samples used to generate edges during search ($u \in \{-\kappa_{max}, 0, +\kappa_{max}\}$)
        self.u_set = np.linspace(-u_max, u_max, m, dtype=np.float64) if m > 1 else np.array([0.0], dtype=np.float64)
        # Footprint offsets for collision checking.
        self.footprint_offsets: Optional[np.ndarray] = None
        if use_rectangle_footprint:
            # using 0.25 multiplier to get denser sampling than old default (0.5*resolution)
            step = config.footprint_sample_step or 0.25 * float(occ_grid.grid.resolution)
            self.footprint_offsets = make_rectangle_footprint_offsets(
                Vehicle.wheelbase, Vehicle.width, Vehicle.front_overhang, Vehicle.rear_overhang, step
            )
        # Map-dependent fields (goal-independent) can be cached safely.
        self._dO = compute_distance_to_obstacles_m(self.map.occ, self.map.grid.resolution)
        self._dV: Optional[np.ndarray] = None
        self._rho: Optional[np.ndarray] = None
        if use_voronoi_edge_cost and config.weights.voronoi_weight > 0.0:
            # TODO: consider making compute_gvd_distance into a class method of VoronoiField and having it persist rather than instantiating just for rho
            #   making it persist is likely necessary for later if we move into dynamic rho fields that depend on the current state of the search
            #       (e.g., learned cost-to-go or dynamic obstacles); for now we compute it once below since it's a purely geometric property of the map
            self._dV = compute_gvd_distance_m(occ_grid.occ, occ_grid.grid.resolution)
            self._rho = VoronoiField(self._dO, self.cfg.heuristics.voronoi_alpha, self.cfg.heuristics.voronoi_dO_max, dV_m=self._dV).rho
        self.refiner = PathRefiner(self.map, self.cfg.smoother, self._dO, self.cfg.curvature.kappa_max, footprint_offsets=self.footprint_offsets, rho=self._rho)
        self.goal_shot_mode = str(getattr(self.cfg.connector, "mode", "beam")).lower().strip()
        # Conservative collision gate radius (distance-transform) + cached footprint per theta bin
        res = float(self.map.grid.resolution)
        # print("SANITY CHECK: map resolution: ", res)
        radius = rectangle_circumscribed_radius(Vehicle.wheelbase, Vehicle.width, Vehicle.front_overhang, Vehicle.rear_overhang)
        self._gate_radius_m = 0.5 * radius + 0.5 * res * SQRT2
        self._exact_margin_m = float(res)
        self._footprint_cache: Optional[List[np.ndarray]] = None
        if self.footprint_offsets is not None:
            self._footprint_cache = build_orientation_binned_footprint_cache(self.footprint_offsets, res, int(occ_grid.grid.theta_bins), dilate_cells=1)
        # Non-holonomic goal-local heuristic table is goal-independent and can be cached
        if self.cfg.heuristics.use_nonholonomic:
            self._nonhol = NonHolonomicWithoutObstaclesTable(config)
            self._nonhol.build_offline()


    def plan(
        self, start: Pose, goal: GoalSpec, max_expansions: int = 200_000,
        tick_callback: Optional[TickCallback] = None, tick_stride: int = 100,
    ) -> Tuple[List[Pose], PlannerStats]:
        raise NotImplementedError("HybridAStarPlannerBase is an abstract base class; subclasses should implement plan()")

    def _smooth_path(self, path: List[Pose]) -> List[Pose]:
        """ Lightweight post-search smoothing pass on (x,y,theta,kappa) with collision safeguards """
        if not self.cfg.use_path_smoothing or len(path) < 5:
            return path
        return self.refiner.smooth_path(path)

    def _heuristic(self, pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D) -> float:
        # Paper uses max(h_holonomic, h_nonholonomic)
        h_hol = h2d(pose)
        if self.cfg.heuristics.use_nonholonomic:
            h_nh = self._nonhol(pose, goal.pose)
            return max(float(h_hol), float(h_nh))
        return float(h_hol)

    def _build_goal_heuristics(self, goal: GoalSpec) -> HolonomicWithObstacles2D:
        # h2d = HolonomicWithObstacles2D(self.map, cost_per_cell=self._rho, dO_m=self._dO, min_clearance_m=self._gate_radius_m)
        # for mazes/corridors, don't prune cells by circumscribed radius here; let the continuous collision checker handle feasibility
            #? NOTE: test with `test_python_backend_can_pass_through_gap`
        h_cost = self._rho
        if self._rho is not None and self.cfg.weights.voronoi_weight > 0.0 and self.cfg.heuristics.use_voronoi:
            # keep 2D heuristic consistent with the edge cost - edge uses $$w_V * \int \rho ds$$
            h_cost *= float(self.cfg.weights.voronoi_weight)
        h2d = HolonomicWithObstacles2D(
            self.map,
            cost_per_cell=h_cost,
            dO_m=self._dO,
            min_clearance_m=self._gate_radius_m, #0.0
            soft_clearance_m = self.cfg.heuristics.h2d_soft_clearance_m,
            soft_clearance_weight = self.cfg.heuristics.h2d_soft_clearance_weight
        )
        h2d.compute(goal.pose)
        return h2d

    def _goal_reached(self, pose: Pose, goal: GoalSpec) -> bool:
        dx = pose.x - goal.pose.x
        dy = pose.y - goal.pose.y
        # checking squared Euclidean distance against squared position tolerance
        if (dx * dx + dy * dy) > goal.pos_tol**2:
            return False
        dth = wrap_angle(pose.theta - goal.pose.theta)
        return abs(dth) <= goal.theta_tol

    def _edge_cost(self, ds: float, direction: int, prev_direction: int, u: float, prev_u: float, rho_int: float) -> float:
        W = self.cfg.weights
        c = ds
        if direction < 0:
            c *= W.reverse_penalty
        # TODO: investigate adding the switch penalty as a kronecker delta term in the integration cost function
        if direction != prev_direction:
            c += W.switch_dir_penalty # hard to believe but any multiplicative factor up to 4x doesn't really help
        # curvature-rate regularization (smooth steering evolution)
        c += W.kappa_rate_weight * u**2 * ds + W.kappa_rate_change_weight * abs(u - prev_u)
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
            # self.map.view_grid(path) #! DEBUGGING
            #! FIXME: might want to make another argument for pose_is_free to optionally count out-of-bounds entries as collisions
            all_coll = [p for p in path if not pose_is_free(p, self.map, self.footprint_offsets)]
            # print(f"Validating path with {len(path)} poses, {len(all_coll)} in collision, starting from goal:")
            for coll in all_coll[::-1]:  # print in reverse order (from start to goal)
                ix, iy = self.map.world_to_grid(coll.x, coll.y)
                print("Collision at pose: ", coll, " - grid indices: ", (iy, ix), "dO at cell: ", self._dO[iy, ix])
            return len(all_coll) == 0
        return all(pose_is_free(p, self.map, self.footprint_offsets) for p in path)

    def _should_try_analytic(self, pose: Pose, goal: Pose, expanded: int) -> bool:
        # Centralize enable/disable so it behaves consistently across modes
        if not self.cfg.use_analytic_connector:
            return False
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
        kappa_ratio = min(1.0, abs(float(pose.kappa)) / max(1e-6, float(self.cfg.curvature.kappa_max)))
        ds /= (1.0 + float(step_cfg.curvature_slowdown_gain) * kappa_ratio)
        ds = min(max(ds, ds_min), ds_max) # clamp to [ds_min, ds_max]
        return ds

    def _pose_is_free_fast(self, pose: Pose) -> bool:
        """ helper method for calling the model's pose_is_free_fast with all the necessary parameters from the planner """
        ix, iy = self.map.world_to_grid(pose.x, pose.y)
        if not self.map.in_bounds(ix, iy):
            return False
        return self.model.pose_is_free_fast(
            pose, self.map, self.footprint_offsets, dO=float(self._dO[iy, ix]),
            gate_radius_m=self._gate_radius_m, exact_check_margin_m=self._exact_margin_m,
            footprint_cache=self._footprint_cache, theta_bins=self.map.grid.theta_bins,
        )


    def _terminal_score(self, pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D) -> float:
        dp = float(np.hypot(pose.x - goal.pose.x, pose.y - goal.pose.y))
        dth = abs(wrap_angle(pose.theta - goal.pose.theta))
        score = sum(w * val for w, val in zip(self.cfg.connector.terminal_score_weights(), (dp, dth)))
        return score + self.cfg.weights.score_heuristic_weight * self._heuristic(pose, goal, h2d)


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
            self.cfg.curvature.kappa_max, self.map, self.footprint_offsets,
            dO_m=self._dO, gate_radius_m=self._gate_radius_m, exact_check_margin_m=self._exact_margin_m,
            footprint_cache=self._footprint_cache, theta_bins=self.map.grid.theta_bins,
            rho=rho,
        )
        if result is None and events is not None:
            events.on_rollout_failure(pose)
        return result


    def _try_goal_shot(
        self, start_pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D, events: Optional[PlannerEventStream] = None
    ) -> Optional[List[Pose]]:
        if not self.cfg.use_analytic_connector:
            return None
        if self.goal_shot_mode == "rs":
            return self._try_goal_shot_rs(start_pose, goal, h2d, events)
        elif self.goal_shot_mode == "beam":
            return self._try_goal_shot_beam(start_pose, goal, h2d, events)
        else:
            raise ValueError(f"Unknown goal shot mode: {self.goal_shot_mode}")

    # TODO: maybe make these global or util methods that accept all the class attributes they need?
    def _try_goal_shot_beam(
        self, start_pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D, events: Optional[PlannerEventStream] = None
    ) -> Optional[List[Pose]]:
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
                        #& UPDATE: now that direction is included in the key, we can just use that directly for duplicate detection instead of a separate tuple
                        key = self.indexer.pose_to_key(nxt, direction)
                        key_tuple = key.as_tuple()
                        if key_tuple in seen_keys:
                            continue
                        seen_keys.add(key_tuple)
                        cost = local_cost + self._edge_cost(ds, direction, prev_dir, u, prev_u, rho_int)
                        tscore = cost + self._terminal_score(nxt, goal, h2d)
                        nxt_path = path + [nxt]
                        candidates.append((tscore, nxt, nxt_path, direction, u, cost))
                        if tscore < best_score:
                            best_score, best_path = tscore, nxt_path
                if not candidates: # if no beam member produced any candidate, the connector is stuck
                    break
                # get best candidates across all current beam members
                beam = heapq.nsmallest(beam_width, candidates, key=lambda it: it[0])
                if not beam:
                    break
        if best_path and self._goal_reached(best_path[-1], goal):
            return best_path
        return None

    def _try_goal_shot_rs(
        self, start_pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D, events: Optional[PlannerEventStream] = None
    ) -> Optional[List[Pose]]:
        """ Reeds-Shepp analytic shot (collision-checked) """
        #! FIXME: need to align with paper by wiring up h2d into the Reeds-Shepp shot generation similarly to the beam search
        # clamp to avoid degenerate case for straight-line shots; also ensures that the step size is well-defined in the reeds_shepp_shot function
        kappa_max = max(self.cfg.curvature.kappa_max, 1e-6)
        turn_radius = 1.0 / kappa_max
        step = float(getattr(self.cfg.connector, "rs_step", 0.25))
        step = max(0.05, step)  # avoid degenerate sampling
        cand = reeds_shepp_shot(
            start=start_pose,
            goal=goal.pose,
            turning_radius=turn_radius,
            step_size=step,
            allow_reverse=self.cfg.allow_reverse,
        )
        if not cand:
            return None
        # enforce goal curvature basin (search state includes kappa) while leaving a smooth transition to the refiner
        cand[-1] = Pose(x=cand[-1].x, y=cand[-1].y, theta=cand[-1].theta, kappa=goal.pose.kappa)
        for p in cand:
            if not pose_is_free(p, self.map, self.footprint_offsets):
                return None
        # Return the relative segment to be spliced after the prefix (exclude the start pose)
        shot_path = cand[1:]
        if events is not None:
            events.on_analytic_shot(shot_path)
        return shot_path
