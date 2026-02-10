


from typing import Dict, List, Optional, Tuple, Literal, Union
import heapq
import time
import numpy as np
# local module imports
from src.structs import Pose, GoalSpec, PlannerStats, HybridNode, PlannerConfig, DiscreteKey #, PlannerTick, TickCallback
from src.models import (
    OccupancyGrid, Indexer, BicycleModel, VoronoiField, HolonomicWithObstacles2D, NonHolonomicWithoutObstaclesTable,
)
from src.utils import (
    SQRT2, wrap_angle, pose_is_free, compute_distance_to_obstacles_m, make_rectangle_footprint_offsets,
    rectangle_circumscribed_radius, build_orientation_binned_footprint_cache
)

try:
    from src.cpp_kernels import run_search_cpp, CPP_AVAILABLE
except Exception:  # pragma: no cover
    CPP_AVAILABLE = False
    run_search_cpp = None  # type: ignore

# TODO: (IMPORTANT) separate manual timing and replace with decorators from `timeit`




class HybridAStarPlannerBase:
    """ Hybrid A* planner (Dolgov et al. style)
        # TODO: EDIT LATER (legacy description)
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
        *,
        use_rectangle_footprint: bool = True,
        use_voronoi_edge_cost: bool = True,
    ):
        self.map = occ_grid
        self.cfg = config
        #& UPDATE: indexer now uses curvature parameters
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
        self._rho: Optional[np.ndarray] = None
        #& UPDATE: integrated the new curvature change penalty from PlannerWeights here
        if use_voronoi_edge_cost and float(config.weights.voronoi_weight) > 0.0:
            self._rho = VoronoiField(self._dO, self.cfg.voronoi_alpha, self.cfg.voronoi_dO_max).rho()
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


    def plan(self, start: Pose, goal: GoalSpec, *, max_expansions: int = 200_000) -> Tuple[List[Pose], PlannerStats]:
        raise NotImplementedError("HybridAStarPlannerBase is an abstract base class; subclasses should implement plan()")


    def _heuristic(self, pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D) -> float:
        # Paper uses max(h_holonomic, h_nonholonomic)
        h_hol = h2d(pose)
        h_nh = self._nonhol(pose, goal.pose)
        return max(float(h_hol), float(h_nh))

    def _build_goal_heuristics(self, goal: GoalSpec) -> HolonomicWithObstacles2D:
        # h2d = HolonomicWithObstacles2D(self.map, cost_per_cell=self._rho, dO_m=self._dO, min_clearance_m=self._gate_radius_m)
        # for mazes/corridors, don't prune cells by circumscribed radius here; let the continuous collision checker handle feasibility
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
        #& UPDATE: early exit on theta tolerance, so final check is for kappa
        if abs(dth) > goal.theta_tol:
            return False
        # return abs(dth) <= goal.theta_tol
        dkappa = pose.kappa - goal.pose.kappa
        return abs(dkappa) <= goal.kappa_tol

    def _edge_cost(self, ds: float, direction: int, prev_direction: int, u: float, prev_u: float, rho_int: float) -> float:
        c = ds #float(self.cfg.step_size)
        if direction < 0:
            c *= float(self.cfg.weights.reverse_penalty)
        if direction != prev_direction:
            c += float(self.cfg.weights.switch_dir_penalty)
        # curvature-rate regularization (smooth steering evolution)
        c += float(self.cfg.weights.kappa_rate_weight) * float(u * u) * float(ds)
        c += float(self.cfg.weights.kappa_rate_change_weight) * abs(float(u - prev_u))
        # integrated Voronoi cost along the edge ($ \rho \in \[0,1\] $) if enabled
        if rho_int > 0.0 and float(self.cfg.weights.voronoi_weight) > 0.0:
            c += float(self.cfg.weights.voronoi_weight) * float(rho_int)
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
            print(f"Validating path with {len(path)} poses, {len(all_coll)} in collision, starting from goal:")
            for coll in all_coll[::-1]:  # print in reverse order (from start to goal)
                ix, iy = self.map.world_to_grid(coll.x, coll.y)
                print("Collision at pose: ", coll, " - grid indices: ", (iy, ix), "dO at cell: ", self._dO[iy, ix])
            return len(all_coll) == 0
        for p in path:
            if not pose_is_free(p, self.map, self.footprint_offsets):
                return False
        return True

    # def _should_try_analytic(self, pose: Pose, goal: Pose, expanded: int) -> bool:
    #     if self.cfg.analytic_every_n <= 0:
    #         return False
    #     if expanded % self.cfg.analytic_every_n != 0:
    #         return False
    #     dx = pose.x - goal.x
    #     dy = pose.y - goal.y
    #     return (dx * dx + dy * dy) <= (self.cfg.analytic_max_distance ** 2)


    def _select_step(self, pose: Pose) -> float:
        """ Variable-resolution step (longer arcs in wide free space on Voronoi regions) """
        ds_min = float(self.cfg.step_size)
        if not bool(self.cfg.use_variable_step):
            return ds_min
        ds_max = float(self.cfg.step_size_max)
        beta = float(self.cfg.variable_step_beta)
        ix, iy = self.map.world_to_grid(pose.x, pose.y)
        if not self.map.in_bounds(ix, iy):
            return ds_min
        dO = float(self._dO[iy, ix])
        # Approximate dV with dO if GVD distance unavailable: ds ≈ beta*(dO + dV) ≈ 2*beta*dO
        ds = 2.0 * beta * dO
        if ds < ds_min:
            return ds_min
        if ds > ds_max:
            return ds_max
        return ds

    def _pose_is_free_fast(self, pose: Pose) -> bool:
        """ Use gate + cached footprint when possible; fall back to exact footprint near obstacles """
        ix, iy = self.map.world_to_grid(pose.x, pose.y)
        if not self.map.in_bounds(ix, iy):
            print("START POSE FAILING IN-BOUNDS CHECK: ", pose, "grid bounds: ", self.map.width, self.map.height)
            return False
        if float(self._dO[iy, ix]) >= float(self._gate_radius_m):
            return True
        if self._footprint_cache is not None and float(self._dO[iy, ix]) >= float(self._gate_radius_m) + float(self._exact_margin_m):
            # Conservative cached check is safe here - (exact theta bin selection occurs inside rollout; using exact for safety)
            return True
        # tight / ambiguous: do exact footprint check
        is_free = pose_is_free(pose, self.map, self.footprint_offsets)
        print("START POSE FAILING EXACT CHECK: ", pose, "WITH POSE_IS_FREE: ", is_free)
        print("START POSE INDICES: ", (iy, ix), "dO: ", self._dO[iy, ix], "gate radius: ", self._gate_radius_m)
        return is_free


    def _try_goal_shot(self, start_pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D) -> Optional[List[Pose]]:
        r""" Fast 'analytic-like' attempt: greedily roll out a short sequence of $(\sigma,u)$ to reach the goal tolerance """
        dx = start_pose.x - goal.pose.x
        dy = start_pose.y - goal.pose.y
        if (dx * dx + dy * dy) > float(self.cfg.analytic_max_distance) ** 2:
            return None
        cur = start_pose
        path: List[Pose] = []
        # prev_dir, prev_u = (+1, 0.0)
        # small, bounded horizon
        for _ in range(30):
            if self._goal_reached(cur, goal):
                return path
            best = None
            best_h = float("inf")
            ds = self._select_step(cur)
            for direction in ((+1, -1) if self.cfg.allow_reverse else (+1,)):
                for u in self.u_set:
                    res = self.model.rollout(
                        cur, float(u), int(direction), float(ds), int(self.cfg.n_substeps),
                        float(self.cfg.kappa_max), self.map, self.footprint_offsets,
                        dO_m=self._dO, gate_radius_m=self._gate_radius_m, exact_check_margin_m=self._exact_margin_m,
                        footprint_cache=self._footprint_cache, theta_bins=int(self.cfg.grid.theta_bins),
                        rho=None,
                    )
                    if res is None:
                        continue
                    nxt, _ = res
                    hh = self._heuristic(nxt, goal, h2d)
                    if hh < best_h:
                        best_h = hh
                        best = (nxt, int(direction), float(u))
            if best is None:
                return None
            cur, prev_dir, prev_u = best
            path.append(cur)
        return path if self._goal_reached(cur, goal) else None




def planner_factory(
    occ_grid: OccupancyGrid,
    config: PlannerConfig,
    *,
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
    def __init__(
        self,
        occ_grid: OccupancyGrid,
        config: PlannerConfig,
        *,
        use_rectangle_footprint: bool = True,
        use_voronoi_edge_cost: bool = True,
    ):
        super().__init__(
            occ_grid,
            config,
            use_rectangle_footprint=use_rectangle_footprint,
            use_voronoi_edge_cost=use_voronoi_edge_cost,
        )
        # sets self._use_dense_best_g and initialize best_g accordingly
        self.best_g: Union[np.ndarray, Dict[int, float]] = self._init_best_g()

    # TODO: consider writing a wrapper class for best_g that abstracts away the dense vs sparse implementation details and provides get/set methods
        # would clean up the code a bit and encapsulate the logic better - could also potentially take over responsibilities of the `Indexer` class

    def _init_best_g(self):
        # determine whether to use dense best_g array or sparse dict based on grid size
        H, W = self.map.height, self.map.width
        theta_bins, kappa_bins = self.cfg.grid.theta_bins, self.cfg.grid.kappa_bins
        total_states = int(H) * int(W) * int(theta_bins) * int(kappa_bins)
        self._use_dense_best_g = total_states <= int(self.cfg.dense_best_g_max_states)
        if self._use_dense_best_g:
            return np.full((H, W, theta_bins, kappa_bins), np.inf, dtype=np.float32)
        else:
            return {}

    #? NOTE: DiscreteKey could easily be written out of this in favor of passing indices directly
    def _update_best_g(self, key: 'DiscreteKey', g: float):
        if self._use_dense_best_g:
            self.best_g[key.iy, key.ix, key.itheta, key.ikappa] = g
        else:
            self.best_g[self.indexer.key_to_flat(key)] = g

    def _is_better_g(self, key: 'DiscreteKey', g: float) -> bool:
        if self._use_dense_best_g:
            return g <= float(self.best_g[key.iy, key.ix, key.itheta, key.ikappa]) + 1e-8
        else:
            flat = self.indexer.key_to_flat(key)
            return g <= float(self.best_g.get(flat, float("inf"))) + 1e-8


    """ Hybrid A* planner as Python implementation of the search loop """
    def plan(
        self,
        start: Pose,
        goal: GoalSpec,
        *,
        max_expansions: int = 200_000,
    ) -> Tuple[List[Pose], PlannerStats]:
        stats = PlannerStats()
        # TODO: if `stats` is kept, might wanna make a `PlannerStatsContext` to compute benchmarking (timing, etc) within a context manager for a more callback-like design
        stats.start_time_s = time.perf_counter()
        # Goal-dependent holonomic-with-obstacles heuristic.
        h2d = self._build_goal_heuristics(goal)
        # total_states = int(h) * int(w) * int(tb) * int(kb)
        # formerly: open set like (f, tie, node_id)
        open_heap: List[Tuple[float, int]] = []  # (f, node_id)
        nodes: List[HybridNode] = []
        start_key = self.indexer.pose_to_key(start) #, direction=+1)
        if not self.map.in_bounds(start_key.ix, start_key.iy) or not self._pose_is_free_fast(start): #, self.map, self.footprint_offsets):
            stats.end_time_s = time.perf_counter()
            print("[WARNING] Start pose is in collision or out of bounds; returning no path.")
            return [], stats
        h0 = self._heuristic(start, goal, h2d)
        n0 = HybridNode(key=start_key, pose=start, g=0.0, h=h0, f=h0, parent_id=-1, parent_action=(+1, 0.0))
        nodes.append(n0)
        #& UPDATE: best_g indexed by kappa bins in final dimension instead of direction
        # best_g[start_key.iy, start_key.ix, start_key.itheta, int(start_key.ikappa)] = 0.0
        self._update_best_g(start_key, 0.0)
        heapq.heappush(open_heap, (n0.f, 0))
        stats.pushed += 1
        while open_heap and stats.expanded < int(max_expansions):
            _, nid = heapq.heappop(open_heap)
            cur = nodes[nid]
            k = cur.key
            # dominance check: skip if no improvement
            if not self.map.in_bounds(k.ix, k.iy) or not self._is_better_g(k, cur.g):
                continue
            stats.expanded += 1
            # goal check for early exit
            if self._goal_reached(cur.pose, goal):
                stats.end_time_s = time.perf_counter()
                path: List[Pose] = self._reconstruct(nodes, nid)
                if path and not self._validate_path_exact(path, verbose=True):
                    print("[WARNING] Goal reached but final path failed exact collision check; returning no path.")
                    return [], stats
                if len(path) == 0:
                    print("[WARNING] Goal reached but failed to reconstruct path; returning no path.")
                return path, stats
            # TODO: move to its own function later
            # test analytic expansion every N expansions # TODO: (should probably put a minimum distance threshold here too)
            if self.cfg.analytic_every_n > 0 and (stats.expanded % int(self.cfg.analytic_every_n) == 0):
                stats.analytic_attempts += 1
                shot: Optional[List[Pose]] = self._try_goal_shot(cur.pose, goal, h2d)
                if shot is not None:
                    stats.analytic_successes += 1
                    # splice shot onto reconstructed prefix
                    prefix: List[Pose] = self._reconstruct(nodes, nid)
                    stats.end_time_s = time.perf_counter()
                    full_path = prefix + shot
                    if full_path and not self._validate_path_exact(full_path, verbose=True):
                        print("[WARNING] Analytic shot path fails exact collision validation; returning no path.")
                        return [], stats
                    if len(full_path) == 0:
                        print("[WARNING] Analytic shot succeeded but failed to reconstruct path; returning no path.")
                    return full_path, stats
            prev_dir, prev_u = cur.parent_action
            # expand children via ~~discrete steering~~ curvature-rate controls and direction
            directions = (+1, -1) if self.cfg.allow_reverse else (+1,)
            for direction in directions:
                ds = self._select_step(cur.pose)
                # for steer in self.steer_set:
                for u in self.u_set:
                    res = self.model.rollout(
                        cur.pose, u, direction, ds,
                        int(self.cfg.n_substeps), self.cfg.kappa_max, self.map, self.footprint_offsets,
                        dO_m=self._dO, theta_bins=int(self.cfg.grid.theta_bins), rho=self._rho,
                        gate_radius_m=self._gate_radius_m, exact_check_margin_m=self._exact_margin_m, footprint_cache=self._footprint_cache,
                    )
                    stats.collision_checks += int(self.cfg.n_substeps) # += 1
                    if res is None:
                        continue
                    nxt_pose, rho_int = res
                    # TODO: needs updating on the backend logic (after inclusion of curvature)
                    nxt_key = self.indexer.pose_to_key(nxt_pose) #, direction)
                    if not self.map.in_bounds(nxt_key.ix, nxt_key.iy):
                        continue
                    # cost: length + direction penalties + optional Voronoi integral # TODO: integrate rho back in
                    g2 = cur.g + self._edge_cost(ds, direction, prev_dir, float(u), float(prev_u), float(rho_int))
                    # dominance check: only keep best g per discrete key
                    if not self._is_better_g(nxt_key, g2):
                        continue
                    h2 = self._heuristic(nxt_pose, goal, h2d)
                    n2 = HybridNode(key=nxt_key, pose=nxt_pose, parent_id=nid, g=g2, h=h2, f = g2 + h2, parent_action=(direction, u))
                    nodes.append(n2)
                    nid2 = len(nodes) - 1
                    self._update_best_g(nxt_key, g2)
                    heapq.heappush(open_heap, (n2.f, nid2))
                    stats.pushed += 1
        stats.end_time_s = time.perf_counter()
        print("[INFO] Planner failed to find a path within the expansion limit; returning no path.")
        return [], stats



#!!! FIXME: update for curvature inclusions
class HybridAStarPlannerCpp(HybridAStarPlannerBase):
    """ Hybrid A* planner with C++ backend """
    def __init__(
        self,
        occ_grid: OccupancyGrid,
        config: PlannerConfig,
        *,
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
        *,
        max_expansions: int = 200_000,
    ) -> Tuple[List[Pose], PlannerStats]:
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
