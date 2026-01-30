


# main.py (add new class)
from typing import Dict, List, Optional, Tuple, Literal
import heapq
import time
import numpy as np

from structs import Pose, GoalSpec, PlannerStats, HybridNode, PlannerConfig #, PlannerTick, TickCallback
from models import (
    OccupancyGrid, Indexer, BicycleModel, VoronoiField, HolonomicWithObstacles2D, NonHolonomicWithoutObstaclesTable,
    compute_distance_to_obstacles_m, make_rectangle_footprint_offsets, pose_is_free
)
from utils import wrap_angle

try:
    from cpp_kernels import run_search_cpp, CPP_AVAILABLE
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
        self.indexer = Indexer(occ_grid)
        self.model = BicycleModel(config.vehicle)
        # Discrete steering controls (constant steer per expansion)
        self.steer_set = np.linspace(
            -config.vehicle.max_steer,
            +config.vehicle.max_steer,
            int(config.steering_samples),
            dtype=np.float64,
        )
        # Footprint offsets for collision checking.
        self.footprint_offsets: Optional[np.ndarray]
        if use_rectangle_footprint:
            step = (
                float(config.footprint_sample_step)
                if config.footprint_sample_step is not None
                else 0.5 * float(config.grid.resolution)
            )
            self.footprint_offsets = make_rectangle_footprint_offsets(config.vehicle, step)
        else:
            self.footprint_offsets = None
        # Map-dependent fields (goal-independent) can be cached safely.
        self._dO = compute_distance_to_obstacles_m(self.map.occ, float(self.map.grid.resolution))
        self._rho: Optional[np.ndarray] = None
        if use_voronoi_edge_cost and float(config.weights.voronoi_weight) > 0.0:
            self._rho = VoronoiField(self._dO, self.cfg.voronoi).rho()
        # Non-holonomic goal-local heuristic table is goal-independent and can be cached
        self._nonhol = NonHolonomicWithoutObstaclesTable(config)
        self._nonhol.build_offline()


    def _heuristic(self, pose: Pose, goal: GoalSpec, h2d: HolonomicWithObstacles2D) -> float:
        # Paper uses max(h_holonomic, h_nonholonomic)
        h_hol = h2d(pose)
        h_nh = self._nonhol(pose, goal.pose)
        return max(float(h_hol), float(h_nh))

    def _build_goal_heuristics(self, goal):
        h2d = HolonomicWithObstacles2D(self.map, cost_per_cell=self._rho)
        h2d.compute(goal.pose)
        return h2d

    def _goal_reached(self, pose: Pose, goal: GoalSpec) -> bool:
        dx = pose.x - goal.pose.x
        dy = pose.y - goal.pose.y
        if (dx * dx + dy * dy) > (goal.pos_tol * goal.pos_tol):
            return False
        dth = wrap_angle(pose.theta - goal.pose.theta)
        return abs(dth) <= goal.theta_tol

    def _edge_cost(self, direction: int, prev_direction: int) -> float:
        c = float(self.cfg.step_size)
        if direction < 0:
            c *= float(self.cfg.weights.reverse_penalty)
        if direction != prev_direction:
            c += float(self.cfg.weights.switch_dir_penalty)
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


    # def _should_try_analytic(self, pose: Pose, goal: Pose, expanded: int) -> bool:
    #     if self.cfg.analytic_every_n <= 0:
    #         return False
    #     if expanded % self.cfg.analytic_every_n != 0:
    #         return False
    #     dx = pose.x - goal.x
    #     dy = pose.y - goal.y
    #     return (dx * dx + dy * dy) <= (self.cfg.analytic_max_distance ** 2)



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
    """ Hybrid A* planner as Python implementation of the search loop """
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
        h, w = self.map.height, self.map.width
        tb = int(self.cfg.grid.theta_bins)
        best_g = np.full((h, w, tb, 2), np.inf, dtype=np.float64)
        # formerly: open set like (f, tie, node_id)
        open_heap: List[Tuple[float, int]] = []  # (f, node_id)
        nodes: List[HybridNode] = []
        start_key = self.indexer.pose_to_key(start, direction=+1)
        if not self.map.in_bounds(start_key.ix, start_key.iy) or not pose_is_free(start, self.map, self.footprint_offsets):
            stats.end_time_s = time.perf_counter()
            return [], stats
        h0 = self._heuristic(start, goal, h2d)
        n0 = HybridNode(key=start_key, pose=start, g=0.0, h=h0, f=h0, parent_id=-1)
        nodes.append(n0)
        best_g[start_key.iy, start_key.ix, start_key.itheta, int(start_key.direction < 0)] = 0.0
        heapq.heappush(open_heap, (n0.f, 0))
        stats.pushed += 1
        while open_heap and stats.expanded < int(max_expansions):
            _, nid = heapq.heappop(open_heap)
            cur = nodes[nid]
            k = cur.key
            if not self.map.in_bounds(k.ix, k.iy) or (best_g[k.iy, k.ix, k.itheta, int(k.direction < 0)] < cur.g):
                continue
            stats.expanded += 1
            # goal check for early exit
            if self._goal_reached(cur.pose, goal):
                stats.end_time_s = time.perf_counter()
                return self._reconstruct(nodes, nid), stats
            # analytic expansion hook (placeholder)
                # from paper: try Reeds-Shepp to the goal for some nodes; add if collision-free.
            # if self._should_try_analytic(cur.pose, goal.pose, stats.expanded):
            #     stats.analytic_attempts += 1
            #     # Hook placeholder to integrate a Reeds-Shepp library later, e.g.
            #         # if self._try_analytic_connection(...): stats.analytic_successes += 1; return path
            #     pass
            prev_dir = k.direction
            # expand children via discrete steering controls and direction
            directions = (+1, -1) if self.cfg.allow_reverse else (+1,)
            for direction in directions:
                for steer in self.steer_set:
                    nxt_pose = self.model.rollout_end_if_free(
                        cur.pose,
                        float(steer),
                        int(direction),
                        float(self.cfg.step_size),
                        int(self.cfg.n_substeps),
                        self.map,
                        self.footprint_offsets,
                    )
                    stats.collision_checks += 1
                    if nxt_pose is None:
                        continue
                    nxt_key = self.indexer.pose_to_key(nxt_pose, direction)
                    if not self.map.in_bounds(nxt_key.ix, nxt_key.iy):
                        continue
                    # cost: length + direction penalties + optional Voronoi integral # TODO: integrate rho back in
                    g2 = cur.g + self._edge_cost(direction, prev_dir)
                    # dominance check: only keep best g per discrete key
                    if g2 >= best_g[nxt_key.iy, nxt_key.ix, nxt_key.itheta, int(nxt_key.direction < 0)]:
                        continue
                    h2 = self._heuristic(nxt_pose, goal, h2d)
                    n2 = HybridNode(key=nxt_key, pose=nxt_pose, parent_id=nid, g=float(g2), h=float(h2), f=float(g2 + h2))
                    nodes.append(n2)
                    nid2 = len(nodes) - 1
                    best_g[nxt_key.iy, nxt_key.ix, nxt_key.itheta, int(nxt_key.direction < 0)] = g2
                    heapq.heappush(open_heap, (n2.f, nid2))
                    stats.pushed += 1
        stats.end_time_s = time.perf_counter()
        return [], stats




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
        # TODO: if this structure for benchmarking is kept, make a PlannerStatsContext to compute and hold benchmarking stats within a context manager
        stats.start_time_s = time.perf_counter()
        # Goal-dependent holonomic-with-obstacles heuristic.
        h2d = self._build_goal_heuristics(goal)
        nh_table, nh_R, nh_res, nh_dth, nh_nxy, nh_theta_bins = self._nonhol.cpp_view()
        path, s2 = run_search_cpp(
            occ_grid=self.map,
            cfg=self.cfg,
            start=start,
            goal=goal,
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
