# src/dolgov_cbmp/planners/planners.py
from typing import List, Optional, Tuple, Literal, Dict, Union, Any, TypeAlias
import heapq
import time
import numpy as np
# local module imports
from dolgov_cbmp.structs import Pose, GoalSpec, PlannerStats, HybridNode, DiscreteKey, WorldModel
from dolgov_cbmp.settings import PlannerConfig
from dolgov_cbmp.models import *
from dolgov_cbmp.planners.planner_base import HybridAStarPlannerBase, PlannerEventStream, TickCallback, StatsCallback

try:
    from dolgov_cbmp.cpp_kernels import run_search_cpp, CPP_AVAILABLE
except Exception:  # pragma: no cover
    CPP_AVAILABLE = False
    run_search_cpp = None  # type: ignore


OpenHeapEntry: TypeAlias = Tuple[float, float, int]


class BestGWrapper:
    """ Best-g container that hides dense vs sparse storage
        - Both dense and sparse are keyed by the same flat id: Indexer.key_to_flat(DiscreteKey)
        - Dense mode uses an epoch+stamp scheme to reuse arrays across plan() calls without O(N) clearing
    """
    def __init__(self, indexer: 'Indexer', max_dense_states: int):
        self._indexer = indexer
        self._max_dense_states = int(max_dense_states)
        self._total_states = int(indexer.get_total_states())
        self._use_dense = self._total_states <= self._max_dense_states
        # TODO: consider using a single structured array for the dense case to improve cache locality instead of separate arrays for g values and stamps
        #   - would need to update the logic accordingly but could be a worthwhile optimization
        self._g_dense: Optional[np.ndarray] = None
        self._stamp: Optional[np.ndarray] = None
        self._epoch: np.uint32 = np.uint32(1)
        self._g_sparse: Dict[int, float] = {}
        if self._use_dense:
            self._g_dense = np.empty(self._total_states, dtype=np.float32)
            self._stamp = np.zeros(self._total_states, dtype=np.uint32)

    @property
    def use_dense(self) -> bool:
        return bool(self._use_dense)

    @property
    def total_states(self) -> int:
        return int(self._total_states)

    @property
    def max_dense_states(self) -> int:
        return int(self._max_dense_states)

    def reset(self) -> None:
        if self._use_dense:
            assert self._stamp is not None
            self._epoch = np.uint32(self._epoch + np.uint32(1))
            if self._epoch == np.uint32(0):
                self._stamp.fill(np.uint32(0))
                self._epoch = np.uint32(1)
        else:
            self._g_sparse.clear()

    def _flat(self, key: 'DiscreteKey') -> int:
        return int(self._indexer.key_to_flat(key))

    def get(self, key: 'DiscreteKey', default: float = float("inf")) -> float:
        flat = self._flat(key)
        if self._use_dense:
            assert self._g_dense is not None and self._stamp is not None
            return float(self._g_dense[flat]) if self._stamp[flat] == self._epoch else float(default)
        return float(self._g_sparse.get(flat, float(default)))

    def set(self, key: 'DiscreteKey', g: float) -> None:
        flat = self._flat(key)
        if self._use_dense:
            assert self._g_dense is not None and self._stamp is not None
            self._g_dense[flat] = np.float32(g)
            self._stamp[flat] = self._epoch
        else:
            self._g_sparse[flat] = float(g)

    def is_better(self, key: 'DiscreteKey', g: float, tol: float = 1e-8) -> bool:
        return float(g) <= (self.get(key) + float(tol))



def planner_factory(
    world_or_grid: Union[OccupancyGrid, WorldModel],
    config: PlannerConfig,
    backend: Literal["python", "cpp"] = "python",
    use_rectangle_footprint: bool = True,
    use_voronoi_edge_cost: bool = True,
) -> HybridAStarPlannerBase:
    """ Factory function to create a Hybrid A* planner with the requested backend. """
    # allow passing either OccupancyGrid or WorldModel for backwards compatibility and convenience; planner will extract occupancy grid from world if needed
    occ_grid = world_or_grid.occupancy_grid if isinstance(world_or_grid, WorldModel) else world_or_grid
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
    def _get_best_g(self) -> BestGWrapper:
        """ Reusable best-g container (dense uses epoch-stamps for O(1) resets)
            Now for any true concurrency across plan() calls, we'll probably have to instantiate separate planners or something - revisit later
        """
        #! TEMPORARY - need to debug this problem of beam search failing for certain tests (only for small dense_g thresholds) - for now just raising an error to catch it when it happens and investigate
        if self.cfg.use_analytic_connector and self.cfg.connector.mode.lower().strip() == "beam":
            total_states = self.indexer.get_total_states()
            if self.cfg.dense_best_g_max_states < total_states:
                print(f"\n[INFO] failure with total_states={total_states} and dense_best_g_max_states={self.cfg.dense_best_g_max_states}")
                #? NOTE: error is likely due to the analytic connector's reliance on best_g for its internal pruning, which can cause it to fail when best_g is too
                #   sparse and misses valid connections. Investigate potential fixes like improving robustness of the beam search or adjusting the pruning strategy
                raise RuntimeError(f"Analytic beam connector enabled but it currently fails for dictionary representation; Investigating bug now.")
        bg: Optional[BestGWrapper] = getattr(self, "_best_g", None)
        if bg is None or bg.max_dense_states != int(self.cfg.dense_best_g_max_states):
            bg = BestGWrapper(self.indexer, max_dense_states=int(self.cfg.dense_best_g_max_states))
            # TODO: guard this behind a verbose flag eventually, probably via callback events for readability/maintainability
            kind = "dense" if bg.use_dense else "sparse"
            print(f"\n[INFO] Initializing best_g: {kind} (total states: {bg.total_states}, threshold: {self.cfg.dense_best_g_max_states})")
            setattr(self, "_best_g", bg)
        else:
            bg.reset()
        return bg

    def _resolve_analytic_shot(self, shot_path, nodes: List[HybridNode], nid: int, verbose=False) -> Tuple[Optional[List[Pose]], bool]:
        # shot: Optional[List[Pose]] = self._try_goal_shot(cur.pose, goal, h2d, events=events)
        if shot_path is None:
            return None, False # return no path and success=False
        # splice shot onto reconstructed prefix
        prefix: List[Pose] = self._reconstruct(nodes, nid)
        #& UPDATE: big source of slowdown fixed: stop smoothing every analytic attempt
        raw_path = prefix + shot_path
        # validate the raw (unsmoothed) splice first; smoothing is expensive and can introduce collisions
        if raw_path and not self._validate_path_exact(raw_path, verbose=verbose):
            # shot found but rejected by validation step
            return raw_path, False
        if not self.cfg.use_path_smoothing:
            return raw_path, True
        smooth_path = self._smooth_path(raw_path)
        if smooth_path and not self._validate_path_exact(smooth_path, verbose=verbose):
            return raw_path, True
        if len(smooth_path) == 0:
            return raw_path, True
        return smooth_path, True


    def _expand_successor_nodes(
        self,
        cur: HybridNode,
        nid: int,
        goal: GoalSpec,
        h2d: HolonomicWithObstacles2D,
        best_g: BestGWrapper,
        nodes: List[HybridNode],
        open_heap: List[OpenHeapEntry],
        events: PlannerEventStream,
        stats: PlannerStats,
    ):
        prev_dir, prev_u = cur.parent_action
        # expand children via curvature-rate controls and direction
        directions = (+1, -1) if self.cfg.allow_reverse else (+1,)
        for direction in directions:
            ds = self._select_step(cur.pose)
            for u in self.u_set:
                res = self._rollout_kinematic_model(cur.pose, u, direction, ds, rho=self._rho, events=events)
                if res is None:
                    continue
                nxt_pose, rho_int = res
                nxt_key: DiscreteKey = self.indexer.pose_to_key(nxt_pose, direction)
                if not self.map.in_bounds(nxt_key.ix, nxt_key.iy):
                    stats.out_of_bounds_skips += 1
                    continue
                # cost: length + direction penalties + optional Voronoi integral
                g2 = cur.g + self._edge_cost(ds, direction, prev_dir, u, prev_u, rho_int)
                # dominance check: only keep best g per discrete key
                if not best_g.is_better(nxt_key, g2):
                    # if events is not None:
                    events.on_pruned_trajectory([cur.pose, nxt_pose])
                    continue
                h2 = self._heuristic(nxt_pose, goal, h2d)
                n2 = HybridNode(key=nxt_key, pose=nxt_pose, parent_id=nid, g=g2, h=h2, f = g2 + h2, parent_action=(direction, u))
                nodes.append(n2)
                events.on_explored_edge(cur.pose, nxt_pose)
                best_g.set(nxt_key, g2)
                tie = -n2.g if self.cfg.prefer_larger_g_tiebreak else 0.0
                heapq.heappush(open_heap, (n2.f, tie, len(nodes) - 1))
                events.on_push()


    def _attempt_analytic_shot(
        self,
        cur: HybridNode,
        goal: GoalSpec,
        h2d: HolonomicWithObstacles2D,
        nodes: List[HybridNode],
        nid: int,
        open_heap_size: int,
        trajectory: List[Pose],
        events: PlannerEventStream,
    ) -> Optional[List[Pose]]:
        shot_path: Optional[List[Pose]] = self._try_goal_shot(cur.pose, goal, h2d, events=events)
        events.on_analytic_attempt()
        shot_path, success = self._resolve_analytic_shot(shot_path, nodes, nid, verbose=False)
        # if a shot was found, either restart loop and skip this node (if validation failed) or return path
        if shot_path is not None:
            events.on_analytic_shot(shot_path)
        if not shot_path or not success:
            return None
        events.emit_terminal(cur_node=cur, cur_pose=cur.pose, trajectory=trajectory, open_size=open_heap_size)
        events.on_analytic_success()
        return shot_path


    def plan(
        self,
        start: Pose,
        goal: GoalSpec,
        max_expansions: int = 200_000,
        tick_callback: Optional[TickCallback] = None,
        tick_stride: int = 100,
        stats_callback: Optional[StatsCallback] = None,
        stats_stride: int = 1000,
        telemetry_session: Optional[Any] = None,
        close_telemetry_session: bool = True,
    ) -> Tuple[List[Pose], PlannerStats]:
        """ Hybrid A* planner as Python implementation of the search loop """
        stats, events = self._initialize_stats_and_events(tick_callback, tick_stride, stats_callback, stats_stride, telemetry_session)
        try:
            # Goal-dependent holonomic-with-obstacles heuristic
            h2d = self._build_goal_heuristics(goal)
            best_g = self._get_best_g()
            open_heap: List[Tuple[float, float, int]] = []  # open set with keys (f, tie_key, node_id)
            nodes: List[HybridNode] = []
            start_key = self.indexer.pose_to_key(start, direction=1) # assume starting with forward direction; will be updated in the node expansion logic and dominance checks
            # if not self.map.in_bounds(start_key.ix, start_key.iy) or not self._pose_is_free_fast(start):
            if not self._pose_is_free_fast(start):
                stats.end_time_s = time.perf_counter()
                events.emit_stats(force=True)
                print("[WARNING] Start pose is in collision or out of bounds; returning no path.")
                return [], stats
            h0 = self._heuristic(start, goal, h2d)
            n0 = HybridNode(key=start_key, pose=start, g=0.0, h=h0, f=h0, parent_id=-1, parent_action=(+1, 0.0))
            nodes.append(n0)
            # update best-g for the start node before pushing to open set
            best_g.set(start_key, 0.0)
            tie0 = -n0.g if bool(self.cfg.prefer_larger_g_tiebreak) else 0.0
            heapq.heappush(open_heap, (n0.f, tie0, 0))
            events.on_push()
            best_h_nid = 0
            best_h_val = n0.h
            while open_heap and stats.expanded < int(max_expansions):
                _, _, nid = heapq.heappop(open_heap)
                # process frontier node (w/ nid) - if its g value has no better path to that discrete key, skip it, else expand it and add its children to the open set
                cur = nodes[nid]
                k = cur.key
                # dominance check: skip if no improvement
                if not self.map.in_bounds(k.ix, k.iy) or not best_g.is_better(k, cur.g):
                    stats.dominated_skips += 1
                    continue
                events.on_expand(cur.pose, len(open_heap))
                # if this node has the best h so far, save it for a potential last-chance analytic connection at the end (helps in sparse/open maps)
                if cur.h < best_h_val:
                    best_h_val = float(cur.h)
                    best_h_nid = int(nid)
                cur_traj: List[Pose] = []
                # only reconstruct trajectories when it'll actually emit a tick
                if events.should_emit_tick():
                    cur_traj = self._reconstruct(nodes, nid) # to reuse in tick callback without reconstructing multiple times per expansion
                events.emit_periodic(cur_node=cur, cur_pose=cur.pose, trajectory=cur_traj, open_size=len(open_heap))
                # goal check for early exit
                if self._goal_reached(cur.pose, goal):
                    if not cur_traj:
                        cur_traj = self._reconstruct(nodes, nid)
                    events.emit_terminal(cur_node=cur, cur_pose=cur.pose, trajectory=cur_traj, open_size=len(open_heap))
                    stats.end_time_s = time.perf_counter()
                    return self._finalize_terminal_path(cur_traj), stats
                if self._should_try_analytic(cur.pose, goal.pose, stats.expanded):
                    shot_path = self._attempt_analytic_shot(cur, goal, h2d, nodes, nid, open_heap_size=len(open_heap), trajectory=cur_traj, events=events)
                    if shot_path is not None:
                        stats.end_time_s = time.perf_counter()
                        events.emit_stats(force=True)
                        return shot_path, stats
                self._expand_successor_nodes(cur, nid, goal, h2d, best_g, nodes, open_heap, events, stats)
            # last-chance bounded connector from best frontier node (helps sparse/open maps)
            if self.cfg.use_analytic_connector and 0 <= best_h_nid < len(nodes):
                conn_path = self._attempt_analytic_shot(
                    nodes[best_h_nid], goal, h2d, nodes, best_h_nid,
                    open_heap_size=len(open_heap), trajectory=self._reconstruct(nodes, best_h_nid), events=events
                )
                stats.end_time_s = time.perf_counter()
                events.emit_stats(force=True)
                if conn_path is not None:
                    return conn_path, stats
            print("\n[INFO] Planner failed to find a path within the expansion limit; returning no path.")
            stats.end_time_s = time.perf_counter()
            return [], stats
        except (KeyboardInterrupt, RuntimeError, OSError):
            stats.end_time_s = time.perf_counter()
            events.emit_stats(force=True)
            raise
        finally:
            if telemetry_session is not None and close_telemetry_session:
                telemetry_session.close()


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
        stats_callback: Optional[StatsCallback] = None,
        stats_stride: int = 1000,
        telemetry_session: Optional[Any] = None,
        close_telemetry_session: bool = True,
    ) -> Tuple[List[Pose], PlannerStats]:
        raise DeprecationWarning("C++ backend is currently disabled pending updates to C++ kernels for curvature support; use Python backend for now.")
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
