# src/dolgov_cbmp/planners/planners.py
from typing import List, Optional, Tuple, Literal, Dict
import heapq
import time
import numpy as np
# local module imports
from dolgov_cbmp.structs import Pose, GoalSpec, PlannerStats, HybridNode, PlannerConfig, DiscreteKey
from dolgov_cbmp.models import *
from dolgov_cbmp.planners.planner_base import HybridAStarPlannerBase, PlannerEventStream, TickCallback

try:
    from dolgov_cbmp.cpp_kernels import run_search_cpp, CPP_AVAILABLE
except Exception:  # pragma: no cover
    CPP_AVAILABLE = False
    run_search_cpp = None  # type: ignore



class BestGWrapper:
    """ Best-g container that hides dense vs sparse storage
        - Both dense and sparse are keyed by the same flat id: Indexer.key_to_flat(DiscreteKey)
        - Dense mode uses an epoch+stamp scheme to reuse arrays across plan() calls without O(N) clearing
    """
    def __init__(self, indexer: 'Indexer', max_dense_states: int):
        self._indexer = indexer
        self._max_dense_states = int(max_dense_states)
        #& UPDATE: direction is now part of DiscreteKey, so we account for that in the total state count and dense array shape
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
    def _get_best_g(self) -> BestGWrapper:
        """ Reusable best-g container (dense uses epoch-stamps for O(1) resets)
            Now for any true concurrency across plan() calls, we'll probably have to instantiate separate planners or something - revisit later
        """
        #! TEMPORARY - need to debug this problem of beam search failing for certain tests (only for small dense_g thresholds) - for now just raising an error to catch it when it happens and investigate
        if self.cfg.use_analytic_connector and self.cfg.connector.mode.lower().strip() == "beam":
            total_states = self.indexer.get_total_states()
            if self.cfg.dense_best_g_max_states < total_states:
                print(f"[INFO] failure with total_states={total_states} and dense_best_g_max_states={self.cfg.dense_best_g_max_states}")
                #? NOTE: error is likely due to the analytic connector's reliance on best_g for its internal pruning, which can cause it to fail when best_g is too
                #   sparse and misses valid connections. Investigate potential fixes like improving robustness of the beam search or adjusting the pruning strategy
                raise RuntimeError(f"Analytic beam connector enabled but it currently fails for dictionary representation; Investigating bug now.")
        bg: Optional[BestGWrapper] = getattr(self, "_best_g", None)
        if bg is None or bg.max_dense_states != int(self.cfg.dense_best_g_max_states):
            bg = BestGWrapper(self.indexer, max_dense_states=int(self.cfg.dense_best_g_max_states))
            # TODO: guard this behind a verbose flag eventually, probably via callback events for readability/maintainability
            kind = "dense" if bg.use_dense else "sparse"
            print(f"[INFO] Initializing best_g: {kind} (total states: {bg.total_states}, threshold: {self.cfg.dense_best_g_max_states})")
            setattr(self, "_best_g", bg)
        else:
            bg.reset()
        return bg

    def _attempt_analytic_connection(
        self, cur: HybridNode, goal: GoalSpec, h2d: HolonomicWithObstacles2D, nodes: List[HybridNode], nid: int,
        verbose=False, events: Optional[PlannerEventStream] = None,
    ) -> Tuple[Optional[List[Pose]], bool]:
        shot: Optional[List[Pose]] = self._try_goal_shot(cur.pose, goal, h2d, events=events)
        if shot is None:
            return None, False # return no path and success=False
        # splice shot onto reconstructed prefix
        prefix: List[Pose] = self._reconstruct(nodes, nid)
        #& UPDATE: big source of slowdown fixed: stop smoothing every analytic attempt
        raw_path = prefix + shot
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
        # use_dense_best_g, best_g = self._init_best_g()
        best_g = self._get_best_g()
        open_heap: List[Tuple[float, float, int]] = []  # open set with keys (f, tie_key, node_id)
        nodes: List[HybridNode] = []
        start_key = self.indexer.pose_to_key(start, direction=1) # assume starting with forward direction; will be updated in the node expansion logic and dominance checks
        # if not self.map.in_bounds(start_key.ix, start_key.iy) or not self._pose_is_free_fast(start):
        if not self._pose_is_free_fast(start):
            stats.end_time_s = time.perf_counter()
            print("\n[WARNING] Start pose is in collision or out of bounds; returning no path.")
            return [], stats
        h0 = self._heuristic(start, goal, h2d)
        n0 = HybridNode(key=start_key, pose=start, g=0.0, h=h0, f=h0, parent_id=-1, parent_action=(+1, 0.0))
        nodes.append(n0)
        # update best-g for the start node before pushing to open set
        # self._update_best_g(best_g, use_dense_best_g, start_key, 0.0) #, +1)
        best_g.set(start_key, 0.0)
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
            if not self.map.in_bounds(k.ix, k.iy) or not best_g.is_better(k, cur.g):
            #not self._is_better_g(best_g, use_dense_best_g, k, cur.g): #, cur.parent_action[0]):
                # print("CHECKPOINT: skipping node due to dominance check or out-of-bounds key")
                stats.dominated_skips += 1
                continue
            events.on_expand(cur.pose, len(open_heap))
            # if this node has the best h so far, save it for a potential last-chance analytic connection at the end (helps in sparse/open maps)
            if cur.h < best_h_val:
                best_h_val = float(cur.h)
                best_h_nid = int(nid)
            # cur_traj = self._reconstruct(nodes, nid) # to reuse in tick callback without reconstructing multiple times per expansion
            #& UPDATE: stop reconstructing trajectories unless emitting ticks
            cur_traj: List[Pose] = []
            # only reconstruct trajectories when it'll actually emit a tick
            if events.callback is not None and ((stats.expanded % events.stride) == 0):
                cur_traj = self._reconstruct(nodes, nid) # to reuse in tick callback without reconstructing multiple times per expansion
            events.emit_tick(force=False, cur_node=cur, cur_pose=cur.pose, trajectory=cur_traj, open_size=len(open_heap))
            # TODO: extract snippet below to a new function - should be easiest to centralize these last minute checks (also think I'm duplicating some logic elsewhere)
            # goal check for early exit
            if self._goal_reached(cur.pose, goal):
                if not cur_traj:
                    cur_traj = self._reconstruct(nodes, nid)
                events.emit_tick(force=True, cur_node=cur, cur_pose=cur.pose, trajectory=cur_traj, open_size=len(open_heap))
                stats.end_time_s = time.perf_counter()
                # path: List[Pose] = self._smooth_path(cur_traj)
                #& UPDATE: no longer smoothing before validating the path to avoid throwing away valid solutions
                path: List[Pose] = self._smooth_path(cur_traj) if self.cfg.use_path_smoothing else cur_traj
                # if smoothing produces an invalid path (should be rare w/ collision-aware smoothing), fall back to the raw reconstructed path instead of failing
                if path and not self._validate_path_exact(path, verbose=False):
                    if self.cfg.use_path_smoothing and cur_traj and self._validate_path_exact(cur_traj, verbose=False):
                        print("\n[WARNING] Smoothed path failed exact collision check but unsmoothed path is valid; returning unsmoothed path.")
                        return cur_traj, stats
                    print("\n[WARNING] Goal reached but final path failed exact collision check; returning no path.")
                    return [], stats
                if len(path) == 0:
                    print("\n[WARNING] Goal reached but failed to reconstruct path; returning no path.")
                return path, stats
            if self._should_try_analytic(cur.pose, goal.pose, stats.expanded):
                stats.analytic_attempts += 1
                shot_path, success = self._attempt_analytic_connection(cur, goal, h2d, nodes, nid, events=events)
                # if a shot was found, either restart loop and skip this node (if validation failed) or return path
                if shot_path is not None:
                    events.on_analytic_shot(shot_path)
                    if success:
                        stats.analytic_successes += 1
                        events.emit_tick(force=True, cur_node=cur, cur_pose=cur.pose, trajectory=cur_traj, open_size=len(open_heap))
                        stats.end_time_s = time.perf_counter()
                        return shot_path, stats
                    #& UPDATE: switched logic so that shot rejected -> keep expanding this node (no longer starves the search)
            prev_dir, prev_u = cur.parent_action
            # TODO: extract loop and needed arguments to a new function for computing the next pose given a curvature-rate action
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
                    nxt_key: DiscreteKey = self.indexer.pose_to_key(nxt_pose, direction)
                    if not self.map.in_bounds(nxt_key.ix, nxt_key.iy):
                        stats.out_of_bounds_skips += 1
                        continue
                    # cost: length + direction penalties + optional Voronoi integral
                    g2 = cur.g + self._edge_cost(ds, direction, prev_dir, u, prev_u, rho_int)
                    # dominance check: only keep best g per discrete key
                    # if not self._is_better_g(best_g, use_dense_best_g, nxt_key, g2): #, direction):
                    if not best_g.is_better(nxt_key, g2): #, direction):
                        if events is not None:
                            events.on_pruned_trajectory([cur.pose, nxt_pose])
                        continue
                    h2 = self._heuristic(nxt_pose, goal, h2d)
                    n2 = HybridNode(key=nxt_key, pose=nxt_pose, parent_id=nid, g=g2, h=h2, f = g2 + h2, parent_action=(direction, u))
                    nodes.append(n2)
                    events.on_explored_edge(cur.pose, nxt_pose)
                    # self._update_best_g(best_g, use_dense_best_g, nxt_key, g2) #, direction)
                    best_g.set(nxt_key, g2)
                    tie = -n2.g if bool(self.cfg.prefer_larger_g_tiebreak) else 0.0
                    heapq.heappush(open_heap, (n2.f, tie, len(nodes) - 1))
                    events.on_push()
        # last-chance bounded connector from best frontier node (helps sparse/open maps)
        if self.cfg.use_analytic_connector and 0 <= best_h_nid < len(nodes):
            shot_path, success = self._attempt_analytic_connection(nodes[best_h_nid], goal, h2d, nodes, best_h_nid, verbose=False, events=events)
            if shot_path is not None:
                events.on_analytic_shot(shot_path)
            if shot_path and success:
                events.emit_tick(
                    force=True,
                    cur_node=nodes[best_h_nid],
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
