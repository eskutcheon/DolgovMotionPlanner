# src/structs.py
from dataclasses import asdict, field #, dataclass
from typing import Optional, Tuple, List
import math
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pose:
    """ continuous pose in world frame - located at the rear axle center by convention """
    x: float
    y: float
    theta: float  # yaw (radians)
    # $\kappa$ here is the signed curvature of the rear axle path: kappa = tan(steering_angle) / wheelbase width
    kappa: float = 0.0  # curvature (1/meters)
    # NOTE: the take-home instructions explicitly define state as `state = (x, y, yaw, curvature)`, but this is a better representation

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.theta, self.kappa)

    def __repr__(self) -> str:
        return f"Pose(x={self.x:.2f}, y={self.y:.2f}, theta={math.degrees(self.theta):.1f} deg, kappa={self.kappa:.3f} 1/m)"


@dataclass(frozen=True, slots=True)
class GoalSpec:
    pose: Pose
    pos_tol: float = Field(default=0.5, ge=0.0, le=10.0)      # meters
    theta_tol: float = Field(default=math.radians(15.0), ge=0.0, le=math.pi)
    #& UPDATE: removed use of kappa tolerance everywhere since it complicates goal checking and isn't found in similar literature
    # kappa_tol: float = 0.1    # 1/meters


@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class GridSpec:
    """ occupancy grid discretization parameters - grid shape inferred from occupancy array shape """
    #? NOTE: resolution performance tradeoff: too small -> explodes runtime, too large -> jagged paths and failure in tight spaces
    resolution: float = Field(default=1.0, gt=0.0, le=5.0)  # meters per cell
    #? NOTE: theta_bins performance tradeoff: too low -> snapping behavior and failure in tight spaces,  too high -> explodes runtime and memory (from hashed state keys)
    theta_bins: int = Field(default=36, ge=4, le=720)       # number of discretized headings
    origin_xy: Tuple[float, float] = (0.0, 0.0)             # world origin of grid [m]
    #? NOTE: kappa_bins performance tradeoff: too low -> worse curvature handling, too high -> explodes runtime and memory
    kappa_bins: int = Field(default=11, ge=1, le=101)       # (grid curvature resolution) number of discrete curvature values
    kappa_max: float = Field(default=0.2, ge=0.0, le=2.0)   # max curvature (1/meters)

    def to_dict(self):
        return asdict(self)


# Could move more methods from the Indexer class to this struct and make it more of a "DiscreteState" class that
#   encapsulates the discrete key and any relevant methods for hashing, neighbor generation, etc
# NOTE: right now, `Indexer.pose_to_key` instantiates DiscreteKey objects, but the Indexer class is still responsible for all the hashing and neighbor generation logic; could move some of that here
    # also, `Indexer.key_to_flat` accepts a `DiscreteKey` and produces the singleton key - overall there seems to be plenty of opportunity to combine the two classes
@dataclass(frozen=True, slots=True)
class DiscreteKey:
    ix: int
    iy: int
    itheta: int
    ikappa: int  # curvature bin index
    #& UPDATE: added direction to the hashed state key to properly handle switch penalties without needing a dominance proxy
    #   increases the size of the state space but it's necessary for correctness when switch penalties are large
    direction: int # +1 forward, -1 reverse (last motion mode)

    def as_tuple(self) -> Tuple[int, ...]:
        return (self.ix, self.iy, self.itheta, self.ikappa, self.direction)


@dataclass(slots=True)
class HybridNode:
    """ node stored in the search; continuous pose + discrete key + costs """
    key: DiscreteKey
    pose: Pose
    g: float    # cost-to-come
    h: float    # heuristic cost-to-go
    f: float    # total estimated cost
    parent_id: int = -1
    # parent action is kept as an edge attribute (NOT part of the hashed DiscreteKey):
        # direction $\sigma \in \{+1,-1\}$, curvature-rate $u = \frac{d\kappa}{ds}$
    parent_action: Tuple[int, float] = (1, 0.0)  # (direction bit, curvature delta)


@dataclass(slots=True)
class PlannerStats:
    expanded: int = 0
    pushed: int = 0
    collision_checks: int = 0
    failed_rollouts: int = 0
    dominated_skips: int = 0
    out_of_bounds_skips: int = 0
    goal_checks: int = 0
    analytic_attempts: int = 0
    analytic_successes: int = 0
    ticks_emitted: int = 0
    max_open_size: int = 0
    start_time_s: float = 0.0
    end_time_s: float = 0.0

    def elapsed_s(self) -> float:
        return self.end_time_s - self.start_time_s if self.end_time_s > self.start_time_s else 0.0


# ---------------------------------------------------------
# Planner tick logging (for benchmarking & later MCAP)
# ---------------------------------------------------------

@dataclass(slots=True)
class PlannerTick:
    iteration: int
    time_s: float
    expanded: int
    pushed: int
    open_size: int
    collision_checks: int
    failed_rollouts: int
    best_f: float
    best_g: float
    pose: Pose
    best_pose: Pose
    trajectory: List[Pose]
    explored_poses: List[Pose]
    collision_poses: List[Pose]
    explored_edges: List[Tuple[Pose, Pose]] = field(default_factory=list)
    pruned_trajectories: List[List[Pose]] = field(default_factory=list)
    analytic_shot: List[Pose] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_scalar_dict(self) -> dict:
        """ return just the scalar fields for stats/debug view (exclude poses/trajectory) """
        # not sure if this is slower, but it should work:
        return {k: v for k, v in asdict(self).items() if isinstance(v, (int, float))}


# TODO: still need to annotate these with the proper pydantic types
@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class VehicleParams:
    """ vehicle physical parameters and kinematic limits
        References for 2006 VW Passat (the make and model of the modified vehicle "Junior" used in the original paper):
            - https://volkswagen-specs.com/passat/2006-2010/specs/
            - https://www.thecarconnection.com/specifications/volkswagen_passat_2006
            - https://robots.stanford.edu/papers/junior08.pdf
    """
    wheelbase: float = Field(default=2.7, gt=0.2, le=8.0)  # right around the target for the 2006 Passat from their paper (106.7 inches)
    max_steer: float = Field(default=math.radians(35.0), gt=0.01, le=math.radians(60.0))
    # Geometry (for rectangle footprint collision)
    width: float = Field(default=1.8, gt=0.5, le=4.0)
    front_overhang: float = Field(default=0.9, ge=0.0, le=3.0)
    rear_overhang: float = Field(default=1.0, ge=0.0, le=3.0)
    # height: float = 1.5           # unused for 2D planning but could be relevant for future 3D extensions
    # min_clearance: float = 0.15   # unused for 2D planning - minimum clearance for 3D obstacles (for collision checking safety margin)

    @property
    def length(self) -> float:
        return float(self.wheelbase + self.front_overhang + self.rear_overhang)

    # TODO: might consider making this some method of the kinematic model that accepts a steering angle and returns the turning radius,
    #   since the effective turning radius changes with steering angle; this is just a heuristic minimum for now
    @property
    def min_turning_radius(self) -> float:
        return self.wheelbase / math.tan(self.max_steer)


@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class PlannerWeights:
    # multiplicative penalty on reverse motion (sigma = -1) to encourage mostly forward motion when possible without completely forbidding reverse
    #? NOTE: if `reverse_penalty` set too high, it may cause failure in tight maps that require reversing out of dead-ends
    reverse_penalty: float = Field(default=1.1, ge=0.0, le=10.0)
    switch_dir_penalty: float = Field(default=10.0, ge=0.0, le=100.0)
    # integrating Voronoi $\rho \in \[0,1\]$ along path edges to prefer paths away from obstacles
    voronoi_weight: float = Field(default=0.5, ge=0.0, le=10.0)  # keeping it at 1.0 for initial regression testing (to keep same weight as before)
    # curvature-rate penalty (encourages smooth steering evolution without instantaneous jumps)
    #? NOTE: kappa_rate_weight performance tradeoff: too low -> steering jerks, too high -> refusal to make any real steering changes
    kappa_rate_weight: float = Field(default=0.05, ge=0.0, le=10.0)              # weight on $\int u^2 ds$ (for curvature change in the cost function)
    #? NOTE: kappa_rate_change_weight := the anti-oscillation weight - if under-weighted, paths tend to show oscillatory behavior
    kappa_rate_change_weight: float = Field(default=0.5, ge=0.0, le=10.0)        # weight on $|u - u_prev|$ (optional extra smoothing)

# @dataclass(frozen=True, slots=True)
# class VoronoiParams:
#     alpha: float = 1.0
#     dO_max: float = 5.0

@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class ConnectorParams:
    """ parameters for the bounded analytic connector
        modes:
            "beam" (legacy) - bounded beam search over curvature rate-actions
            "rs" (optimal)  - Reeds-Shepp shot (preferred when reverse is allowed)
        rs_step - sampling step (in meters) used when discretizing the RS curve for collision checking
    """
    # TODO: look into modes for "dubins" (forward-only) and "astar" (fallback obstacle-free A* in SE(2) with curvature-bounded primitives)
    mode: str = Field(default="rs", pattern=r"^(rs|beam)$")
    rs_step: float = Field(default=0.5, gt=0.0, le=10.0)
    # beam-search settings (used when mode == "beam")
    connector_horizon: int = Field(default=24, ge=1, le=2_000)
    connector_beam_width: int = Field(default=8, ge=1, le=1_000)
    # terminal scoring weights (used by beam search)
    terminal_pos_weight: float = Field(default=8.0, ge=0.0, le=1_000.0)
    terminal_theta_weight: float = Field(default=3.0, ge=0.0, le=1_000.0)
    terminal_kappa_weight: float = Field(default=2.0, ge=0.0, le=1_000.0)

    def terminal_score_weights(self) -> Tuple[float, float, float]:
        return (self.terminal_pos_weight, self.terminal_theta_weight, self.terminal_kappa_weight)


@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class AnalyticScheduleParams:
    every_n: int = Field(default=100, ge=1, le=100_000)
    max_distance: float = Field(default=10.0, gt=0.0, le=1_000.0) # only attempt connection if within this Euclidean distance of the goal
    use_adaptive_schedule: bool = False
    min_interval: int = Field(default=20, ge=1, le=100_000)
    max_interval: int = Field(default=180, ge=1, le=100_000)
    distance_power: float = Field(default=1.5, ge=0.1, le=10.0)

    @model_validator(mode="after")
    def _validate_intervals(self) -> "AnalyticScheduleParams":
        if self.min_interval > self.max_interval:
            raise ValueError("analytic.min_interval must be <= analytic.max_interval")
        return self

@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class StepPolicyParams:
    # using variable-resolution step (from 2010 paper) - longer arcs in wide-open space / wider Voronoi regions
    #? NOTE: set use_variable_step=False for tighter, highly discretized mazes; Set true for large open spaces + sparse obstacles
    use_variable_step: bool = False     #? NOTE: seemingly improves behavior for escaping degenerate paths
    step_size_max: float = Field(default=3.0, gt=0.0, le=50.0)
    variable_step_beta: float = Field(default=0.5, ge=0.0, le=10.0) # $ds \approx \beta*(dO + dV)$ - if dV unavailable we approximate with $dO$
    # curvature-aware down-scaling for tight maneuvers
    curvature_slowdown_gain: float = Field(default=2.5, ge=0.0, le=50.0)


@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class PathSmootherParams:
    # post-search path smoothing parameters
    smoothing_passes: int = Field(default=3, ge=0, le=100)
    smoothing_window: int = Field(default=10, ge=1, le=1000)
    smoothing_alpha: float = Field(default=0.15, ge=0.0, le=1.0)
    # objective-based refinement with anchored safety retries
    use_objective_smoother: bool = True
    objective_smoothing_iters: int = Field(default=6, ge=0, le=1000)
    objective_smoothing_lr: float = Field(default=0.08, gt=0.0, le=10.0)
    objective_smoothing_fd_eps: float = Field(default=0.05, gt=0.0, le=10.0)
    objective_smoothing_safe_distance_m: float = Field(default=1.5, ge=0.0, le=100.0)
    smoothing_anchor_rounds: int = Field(default=3, ge=0, le=1000)
    objective_w_length: float = Field(default=0.05, ge=0.0, le=100.0)
    objective_w_smooth: float = Field(default=0.35, ge=0.0, le=100.0)
    objective_w_obstacle: float = Field(default=1.0, ge=0.0, le=100.0)
    objective_w_curvature: float = Field(default=0.5, ge=0.0, le=100.0)
    # curvature-continuity proxy (penalizes changes in curvature along the path)
    objective_w_curvature_rate: float = Field(default=0.25, ge=0.0, le=100.0)
    # align smoother objective with Voronoi field when available
    objective_w_voronoi: float = 0.25
    # extra performance controls for objective refinement
    objective_solver_maxiter: int = Field(default=30, ge=1, le=10_000)
    objective_solver_tol: float = Field(default=1e-3, gt=0.0, le=1.0)
    objective_max_points: int = Field(default=120, ge=3, le=10_000)


@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class HeuristicParams:
    #? NOTE: performance tradeoff: True -> h2d more informed but slower, False -> h2d faster but less informed (more expansions)
    use_voronoi: bool = True # whether to use Voronoi distance in the h2d heuristic to trade off guidance for speed
    use_nonholonomic: bool = True
    nonholonomic_weight: float = Field(default=0.5, ge=0.0, le=10.0)
    euclidean_fallback_distance: float = Field(default=15.0, ge=0.0, le=1000.0) # simple Euclidean threshold heuristic when far from the goal
    #? NOTE: higher radius and resolution helps guidance but raises precompute costs
    # table radius for non-holonomic heuristic (goal-local frame)
    nh_table_xy_radius: float = Field(default=20.0, gt=0.0, le=1000.0)
    nh_table_xy_res: float = Field(default=1.0, gt=0.0, le=100.0)
    nh_table_theta_res: float = Field(default=math.radians(5.0), gt=0.0, le=math.radians(30.0))
    # non-holonomic table tightening - optional coarser/explicit curvature bins
    nh_kappa_bins: Optional[int] = Field(default=3, ge=1, le=72)
    # Holonomic-with-obstacles heuristic tuning (2D DP / Dijkstra)
    h2d_min_clearance_m: float = Field(default=0.0, ge=0.0, le=100.0)    # HARD prune threshold based on dO (meters) - 0.0 avoids motion degeneracy in tight maps
    h2d_soft_clearance_m: float = Field(default=0.0, ge=0.0, le=100.0)   # OPTIONAL: soft bias away from obstacles in the 2D DP (no pruning)
    h2d_soft_clearance_weight: float = Field(default=0.0, ge=0.0, le=100.0)


@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class PlannerConfig:
    grid: GridSpec # TODO: rename to grid_spec to be more transparent - lots of places to update this though, so leaving as-is for now
    vehicle: VehicleParams
    weights: PlannerWeights = field(default_factory=PlannerWeights)
    heuristics: HeuristicParams = field(default_factory=HeuristicParams)
    # voronoi: VoronoiParams = field(default_factory=VoronoiParams)
    voronoi_alpha: float = Field(default=1.0, ge=0.0, le=100.0)
    voronoi_dO_max: float = Field(default=5.0, ge=0.0, le=1000.0)
    step_size: float = Field(default=1.0, gt=0.0, le=50.0)      # propagation distance per expansion [m] - should be a multiple of the grid resolution
    #? NOTE: n_substeps is a major performance knob - more substeps means better edge collision detection but more expensive edge checks
    n_substeps: int = Field(default=5, ge=1, le=100)         # collision sampling along edge - increase to address failing narrow paths - if 1, only check at the endpoint
    #? NOTE: kappa_rate_samples is the branching factor - 3: decent, 5: much slower, 7: often terrible slowdown
    kappa_rate_samples: int = Field(default=3, ge=1, le=101) # curvature-rate control samples $u = d\kappa/ds$. Typically 3: [-u_max, 0, +u_max]
    allow_reverse: bool = True
    analytic: AnalyticScheduleParams = field(default_factory=AnalyticScheduleParams)
    # rectangle collision sampling in vehicle frame; if None, planners choose a default based on grid resolution
    footprint_sample_step: Optional[float] = Field(default=None, gt=0.0, le=10.0)
    kappa_max: float = Field(default=0.2, ge=0.0, le=2.0)                  # max curvature $|\kappa|$ (1/meters)
    #? NOTE: kappa_rate_max being too low may lead to more aggressive curvature changes
    kappa_rate_max: float = Field(default=0.1, ge=0.0, le=2.0)            # max curvature change per step - $|u| = |\frac{d\kappa}{ds}|$ (1/m^2)
    step_policy: StepPolicyParams = field(default_factory=StepPolicyParams)
    #? NOTE: too low -> use sparse dict w/ slower lookup but lower memory, too high -> may allocate enormous arrays and thrash RAM (slow anyway)
    dense_best_g_max_states: int = Field(default=10_000_000, ge=100, le=100_000_000)   # large-grid support - avoids dense best_g if the full 4D lattice is too large
    # open-list tie break - when f is equal, prefer deeper nodes (larger g)
    prefer_larger_g_tiebreak: bool = True
    # bounded analytic connector (beam search over curvature-rate actions)
    #! FIXME: setting to False always fails except when using beam search, nonholonomic heuristic enabled, and with high dense_g threshold
    use_analytic_connector: bool = True
    connector: ConnectorParams = field(default_factory=ConnectorParams)
    use_path_smoothing: bool = True # knob for post-search path smoothing
    smoother: PathSmootherParams = field(default_factory=PathSmootherParams)


    @model_validator(mode="after")
    def _validate_cross_parameters(self) -> "PlannerConfig":
        #! TEMPORARY - enforce kappa_max consistency - eventually want to use a single source of truth, but I'll be refactoring a bunch for pydantic later anyway
        if self.kappa_max != self.grid.kappa_max:
            object.__setattr__(self, "kappa_max", self.grid.kappa_max)
        if self.step_policy.step_size_max < self.step_size:
            raise ValueError("step_policy.step_size_max must be >= step_size")
        if self.kappa_rate_samples % 2 == 0:
            raise ValueError("kappa_rate_samples should be odd to include a straight control")
        return self