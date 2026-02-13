
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional, Tuple
import math



@dataclass(frozen=True, slots=True)
class Pose:
    """ continuous pose in world frame - located at the rear axle center by convention """
    x: float
    y: float
    theta: float  # yaw (radians)
    # $\kappa$ here is the signed curvature of the rear axle path: kappa = tan(steering_angle) / wheelbase width
    kappa: float = 0.0  # curvature (1/meters)
    # NOTE: keeping a direction bit in the state isn't a bad idea, but the take-home instructions explicitly define state as `state = (x, y, yaw, curvature`
        # $\sigma \in \{0,1\}$ can still indicate forward/reverse motion along an edge, but not as a hashed state component
        # instead infer it from the action that generated the node

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.theta, self.kappa)

    def __repr__(self) -> str:
        return f"Pose(x={self.x:.2f}, y={self.y:.2f}, theta={math.degrees(self.theta):.1f} deg, kappa={self.kappa:.3f} 1/m)"


@dataclass(frozen=True, slots=True)
class GoalSpec:
    pose: Pose
    pos_tol: float = 0.5      # meters
    theta_tol: float = math.radians(10.0)
    kappa_tol: float = 0.1    # 1/meters
    # kappa_goal: Optional[float] = None  # desired goal curvature (if any)

    def __post_init__(self) -> None:
        if float(self.pos_tol) < 0.0:
            raise ValueError("GoalSpec.pos_tol must be >= 0")
        if float(self.theta_tol) < 0.0:
            raise ValueError("GoalSpec.theta_tol must be >= 0")
        if float(self.kappa_tol) < 0.0:
            raise ValueError("GoalSpec.kappa_tol must be >= 0")
        object.__setattr__(self, "pos_tol", float(self.pos_tol))
        object.__setattr__(self, "theta_tol", float(self.theta_tol))
        object.__setattr__(self, "kappa_tol", float(self.kappa_tol))



@dataclass(frozen=True, slots=True)
class VehicleParams:
    """ vehicle physical parameters and kinematic limits - reference: https://volkswagen-specs.com/passat/2006-2010/specs/ """
    wheelbase: float = 2.7  # right around the target for the 2006 Passat from their paper (106.7 m)
    max_steer: float = math.radians(35.0)
    # Geometry (for rectangle footprint collision)
    width: float = 1.8
    front_overhang: float = 0.9
    rear_overhang: float = 1.0

    @property
    def length(self) -> float:
        return float(self.wheelbase + self.front_overhang + self.rear_overhang)


@dataclass(frozen=True, slots=True)
class GridSpec:
    """ occupancy grid discretization parameters - grid shape inferred from occupancy array shape """
    resolution: float          # meters per cell
    theta_bins: int            # number of discretized headings
    origin_xy: Tuple[float, float] = (0.0, 0.0)  # world origin of grid [m]
    #& update: adding curvature parameters for the grid spec
    kappa_bins: int = 11       # (grid curvature resolution) number of discrete curvature values
    kappa_max: float = 0.2     # max curvature (1/meters)

    def __post_init__(self) -> None:
        if float(self.resolution) <= 0.0:
            raise ValueError("GridSpec.resolution must be > 0")
        if int(self.theta_bins) <= 0:
            raise ValueError("GridSpec.theta_bins must be an integer > 0")
        if int(self.kappa_bins) <= 0:
            raise ValueError("GridSpec.kappa_bins must be an integer > 0")
        if float(self.kappa_max) < 0.0:
            raise ValueError("GridSpec.kappa_max must be >= 0")
        object.__setattr__(self, "resolution", float(self.resolution))
        object.__setattr__(self, "theta_bins", int(self.theta_bins))
        object.__setattr__(self, "kappa_bins", int(self.kappa_bins))
        object.__setattr__(self, "kappa_max", float(self.kappa_max))

    def to_dict(self):
        return asdict(self)



@dataclass(frozen=True, slots=True)
class PlannerWeights:
    reverse_penalty: float = 0.1
    switch_dir_penalty: float = 10.0
    # integrating Voronoi $\rho \in \[0,1\]$ along path edges to prefer paths away from obstacles
    voronoi_weight: float = 0.5  # placeholder weights for later extension
    # steer_change_weight: float = 0.0
    # curvature-rate penalty (encourages smooth steering evolution without instantaneous jumps)
    kappa_rate_weight: float = 0.05              # weight on $\int u^2 ds$ (for curvature change in the cost function)
    kappa_rate_change_weight: float = 0.5        # weight on $|u - u_prev|$ (optional extra smoothing)

# @dataclass(frozen=True, slots=True)
# class VoronoiParams:
#     alpha: float = 1.0
#     dO_max: float = 5.0

@dataclass(frozen=True, slots=True)
class ConnectorParams:
    """ parameters for the bounded analytic connector (beam search over curvature rate actions) """
    connector_horizon: int = 24
    connector_beam_width: int = 8
    terminal_pos_weight: float = 8.0
    terminal_theta_weight: float = 3.0
    terminal_kappa_weight: float = 2.0

    @property
    def terminal_score_weights(self) -> Tuple[float, float, float]:
        return (self.terminal_pos_weight, self.terminal_theta_weight, self.terminal_kappa_weight)


@dataclass(frozen=True, slots=True)
class AnalyticScheduleParams:
    every_n: int = 100
    max_distance: float = 15.0 # only attempt analytic connection if within this (Euclidean)
    use_adaptive_schedule: bool = True
    min_interval: int = 20
    max_interval: int = 180
    distance_power: float = 1.5


@dataclass(frozen=True, slots=True)
class StepPolicyParams:
    # using variable-resolution step (from 2010 paper) - longer arcs in wide-open space / wider Voronoi regions
    use_variable_step: bool = False #? NOTE: set False for tighter, highly discretized mazes; Set true for large open spaces + sparse obstacles
    step_size_max: float = 3.0
    variable_step_beta: float = 0.5 # $ds \approx \beta*(dO + dV)$ - if dV unavailable we approximate with $dO$
    # curvature-aware down-scaling for tight maneuvers
    curvature_slowdown_gain: float = 2.5


@dataclass(frozen=True, slots=True)
class PathSmootherParams:
    # post-search path smoothing parameters
    smoothing_passes: int = 3
    smoothing_window: int = 10
    # objective-based refinement with anchored safety retries
    use_objective_smoother: bool = True
    objective_smoothing_iters: int = 24
    objective_smoothing_lr: float = 0.08
    objective_smoothing_fd_eps: float = 0.05
    objective_smoothing_safe_distance_m: float = 1.5
    smoothing_anchor_rounds: int = 3
    objective_w_length: float = 0.05
    objective_w_smooth: float = 0.35
    objective_w_obstacle: float = 1.0
    objective_w_curvature: float = 0.5
    # align smoother objective with Voronoi field when available
    objective_w_voronoi: float = 0.25
    # extra performance controls for objective refinement
    objective_solver_maxiter: int = 30
    objective_solver_tol: float = 1e-3
    objective_max_points: int = 120


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    grid: GridSpec
    vehicle: VehicleParams
    weights: PlannerWeights = field(default_factory=PlannerWeights)
    # voronoi: VoronoiParams = field(default_factory=VoronoiParams)
    voronoi_alpha: float = 1.0
    voronoi_dO_max: float = 5.0
    step_size: float = 1.0          # propagation distance per expansion [m] - should be a multiple of the grid resolution
    n_substeps: int = 5             # collision sampling along edge
    kappa_rate_samples: int = 3     # curvature-rate control samples $u = d\kappa/ds$. Typically 3: [-u_max, 0, +u_max]
    allow_reverse: bool = True
    #& UPDATE: moved to new AnalyticScheduleParams dataclass
    # analytic_every_n: int = 100
    # analytic_max_distance: float = 15.0    # only attempt analytic connection if within this (Euclidean)
    analytic: AnalyticScheduleParams = field(default_factory=AnalyticScheduleParams)
    # table radius for non-holonomic heuristic (goal-local frame)
    nh_table_xy_radius: float = 20.0
    nh_table_xy_res: float = 1.0
    nh_table_theta_res: float = math.radians(5.0)
    # non-holonomic table tightening - optional coarser/explicit curvature bins
    nh_kappa_bins: Optional[int] = None
    # rectangle collision sampling in vehicle frame; if None, planners choose a default based on grid resolution
    footprint_sample_step: Optional[float] = None
    kappa_max: float = 0.2                  # max curvature $|\kappa|$ (1/meters)
    kappa_rate_max: float = 0.1            # max curvature change per step - $|u| = |\frac{d\kappa}{ds}|$ (1/m^2)
    #& UPDATE: moved variable step parameters to new StepPolicyParams dataclass
    step_policy: StepPolicyParams = field(default_factory=StepPolicyParams)
    # large-grid support - avoids dense best_g if the full 4D lattice is too large
    dense_best_g_max_states: int = 5_000_000
    # keep separate best-g per last-motion direction (+1/-1) while preserving the same hashed state key (x, y, theta, kappa)
    use_directional_dominance: bool = True
    # open-list tie break - when f is equal, prefer deeper nodes (larger g)
    prefer_larger_g_tiebreak: bool = True
    # bounded analytic connector (beam search over curvature-rate actions)
    use_analytic_connector: bool = True
    connector: ConnectorParams = field(default_factory=ConnectorParams)
    use_path_smoothing: bool = True # knob for post-search path smoothing
    #& UPDATE: moved path smoother parameters to new PathSmootherParams dataclass
    smoother: PathSmootherParams = field(default_factory=PathSmootherParams)


# TODO: might also want to do away with this one and keep it all in `HybridNode.key` as is currently used
    # alternativately, I can move more methods from the Indexer class to this struct and make it a bit more of a "DiscreteState" class that encapsulates the discrete key and any relevant methods for hashing, neighbor generation, etc
    # NOTE: right now, `Indexer.pose_to_key` instantiates DiscreteKey objects, but the Indexer class is still responsible for all the hashing and neighbor generation logic; could move some of that here
        # also, `Indexer.key_to_flat` accepts a `DiscreteKey` and produces the singleton key - overall there seems to be plenty of opportunity to combine the two classes
@dataclass(frozen=True, slots=True)
class DiscreteKey:
    ix: int
    iy: int
    itheta: int
    # direction: int  # +1 forward, -1 reverse
    ikappa: int  # curvature bin index

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.ix, self.iy, self.itheta, self.ikappa)


@dataclass(slots=True)
class HybridNode:
    # TODO: really need to add annotations for parameters
    """ node stored in the search; continuous pose + discrete key + costs """
    key: DiscreteKey
    pose: Pose
    g: float    # cost-to-come
    h: float    # heuristic cost-to-go
    f: float    # total estimated cost
    parent_id: int = -1
    #& Parent action is kept as an edge attribute (NOT part of the hashed DiscreteKey):
        # direction $\sigma \in \{+1,-1\}$, curvature-rate $u = \frac{d\kappa}{ds}$
    parent_action: Tuple[int, float] = (1, 0.0)  # (direction bit, curvature delta)

#
# Tradeoff (explicit): if we exclude direction from the dominance key, we may prune a state that is geometrically identical
#     but reached with a different last-motion mode. If switch penalties are large, that can change optimality. If this becomes
#     an issue, one solution is to store two best-g values per key internally (one for last sigma = +1 and for -1) without
#     it technically being part of the state key used for hashing/lookup.
# - Honestly may just want to go rogue and keep it in the state, regardless of what the instructions say, since it makes sense for switch penalties
#



# TODO: work on the MCAP visualization code to utilize the stats classes


@dataclass(slots=True)
class PlannerStats:
    expanded: int = 0
    pushed: int = 0
    collision_checks: int = 0
    analytic_attempts: int = 0
    analytic_successes: int = 0
    start_time_s: float = 0.0
    end_time_s: float = 0.0

    def elapsed_s(self) -> float:
        return self.end_time_s - self.start_time_s if self.end_time_s > self.start_time_s else 0.0


# ---------------------------------------------------------
# Planner tick logging (for benchmarking & later MCAP)
# ---------------------------------------------------------

@dataclass(slots=True)
class PlannerTick:
    expanded: int
    open_size: int
    best_f: float
    best_g: float
    pose: Pose


TickCallback = Callable[[PlannerTick], None]