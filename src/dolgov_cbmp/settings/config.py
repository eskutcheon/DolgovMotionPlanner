# src/dolgov_cbmp/settings/config.py
import math
from dataclasses import field
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass
# project imports
from dolgov_cbmp.structs import GoalSpec, Pose


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
    kappa_rate_weight: float = Field(default=0.05, ge=0.0, le=10.0)           # weight on $\int u^2 ds$ (for curvature change in the cost function)
    #? NOTE: kappa_rate_change_weight := the anti-oscillation weight - if under-weighted, paths tend to show oscillatory behavior
    kappa_rate_change_weight: float = Field(default=0.5, ge=0.0, le=10.0)     # weight on $|u - u_prev|$ (optional extra smoothing)
    score_heuristic_weight: float = Field(default=0.1, ge=0.0, le=1.0)        # weight on heuristic score for computing the terminal score with the goal tolerances


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
    #& UPDATE: moved Voronoi parameters here for better organization
    voronoi_alpha: float = Field(default=1.0, ge=0.0, le=100.0)
    voronoi_dO_max: float = Field(default=5.0, ge=0.0, le=1000.0)


@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class CurvatureParams:
    """ curvature discretization and steering-rate constraints owned by the planner
        - affects both the search space and the analytic connector when using curvature-rate control
    """
    # kappa_bins: int = Field(default=11, ge=1, le=101) # needs validation if left in
    kappa_max: float = Field(default=0.2, ge=0.0, le=2.0)   # max curvature $|\kappa|$ (1/meters)
    #? NOTE: kappa_rate_max being too low may lead to more aggressive curvature changes
    kappa_rate_max: float = Field(default=0.1, ge=0.0, le=2.0) # max curvature change per step - $|u| = |\frac{d\kappa}{ds}|$ (1/m^2)
    #? NOTE: kappa_rate_samples is the branching factor - 3: decent, 5: much slower, 7: often terrible slowdown
    kappa_rate_samples: int = Field(default=3, ge=1, le=101) # curvature-rate control samples $u = d\kappa/ds$. Typically 3: [-u_max, 0, +u_max]

    @model_validator(mode="after")
    def _validate_samples(self) -> "CurvatureParams":
        if self.kappa_rate_samples % 2 == 0:
            raise ValueError("curvature.kappa_rate_samples should be odd to include a straight control")
        return self


@dataclass(frozen=True, slots=True, config=ConfigDict(validate_assignment=True))
class PlannerConfig:
    vehicle: VehicleParams
    weights: PlannerWeights = field(default_factory=PlannerWeights)
    heuristics: HeuristicParams = field(default_factory=HeuristicParams)
    analytic: AnalyticScheduleParams = field(default_factory=AnalyticScheduleParams)
    step_policy: StepPolicyParams = field(default_factory=StepPolicyParams)
    # bounded analytic connector (beam search over curvature-rate actions)
    #! FIXME: setting to False always fails except when using beam search, nonholonomic heuristic enabled, and with high dense_g threshold
    #   TODO: make an issue for this later
    use_analytic_connector: bool = True
    connector: ConnectorParams = field(default_factory=ConnectorParams)
    use_path_smoothing: bool = True # knob for post-search path smoothing
    smoother: PathSmootherParams = field(default_factory=PathSmootherParams)
    curvature: CurvatureParams = field(default_factory=CurvatureParams)
    step_size: float = Field(default=1.0, gt=0.0, le=50.0)      # propagation distance per expansion [m] - should be a multiple of the grid resolution
    #? NOTE: n_substeps is a major performance knob - more substeps means better edge collision detection but more expensive edge checks
    n_substeps: int = Field(default=5, ge=1, le=100)         # collision sampling along edge - increase to address failing narrow paths - if 1, only check at the endpoint
    allow_reverse: bool = True
    # rectangle collision sampling in vehicle frame; if None, planners choose a default based on grid resolution
    footprint_sample_step: Optional[float] = Field(default=None, gt=0.0, le=10.0)
    #? NOTE: too low -> use sparse dict w/ slower lookup but lower memory, too high -> may allocate enormous arrays and thrash RAM (slow anyway)
    dense_best_g_max_states: int = Field(default=10_000_000, ge=100, le=100_000_000)   # large-grid support - avoids dense best_g if the full 4D lattice is too large
    # open-list tie break - when f is equal, prefer deeper nodes (larger g)
    prefer_larger_g_tiebreak: bool = True


    @model_validator(mode="after")
    def _validate_cross_parameters(self) -> "PlannerConfig":
        if self.step_policy.step_size_max < self.step_size:
            raise ValueError("step_policy.step_size_max must be >= step_size")
        return self


class PlanningRunConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = Field(default="python", pattern=r"^(python|cpp)$")
    max_expansions: int = Field(default=100_000, ge=1000, le=1_000_000)
    world_cfg_path: Optional[str] = Field(default=None, pattern=r".*\.(yaml|yml|json|jsonl|pkl|npz|hdf5)$")
    planner: PlannerConfig
    start: Pose
    goal: GoalSpec

    @field_validator("world_cfg_path")
    @classmethod
    def _validate_world_cfg_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not Path(value).expanduser().is_file():
            raise ValueError(f"world_cfg_path {value} does not exist or is not a file")
        return value