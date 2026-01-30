
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import math


@dataclass(frozen=True, slots=True)
class Pose:
    """ continuous pose in world frame - located at the rear axle center by convention """
    x: float
    y: float
    theta: float  # yaw (radians)


@dataclass(frozen=True, slots=True)
class GoalSpec:
    pose: Pose
    pos_tol: float = 0.5      # meters
    theta_tol: float = math.radians(10.0)

# TODO: recheck these defaults (auto-filled by IDE Copilot) later and see what I can find about the vehicles in the original paper
@dataclass(frozen=True, slots=True)
class VehicleParams:
    """ vehicle physical parameters and kinematic limits """
    wheelbase: float = 2.7
    max_steer: float = math.radians(35.0)
    # Geometry (for rectangle footprint collision)
    width: float = 1.9
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


@dataclass(frozen=True, slots=True)
class PlannerWeights:
    reverse_penalty: float = 2.0
    switch_dir_penalty: float = 20.0
    # integrating Voronoi $\rho \in \[0,1\]$ along path edges to prefer paths away from obstacles
    voronoi_weight: float = 0.5  # placeholder weights for later extension
    # steer_change_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class VoronoiParams:
    alpha: float = 1.0
    dO_max: float = 5.0


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    grid: GridSpec
    vehicle: VehicleParams
    weights: PlannerWeights = PlannerWeights()
    voronoi: VoronoiParams = VoronoiParams()
    step_size: float = 0.5                 # propagation distance per expansion [m]
    n_substeps: int = 5                    # collision sampling along edge
    steering_samples: int = 9              # number of discrete steering controls
    allow_reverse: bool = True
    # [PLACEHOLDER] analytic expansion hook: try every N expansions; larger N => less frequent
    analytic_every_n: int = 20
    analytic_max_distance: float = 15.0    # only attempt analytic connection if within this (Euclidean)
    # table radius for non-holonomic heuristic (goal-local frame)
    nonholonomic_table_xy_radius: float = 20.0
    nonholonomic_table_xy_res: float = 1.0
    nonholonomic_table_theta_res: float = math.radians(5.0)
    # rectangle collision sampling in vehicle frame; if None, planners choose a default based on grid resolution
    footprint_sample_step: Optional[float] = None


#! MIGHT DELETE
@dataclass(frozen=True, slots=True)
class DiscreteKey:
    ix: int
    iy: int
    itheta: int
    direction: int  # +1 forward, -1 reverse


@dataclass(slots=True)
class HybridNode:
    """ node stored in the search; continuous pose + discrete key + costs """
    key: DiscreteKey
    pose: Pose
    g: float
    h: float
    f: float
    parent_id: int = -1
    # parent_action: Tuple[int, float] = (1, 0.0)  # (direction, steer_angle)


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