# src/dolgov_cbmp/structs.py

from dataclasses import asdict, field
from typing import Optional, Tuple, List, Any
import math
from pydantic import ConfigDict, Field
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
    # kappa_max: float = Field(default=0.2, ge=0.0, le=2.0)   # max curvature (1/meters)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class WorldModel:
    """ runtime world state passed around planner entry points - also meant to be constructed by data loaded from `world_cfg_path` later """
    occupancy_grid: Any
    start: Pose
    goal: GoalSpec
    vehicle: Optional[Any] = None

    @property
    def grid(self) -> GridSpec:
        return self.occupancy_grid.grid

    def validate(self) -> None:
        from .settings import VehicleParams
        from .models import OccupancyGrid
        assert isinstance(self.occupancy_grid, OccupancyGrid), "occupancy_grid must be an instance of OccupancyGrid"
        if self.vehicle is not None:
            assert isinstance(self.vehicle, VehicleParams), "vehicle must be an instance of VehicleParams if provided"
        if not hasattr(self.occupancy_grid, "world_to_grid"):
            raise AttributeError("occupancy_grid must have a world_to_grid method for validating start/goal poses")
        if not hasattr(self.occupancy_grid, "_sanity_check_poses"):
            raise AttributeError("occupancy_grid must have a _sanity_check_poses method for validating start/goal poses")
        s_ix, s_iy = self.occupancy_grid.world_to_grid(self.start.x, self.start.y)
        g_ix, g_iy = self.occupancy_grid.world_to_grid(self.goal.pose.x, self.goal.pose.y)
        self.occupancy_grid._sanity_check_poses(s_ix, s_iy, g_ix, g_iy, "maps to out-of-bounds or occupied cell")


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


# ---------------------------------------------------------
# Planner tick logging (for benchmarking & later MCAP)
# ---------------------------------------------------------

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

    def to_dict(self) -> dict:
        return asdict(self)

    def clone(self) -> "PlannerStats":
        return PlannerStats(**asdict(self))


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
