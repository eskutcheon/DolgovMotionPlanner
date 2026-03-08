# src/dolgov_cbmp/telemetry/foxglove.py
from pathlib import Path
import math
import struct
from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple, Optional, List, TypeAlias, Dict
# local imports
from dolgov_cbmp.structs import PlannerTick, Pose, GoalSpec, WorldModel
from dolgov_cbmp.settings import VehicleParams
from dolgov_cbmp.models.models import OccupancyGrid
from dolgov_cbmp.utils import get_occupied_rectangles

try:
    import foxglove
except (ModuleNotFoundError, ImportError) as e:
    raise ImportError("foxglove-sdk is not installed. Install it with `pip install foxglove-sdk`") from e

from foxglove import Channel
from foxglove.channels import (
    SceneUpdateChannel,
    PointCloudChannel,
    GridChannel,
    FrameTransformsChannel,
)
from foxglove.schemas import (
    SceneUpdate, SceneEntity, Timestamp, Duration,
    ArrowPrimitive, SpherePrimitive, LinePrimitive, CubePrimitive, TextPrimitive, CylinderPrimitive,
    Grid, Vector2, Vector3, Quaternion,
    Pose as FGPose, Color, Point3, PointCloud, # VoxelGrid,
    PackedElementField, PackedElementFieldNumericType as NumericType,
    FrameTransform, FrameTransforms,
)


# RGBA constants (0.0-1.0 floats) for easy tweaking
RGBA_WHITE_SOFT = (1.00, 1.00, 1.00, 0.8)      # (255, 255, 255, 208)  # soft white
RGBA_WHITE_FULL = (1.00, 1.00, 1.00, 1.00)      # (255, 255, 255, 255)  # opaque white
RGBA_BLUE_EDGE = (0.55, 0.80, 1.00, 0.45)       # (140, 204, 255, 114)  # light blue, semi-transparent
RGBA_BLUE_NODE = (0.47, 0.75, 1.00, 0.67)       # (119, 191, 255, 170)  # light blue, soft opaque
RGBA_BLUE_NODE_FAINT = (0.55, 0.80, 1.00, 0.65) # (140, 204, 255, 165)  # light blue, faint
RGBA_RED_COLLISION = (1.00, 0.24, 0.24, 0.94)   # (255, 61, 61, 239)    # bright red, high alpha
RGBA_GREEN_START = (0.20, 1.00, 0.20, 1.00)     # (51, 255, 51, 255)    # bright green
RGBA_GREEN_BEST = (0.00, 1.00, 0.20, 0.80)      # (0, 255, 51, 204)     # vivid green, slightly transparent
RGBA_BLUE_CURRENT = (0.20, 0.60, 1.00, 1.00)    # (51, 153, 255, 255)   # strong blue
RGBA_TRAJECTORY_NODE = (0.95, 0.15, 0.90, 1.00) # (242, 38, 229, 255)   # hot magenta, opaque
RGBA_TRAJECTORY_LINE = (0.95, 0.15, 0.90, 0.75) # (242, 38, 229, 191)   # hot magenta, soft alpha
RGBA_PURPLE_ANALYTIC = (0.80, 0.25, 0.95, 0.90) # (204, 63, 242, 229)   # purple, soft alpha
RGBA_AMBER_GOAL = (1.00, 0.80, 0.00, 0.25)      # (255, 204, 0, 63)     # amber, translucent
RGBA_RED_GOAL = (1.00, 0.20, 0.20, 0.95)        # (255, 51, 51, 242)    # red, soft opaque
RGBA_OBSTACLE = (0.86, 0.86, 0.86, 0.82)        # (219, 219, 219, 209   # light gray, soft opaque
RGBA_GRID_OCCUPIED = (0.25, 0.25, 0.25, 0.75)   # (63, 63, 63, 191)     # dark gray, semi-transparent
RGBA_GRID_FREE = (0.00, 0.00, 0.00, 0.00)       # (0, 0, 0, 0)          # fully transparent
RGBA_PRUNED_NEAR = (0.80, 0.70, 0.18, 0.55)     # (204, 178, 45, 140)   # warm yellow, mid alpha
RGBA_PRUNED_FAR = (0.55, 0.90, 0.30, 0.22)      # (140, 229, 76, 56)    # greenish yellow, low alpha
RGBA_VEHICLE_SHADOW = (0.55, 0.55, 0.55, 0.60)  # (140, 140, 140, 153)  # medium gray, semi-transparent
RGBA_BLACK = (0.0, 0.0, 0.0, 1.0)               # (0, 0, 0, 255)        # opaque black


# Done just for brevity for all the RGB and XYZ tuples
FloatTripleType: TypeAlias = Tuple[float, float, float]
RGBAType: TypeAlias = Tuple[float, float, float, float]  # (r, g, b, a) with values in [0.0, 1.0]

# TODO: might need to deal with enum casting differences across foxglove-sdk versions (i.e. Float32 vs FLOAT32)
FIELDS_XYZ_RGBA = [
    PackedElementField(name="x", offset=0, type=NumericType.Float32),
    PackedElementField(name="y", offset=4, type=NumericType.Float32),
    PackedElementField(name="z", offset=8, type=NumericType.Float32),
    PackedElementField(name="red", offset=12, type=NumericType.Uint8),
    PackedElementField(name="green", offset=13, type=NumericType.Uint8),
    PackedElementField(name="blue", offset=14, type=NumericType.Uint8),
    PackedElementField(name="alpha", offset=15, type=NumericType.Uint8),
]

FIELDS_RGBA = [
    PackedElementField(name="red", offset=0, type=NumericType.Uint8),
    PackedElementField(name="green", offset=1, type=NumericType.Uint8),
    PackedElementField(name="blue", offset=2, type=NumericType.Uint8),
    PackedElementField(name="alpha", offset=3, type=NumericType.Uint8),
]


# TODO: might condense these into fewer specs classes that define the primitives' arguments; could be pretty versatile
#   e.g. they could have all optional rgba, diam/thickness, position, height, width, etc.
#   would make it easier to use common helper methods with the same argument names to try

#? NOTE: making smaller container dataclasses to improve manageability and extensibility; should also help with more granular validation in the future
@dataclass(frozen=True)
class HUDSpec:
    """ specifications for the HUD text element - simple text block in the top-left corner
        - could be extended with more parameters for font type, positioning, or even multiple text elements if needed
    """
    enabled: bool = True
    anchor_xyz: FloatTripleType = (0.0, 0.0, 1.6)
    offsets: FloatTripleType = (-0.5, -0.5, 2.0) # default offsets to adjust the position (after scaling them by the resolution)
    font_size: float = 12.0
    rgba: RGBAType = RGBA_WHITE_SOFT

@dataclass(frozen=True)
class ExploredSpec:
    """ specifications for explored node and edge visualization
        - currently just simple spheres and lines but could be extended to more complex primitives or even point clouds if needed (e.g. for large numbers of explored nodes)
    """
    enabled: bool = True
    edge_rgba: RGBAType = RGBA_BLUE_EDGE
    edge_thickness: float = 0.1
    node_rgba: RGBAType = RGBA_BLUE_NODE
    node_diam: float = 0.25
    collisions_rgba: RGBAType = RGBA_RED_COLLISION
    collisions_diam: float = 0.25


@dataclass(frozen=True)
class StaticElementSpec:
    """ specifications for static scene elements that don't change per tick and can be logged once at the beginning """
    start_pose_diam: float = 0.75
    start_pose_rgba: RGBAType = RGBA_GREEN_START
    goal_marker_size: FloatTripleType = (0.75, 0.75, 0.5)
    goal_tol_height: float = 0.25
    goal_tol_rgba: RGBAType = RGBA_AMBER_GOAL
    goal_marker_rgba: RGBAType = RGBA_RED_GOAL
    # Grid and obstacles
    obstacles_rgba: RGBAType = RGBA_OBSTACLE
    obstacles_height_m: float = 1.0
    # obstacles_stride_cells: int = 1
    grid_flip_y: bool = False
    grid_flip_x: bool = False

@dataclass(frozen=True)
class TrajectorySpec:
    """ specifications for the trajectory visualization - a line with pose markers """
    thickness: float = 0.25
    line_rgba: RGBAType = RGBA_TRAJECTORY_LINE
    node_rgba: RGBAType = RGBA_TRAJECTORY_NODE
    pose_marker_diam: float = 0.3

@dataclass(frozen=True)
class PrunedSpec:
    """ specifications for pruned trajectory visualization - thin, semi-transparent lines fading out with distance from vehicle """
    enabled: bool = False
    thickness: float = 0.05
    near_rgba: RGBAType = RGBA_PRUNED_NEAR
    far_rgba: RGBAType = RGBA_PRUNED_FAR

@dataclass(frozen=True)
class AnalyticShotSpec:
    """ specifications for the analytic shot visualization - a single line from current pose to the goal (if enabled) """
    enabled: bool = False
    thickness: float = 0.08
    rgba: RGBAType = RGBA_PURPLE_ANALYTIC


@dataclass(frozen=True)
class TickUpdateSpec:
    """ specifications for the SceneUpdate message construction - includes parameters for scene entities updated each tick """
    current_pose_diam: float = 0.25
    current_pose_rgba: RGBAType = RGBA_BLUE_CURRENT
    best_pose_diam: float = 0.25
    best_pose_rgba: RGBAType = RGBA_GREEN_BEST


@dataclass(frozen=True)
class VehicleSpec:
    """ specifications for vehicle visualization - for rectangular footprint, (maybe) dimensions from VehicleParams, and for the possible inclusion of heading arrows """
    height: float = 0.15
    footprint_rgba: RGBAType = RGBA_VEHICLE_SHADOW
    # for later addition of optional heading arrows anchored to vehicle shadow center
    show_heading: bool = False

@dataclass(frozen=True)
class AxisLabelSpec:
    enabled: bool = True
    rgba_map: Dict[str, RGBAType] = field(
            default_factory=lambda: {
            "x": (1.0, 0.0, 0.0, 1.0),
            "y": (0.0, 1.0, 0.0, 1.0),
            "z": (0.0, 0.0, 1.0, 1.0),
        }
    )
    # multipliers for arrow size adjustments
    shaft_diam_m: float = 0.02
    head_len_m: float = 0.1
    head_diam_m: float = 0.06
    text_shift_m: float = 1.25
    text_font_m: float = 0.1


@dataclass(frozen=True)
class VizConfig:
    # Primary frame for all visualization data
    frame_id: str = "grid"
    # Parent frame (typically Foxglove's fixed frame)
    grid_frame_id: str = "map"
    # Scene
    trajectory: TrajectorySpec = field(default_factory=TrajectorySpec)
    pruned: PrunedSpec = field(default_factory=PrunedSpec)
    analytic_shot: AnalyticShotSpec = field(default_factory=AnalyticShotSpec)
    # curvature "warms" the color (more orange) as |kappa| grows
    # kappa_ref: float = 0.25  # 1/m where warming saturates
    static: StaticElementSpec = field(default_factory=StaticElementSpec)
    scene_spec: TickUpdateSpec = field(default_factory=TickUpdateSpec)
    explored: ExploredSpec = field(default_factory=ExploredSpec)
    hud: HUDSpec = field(default_factory=HUDSpec)
    vehicle: VehicleSpec = field(default_factory=VehicleSpec)
    axis_spec: AxisLabelSpec = field(default_factory=AxisLabelSpec)



# TODO: might consider making this a context manager that opens the MCAP file on init and closes on exit, and then just pass the `log_tick` method as the callback to the planner
class FoxgloveTickSink:
    """ Streaming writer - used like `planner.tick_callback` by passing to `planner.plan`, then cleaning up with `close()` """
    def __init__(
        self,
        out_path: str | Path,
        viz: Optional[VizConfig] = None,
        occ_grid: Optional[OccupancyGrid] = None,
        start_pose: Optional[Pose] = None,
        goal: Optional[GoalSpec] = None,
        vehicle: Optional[VehicleParams] = None,
    ):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.viz = viz or VizConfig()
        # TODO: might want to keep a non-optional grid spec object in memory to handle things easier (should also add `dims` arguments to GridSpec if reusing it here)
        self.occ_grid = occ_grid
        self.start_pose = start_pose
        self.goal = goal
        self.vehicle = vehicle
        # open the MCAP file and create channels
        self._ctx = foxglove.open_mcap(self.out_path, allow_overwrite=True)
        # create channels inside the open_mcap context - Reference: https://foxglove.dev/blog/using-the-foxglove-sdk-to-generate-mcap
        self._ctx.__enter__()
        self.channels: Dict[str, Optional[Channel]] = {
            "scene": SceneUpdateChannel("/planner/scene"),
            "tick": Channel("/planner/tick", message_encoding="json"),
            "explored": PointCloudChannel("/planner/explored") if self.viz.explored.enabled else None,
            "collisions": PointCloudChannel("/planner/collisions"),
            "grid": GridChannel("/planner/grid") if self.occ_grid is not None else None,
            "transforms": FrameTransformsChannel("/planner/tf") if self.occ_grid is not None else None,
            # "obstacles": VoxelGridChannel("/planner/obstacles") if self.occ_grid is not None else None,
            # "transform": FrameTransformChannel("/planner/world_to_viz") if self.occ_grid is not None else None,
        }
        self._static_logged = False

    def __call__(self, tick: PlannerTick) -> None:
        #   since they don't change per tick and it would be wasteful to log them repeatedly
        stamp = _ts_from_time_s(float(tick.time_s))
        # static layers (grid, obstacles, start/goal)
        if not self._static_logged:
            if self.channels['grid'] is not None and self.occ_grid is not None:
                self.channels['grid'].log(
                    occupancy_to_grid_rgba(self.occ_grid, stamp, frame_id=self.viz.frame_id, flip_y=self.viz.static.grid_flip_y, flip_x=self.viz.static.grid_flip_x)
                )
                if self.channels['transforms'] is not None:
                    ox, oy = self.occ_grid.grid.origin_xy
                    res = float(self.occ_grid.grid.resolution)
                    cx = ox - (self.occ_grid.width * res / 2.0)
                    cy = oy - (self.occ_grid.height * res / 2.0)
                    self.channels['transforms'].log(
                        FrameTransforms(
                            transforms=[
                                FrameTransform(
                                    timestamp=stamp,
                                    # thinking that this maybe should be reversed
                                    parent_frame_id=self.viz.grid_frame_id,
                                    child_frame_id=self.viz.frame_id,
                                    translation=Vector3(x=float(cx), y=float(cy), z=0.0),
                                    rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                                )
                            ]
                        )
                    )
            self.channels['scene'].log(
                _static_scene(stamp=stamp, cfg=self.viz, start_pose=self.start_pose, goal=self.goal, occ_grid=self.occ_grid)
            )
            self._static_logged = True
        # "tick" JSON is useful for tables/raw inspection
        self.channels['tick'].log(tick.to_scalar_dict())
        # log the scene update with current pose, best pose, and trajectory
        res = self.occ_grid.grid.resolution if self.occ_grid is not None else 1.0
        self.channels['scene'].log(tick_to_scene_update(tick, cfg=self.viz, vehicle=self.vehicle, resolution=res))
        fid = self.viz.frame_id
        if self.channels['explored'] is not None:
            self.channels['explored'].log(poses_to_pointcloud(tick.explored_poses, stamp, frame_id=fid, rgba=self.viz.explored.node_rgba))
        self.channels['collisions'].log(poses_to_pointcloud(tick.collision_poses, stamp, frame_id=fid, rgba=self.viz.explored.collisions_rgba))

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)



# -----------------------------------------------------------------------------
# scene construction and static element helper functions
# -----------------------------------------------------------------------------

def _static_scene(stamp: Timestamp, cfg: VizConfig, start_pose: Optional[Pose], goal: Optional[GoalSpec], occ_grid: Optional[OccupancyGrid]) -> SceneUpdate:
    # static elements that don't change per tick (start/goal, maybe grid and obstacles too) can be logged once at the beginning instead of every tick
    zero_lifetime = Duration(sec=0, nsec=0)
    ents: List[SceneEntity] = []
    common_kwargs = dict(frame_id=cfg.frame_id, timestamp=stamp, lifetime=zero_lifetime) #, frame_locked=False, metadata=[])
    if occ_grid is not None:
        axis_len = max(1.0, min(occ_grid.width, occ_grid.height) * float(occ_grid.grid.resolution) * 0.025)
        ents.append(_axis_entity(stamp=stamp, frame_id=cfg.frame_id, axis_spec=cfg.axis_spec, axis_len=axis_len)) #! changed from grid_frame_id
    if start_pose is not None:
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/start_pose",
                spheres=[_sphere(start_pose, radius=0.5 * cfg.static.start_pose_diam, rgba=cfg.static.start_pose_rgba)],
            )
        )
    if goal is not None:
        # "diamond" goal: rotate cube 45 degrees in yaw (looks rhomboid-ish top-down)
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/goal_pose",
                cubes=[
                    _cube(
                        pos=(goal.pose.x, goal.pose.y, goal.pose.theta + math.pi / 4.0),
                        size=cfg.static.goal_marker_size,
                        rgba=cfg.static.goal_marker_rgba,
                        z=cfg.static.goal_tol_height + 0.01,  # slightly above the goal tolerance cylinder to avoid z-fighting
                    )
                ],
            )
        )
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/goal_tolerance",
                cylinders=[_cylinder(goal, rgba=cfg.static.goal_tol_rgba, height=cfg.static.goal_tol_height)]
            )
        )
    if occ_grid is not None:
        ents.append(SceneEntity(**common_kwargs, id="planner/obstacles", cubes=_obstacle_cubes(occ_grid, cfg)))
    return SceneUpdate(deletions=[], entities=ents)



# TODO: create a scene generator function that takes in a PlannerTick and outputs a SceneUpdate with all the relevant entities for that tick, and then just call that from the `__call__` method of the FoxgloveTickSink
    # - should be cleaner and more modular than having all the scene construction logic directly in the `__call__` method
    # - also should be able to use class methods or helper functions to break down the scene construction into smaller pieces
    #   - (e.g. one method for trajectory visualization, one for explored nodes/edges, etc.) to make it more manageable and extensible
    #   - this would break future parallelism, but honestly a simple producer-consumer pattern with a queue between the planner and the scene
    #       generator should be enough for decoupling and would allow scene generation to keep up even if it takes a bit longer than the
    #       planner's tick rate, without blocking the planner or dropping ticks

def tick_to_scene_update(
    tick: PlannerTick,
    cfg: VizConfig,
    vehicle: Optional[VehicleParams] = None,
    resolution: float = 1.0,
) -> SceneUpdate:
    stamp = _ts_from_time_s(float(tick.time_s))
    zero_lifetime = Duration(sec=0, nsec=0)
    common_kwargs = dict(frame_id=cfg.frame_id, timestamp=stamp, lifetime=zero_lifetime, frame_locked=False, metadata=[])
    # vehicle current pose marker (sphere)
    #!!! FIXME: I think the majority of these ids may be invalid - or at least I can't find them when opening the MCAP file in Foxglove Studio
    ents: List[SceneEntity] = [
        SceneEntity(
            **common_kwargs, id="planner/current_pose",
            spheres=[_sphere(tick.pose, radius=0.5 * cfg.scene_spec.current_pose_diam, rgba=cfg.scene_spec.current_pose_rgba)]
        ),
        # TODO: consider dropping best_pose entirely - should be apparent when it's taken as the next "current pose" anyway
        SceneEntity(
            **common_kwargs, id="planner/best_pose",
            spheres=[_sphere(tick.best_pose, radius=0.5 * cfg.scene_spec.best_pose_diam, rgba=cfg.scene_spec.best_pose_rgba)]
        ),
        SceneEntity(
            **common_kwargs, id="planner/trajectory",
            # lines = _trajectory_segments(tick.trajectory, cfg=cfg),
            lines=[_line_strip(tick.trajectory, rgba=cfg.trajectory.line_rgba, thickness=cfg.trajectory.thickness)],
            spheres=_trajectory_pose_markers(tick.trajectory, cfg),
        )
    ]
    if cfg.pruned.enabled and tick.pruned_trajectories:
        r, rgba = 0.5 * cfg.explored.node_diam, cfg.explored.node_rgba
        ents.extend([
            SceneEntity(
                **common_kwargs, id="planner/pruned_trajectories",
                lines=_pruned_tendrils(tick.pruned_trajectories, cfg=cfg),
            ),
            SceneEntity(
                **common_kwargs, id="planner/pruned_terminals",
                spheres=[_sphere(path[-1], radius=r, rgba=rgba) for path in tick.pruned_trajectories if path],
            )
        ])
        if cfg.explored.enabled:
            ents.extend([
                SceneEntity(
                    **common_kwargs, id="planner/explored_edges",
                    lines=_explored_edge_lines(tick.explored_edges, cfg),
                    spheres=[_sphere(p, radius=0.5 * cfg.explored.node_diam, rgba=cfg.explored.node_rgba) for p in tick.explored_poses]
                ),
            # TODO: (maybe) couple collision nodes with the explored nodes as well - add to the same `spheres` argument
            SceneEntity(
                **common_kwargs, id="planner/collision_nodes",
                spheres=[_sphere(p, radius=0.5 * cfg.explored.collisions_diam, rgba=cfg.explored.collisions_rgba) for p in tick.collision_poses],
            )
        ])
    # type: List[SceneEntity]  # current pose, best pose, trajectory, pruned trajectories, pruned terminals, explored edges, explored nodes, collision nodes
    if cfg.analytic_shot.enabled and tick.analytic_shot:
        analytic_line = _line_strip(
            tick.analytic_shot,
            rgba=cfg.analytic_shot.rgba,
            thickness=cfg.analytic_shot.thickness,
        )
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/analytic_shot",
                lines=[] if analytic_line is None else [analytic_line],
            )
        )
    # add vehicle footprint at z=0 if vehicle params are available
    if vehicle is not None:
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/vehicle",
                cubes=[_vehicle_footprint_cube(tick.pose, vehicle, cfg)],
            )
        )
    if cfg.hud.enabled:
        # lightweight HUD near current pose
        hud = f"it={tick.iteration}  open={tick.open_size}\nf={tick.best_f:.2f}  g={tick.best_g:.2f}"
        offsets = tuple(t * resolution for t in cfg.hud.offsets)
        anchors = [a + offset for a, offset in zip(cfg.hud.anchor_xyz, offsets)]
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/hud",
                texts=[
                    TextPrimitive(
                        pose=_to_fg_pose_xyyaw(anchors[0], anchors[1], yaw=0.0, z=anchors[2]),
                        billboard=True,
                        font_size=cfg.hud.font_size,
                        scale_invariant=True,
                        color=Color(**_rgba_dict(cfg.hud.rgba)),
                        text=hud,
                    )
                ],
            )
        )
    return SceneUpdate(deletions=[], entities=ents)


# TODO: might be worth making this a static method of `FoxgloveTickSink` and just have it open/close the MCAP file internally - need to pass ticks and output path
def write_ticks_mcap_foxglove(
    ticks: Iterable[PlannerTick],
    out_path: str | Path,
    # frame_id: str = "map",
    viz: VizConfig = VizConfig(),
    occ_grid: Optional[OccupancyGrid] = None,
    start_pose: Optional[Pose] = None,
    goal: Optional[GoalSpec] = None,
    vehicle: Optional[VehicleParams] = None,
    world: Optional[WorldModel] = None
) -> None:
    #& UPDATE: now supporting importing world model directly - in the future this should be a primary positional argument without a default
    if world is not None:
        occ_grid = world.occupancy_grid
        start_pose = world.start
        goal = world.goal
    sink = FoxgloveTickSink(out_path, viz=viz, occ_grid=occ_grid, start_pose=start_pose, goal=goal, vehicle=vehicle)
    # iterate through ticks and log to channels
    try:
        for t in ticks:
            sink(t)
    finally:
        sink.close()



# -----------------------------------------------------------------------------
# time and geometry resolution helper functions
# -----------------------------------------------------------------------------

def _ts_from_time_s(t_s: float) -> Timestamp:
    sec = int(t_s)
    nsec = int(round((t_s - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec = 0
    return Timestamp(sec=sec, nsec=nsec)

def _yaw_to_quaternion(yaw: float) -> Quaternion:
    # rotation about +Z
    return Quaternion(x=0, y=0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))

def _to_fg_pose_xyyaw(x: float, y: float, yaw: float, z: float = 0.0) -> FGPose:
    pos_vector = Vector3(x=float(x), y=float(y), z=float(z))
    return FGPose(position=pos_vector, orientation=_yaw_to_quaternion(float(yaw)))

def to_fg_pose(p: Pose, z: float = 0.0) -> FGPose:
    return _to_fg_pose_xyyaw(p.x, p.y, p.theta, z=z)

def _mix_rgb(a: FloatTripleType, b: FloatTripleType, t: float) -> FloatTripleType:
    """ linearly interpolate between two RGB colors a and b with parameter t in [0, 1] """
    t = max(0.0, min(1.0, float(t)))
    return tuple((ai + t * (bi - ai) for ai, bi in zip(a, b)))

def _rgba_dict(rgba: RGBAType) -> Dict[str, float]:
    """ helper method to easily pass **kwargs to foxglove.schemas.Color from RGBAType tuples """
    return {k: v for k, v in zip(["r", "g", "b", "a"], rgba)}

# -----------------------------------------------------------------------------
# visualization primitive helper functions - mostly for brevity
# -----------------------------------------------------------------------------

def _sphere(p: Pose, radius: float, rgba: RGBAType = RGBA_BLUE_NODE, z: float = 0.0) -> SpherePrimitive:
    return SpherePrimitive(
        pose = _to_fg_pose_xyyaw(p.x, p.y, yaw=0.0, z=z),
        size=Vector3(x = 2 * radius, y = 2 * radius, z = 2 * radius),
        color=Color(**_rgba_dict(rgba)),
    )

def _cube(pos: FloatTripleType, size: FloatTripleType, rgba: RGBAType, z: float = 0.0) -> CubePrimitive:
    x, y, yaw = pos
    return CubePrimitive(
        pose = _to_fg_pose_xyyaw(x, y, yaw=yaw, z=z),
        size=Vector3(x=size[0], y=size[1], z=size[2]),
        color=Color(**_rgba_dict(rgba)) ,
    )

def _cylinder(goal: GoalSpec, rgba: RGBAType = RGBA_AMBER_GOAL, height: float = 0.3) -> CylinderPrimitive:
    # r, g, b = rgba[0], rgba[1], rgba[2]
    return CylinderPrimitive(
        pose = _to_fg_pose_xyyaw(goal.pose.x, goal.pose.y, yaw=0.0, z=height),
        size=Vector3(x = goal.pos_tol, y = goal.pos_tol, z = height),
        bottom_scale=1.0,
        top_scale=1.0,
        color=Color(**_rgba_dict(rgba)),
    )

def _line_strip(points: Sequence[Pose], rgba: RGBAType, thickness: float, z: float = 0.0) -> Optional[LinePrimitive]:
    if len(points) < 2:
        return None
    return LinePrimitive(
        thickness=float(thickness),
        scale_invariant=False,
        points=[Point3(x=p.x, y=p.y, z=z) for p in points],
        color=Color(**_rgba_dict(rgba)),
    )

def _arrow(pose: FGPose, shaft_l: float, shaft_d: float, head_l: float, head_d: float, rgba: RGBAType = RGBA_BLACK) -> ArrowPrimitive:
    return ArrowPrimitive(
        pose=pose,
        shaft_length=shaft_l,
        shaft_diameter = shaft_d,
        head_length = head_l,
        head_diameter = head_d,
        color=Color(**_rgba_dict(rgba)),
    )

def _axis_entity(stamp: Timestamp, frame_id: str, axis_spec: AxisLabelSpec, axis_len: float = 1.0) -> SceneEntity:
    arrow_kwargs = {
        "shaft_l": axis_len,
        "shaft_d": axis_spec.shaft_diam_m * axis_len,
        "head_l": axis_spec.head_len_m * axis_len,
        "head_d": axis_spec.head_diam_m * axis_len
    }
    label_kwargs = {
        "billboard": True,
        "font_size": axis_spec.text_font_m * axis_len,
        "scale_invariant": False,
    }
    arrows = [
        _arrow(pose=_to_fg_pose_xyyaw(0.0, 0.0, yaw=0.0, z=0.0), rgba=axis_spec.rgba_map["x"], **arrow_kwargs),
        _arrow(pose=_to_fg_pose_xyyaw(0.0, 0.0, yaw=math.pi / 2.0, z=0.0), rgba=axis_spec.rgba_map["y"], **arrow_kwargs),
        _arrow(
            pose=FGPose(
                position=Vector3(x=0.0, y=0.0, z=0.0),
                orientation=Quaternion(x=0.0, y=math.sin(-math.pi / 4.0), z=0.0, w=math.cos(-math.pi / 4.0)),
            ),
            rgba=axis_spec.rgba_map["z"],
            **arrow_kwargs,
        )
    ]
    labels = []
    for axis in ("x", "y", "z"):
        pose_kwargs = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        pose_kwargs[axis] = axis_len * axis_spec.text_shift_m
        labels.append(
            TextPrimitive(
                pose=_to_fg_pose_xyyaw(**pose_kwargs),
                **label_kwargs,
                color=Color(**_rgba_dict(axis_spec.rgba_map.get(axis, RGBA_BLACK))),
                text=axis,
            )
        )
    return SceneEntity(
        frame_id=frame_id,
        timestamp=stamp,
        lifetime=Duration(sec=0, nsec=0),
        id="planner/axes",
        arrows=arrows,
        texts=labels,
    )


# -----------------------------------------------------------------------------
# scene construction and static element helper functions
# -----------------------------------------------------------------------------

def _trajectory_pose_markers(traj: Sequence[Pose], cfg: VizConfig) -> List[SpherePrimitive]:
    if cfg.trajectory.pose_marker_diam <= 0.0:
        return []
    radius = 0.5 * cfg.trajectory.pose_marker_diam
    return [_sphere(p, radius=radius, rgba=cfg.trajectory.node_rgba, z=0.03) for p in traj]



def _explored_edge_lines(edges: Sequence[Tuple[Pose, Pose]], cfg: VizConfig) -> List[LinePrimitive]:
    lines: List[LinePrimitive] = []
    for p0, p1 in edges:
        line = _line_strip([p0, p1], rgba=cfg.explored.edge_rgba, thickness=cfg.explored.edge_thickness)
        if line is not None:
            lines.append(line)
    return lines


def _pruned_tendrils(pruned_trajectories: Sequence[Sequence[Pose]], cfg: VizConfig) -> List[LinePrimitive]:
    lines: List[LinePrimitive] = []
    n = max(1, len(pruned_trajectories))
    for i, path in enumerate(pruned_trajectories):
        if len(path) < 2:
            continue
        p0 = path[0]
        p1 = path[-1]
        t = i / max(1, n - 1)
        rgb = _mix_rgb(cfg.pruned.near_rgba[:3], cfg.pruned.far_rgba[:3], t)
        alpha = cfg.pruned.near_rgba[3] + t * (cfg.pruned.far_rgba[3] - cfg.pruned.near_rgba[3])
        line = _line_strip([p0, p1], rgba=(*rgb, alpha), thickness=cfg.pruned.thickness)
        if line is not None:
            lines.append(line)
    return lines


def _vehicle_footprint_cube(pose: Pose, vehicle: VehicleParams, cfg: VizConfig) -> CubePrimitive:
    length = float(vehicle.wheelbase + vehicle.front_overhang + vehicle.rear_overhang)
    return _cube(
        pos = [
            pose.x + 0.5 * length * math.cos(pose.theta),
            pose.y + 0.5 * length * math.sin(pose.theta),
            pose.theta
        ],
        size=(length, vehicle.width, 0.1),
        rgba=cfg.vehicle.footprint_rgba,
        z=cfg.vehicle.height / 2.0,
    )
    # TODO: consider adding the arrows for the current heading as discussed in other TODOs


# -----------------------------------------------------------------------------
# point cloud generation helper functions
# -----------------------------------------------------------------------------

def _points_to_pointcloud(points_xyz: Sequence[FloatTripleType], stamp: Timestamp, frame_id: str, rgba: RGBAType) -> PointCloud:
    r, g, b, a = map(lambda x: int(255 * x) & 0xFF, rgba)
    point_stride = 16
    buf = bytearray(point_stride * len(points_xyz))
    fmt = "<fffBBBB"  # little-endian: 3 float32 + 4 uint8
    for i, (x, y, z) in enumerate(points_xyz):
        struct.pack_into(fmt, buf, i * point_stride, float(x), float(y), float(z), r, g, b, a)
    return PointCloud(
        timestamp=stamp,
        frame_id=frame_id,
        pose=None,
        point_stride=point_stride,
        fields=FIELDS_XYZ_RGBA,
        data=bytes(buf),
    )

def poses_to_pointcloud(poses: Sequence[Pose], stamp: Timestamp, frame_id: str, rgba: RGBAType, z: float = 0.0) -> PointCloud:
    return _points_to_pointcloud([(p.x, p.y, float(z)) for p in poses], stamp, frame_id=frame_id, rgba=rgba)


# -----------------------------------------------------------------------------
# occupancy grid and pose generation helper functions
# -----------------------------------------------------------------------------

def occupancy_to_grid_rgba(occ: OccupancyGrid, stamp: Timestamp, frame_id: str = "map", flip_y: bool = False, flip_x: bool = False) -> Grid:
    """ minimal occupancy visualization (occupied is dark & opaque; free is transparent) """
    # expects that `occ.occ` is 2D bool-like [rows, cols], occ.grid.resolution, occ.grid.origin_xy
    res = float(occ.grid.resolution)
    xo, yo = occ.grid.origin_xy
    rows, cols = occ.occ.shape
    obs_rgba = tuple(int(255 * x) for x in RGBA_GRID_OCCUPIED)
    buf = bytearray(4 * cols * rows)
    for y in range(rows):
        yy = (rows - 1 - y) if flip_y else y
        for x in range(cols):
            xx = (cols - 1 - x) if flip_x else x
            i = 4 * (yy * cols + xx)
            buf[i:i+4] = bytes(obs_rgba) if occ.occ[y, x] else bytes((0, 0, 0, 0))
    return Grid(
        timestamp=stamp,
        frame_id=frame_id,
        pose=_to_fg_pose_xyyaw(xo, yo, yaw=0.0, z=0.0),
        column_count=cols,
        cell_size=Vector2(x=res, y=res),
        row_stride=int(4 * cols),
        cell_stride=4,
        fields=FIELDS_RGBA,
        data=bytes(buf),
    )


def _obstacle_cubes(occ_grid: OccupancyGrid, cfg: VizConfig) -> List[CubePrimitive]:
    """ create cube primitives for occupied cells in the occupancy grid, merging adjacent occupied cells into larger rectangular prisms """
    flip_x, flip_y = cfg.static.grid_flip_x, cfg.static.grid_flip_y
    cubes: List[CubePrimitive] = []
    res = float(occ_grid.grid.resolution)
    ox, oy = occ_grid.grid.origin_xy
    height = float(cfg.static.obstacles_height_m)
    for ix0, iy0, ix1, iy1 in get_occupied_rectangles(occ_grid.occ):
        width = (ix1 - ix0) * res
        depth = (iy1 - iy0) * res
        # calculate the center position of the cube based on the rectangle corners and resolution, then apply flipping if needed
        x_offset = occ_grid.width - ix0 - ix1 if flip_x else ix0 + ix1
        y_offset = occ_grid.height - iy0 - iy1 if flip_y else iy0 + iy1
        cx = ox + x_offset * 0.5 * res
        cy = oy + y_offset * 0.5 * res
        cubes.append(_cube((cx, cy, 0.0), (width, depth, height), rgba=cfg.static.obstacles_rgba, z=height / 2.0))
    return cubes

