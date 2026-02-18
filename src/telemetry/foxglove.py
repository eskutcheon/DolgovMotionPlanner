# src/telemetry/foxglove.py
from pathlib import Path
import math
import struct
from dataclasses import dataclass, field #, asdict
from typing import Iterable, Sequence, Tuple, Optional, List, TypeAlias
# local imports
from src.structs import PlannerTick, Pose, GoalSpec, VehicleParams
from src.models.models import OccupancyGrid

try:
    import foxglove
except (ModuleNotFoundError, ImportError) as e:
    raise ImportError("foxglove-sdk is not installed. Install it with `pip install foxglove-sdk`") from e

from foxglove import Channel
from foxglove.channels import SceneUpdateChannel, PointCloudChannel, GridChannel, FrameTransformChannel
from foxglove.schemas import (
    SceneUpdate, SceneEntity, Timestamp, Duration, FrameTransform,
    SpherePrimitive, LinePrimitive, CubePrimitive, TextPrimitive, CylinderPrimitive,
    Grid, Vector2, Vector3, Quaternion,
    Pose as FGPose, Color, Point3,
    PointCloud, PackedElementField, PackedElementFieldNumericType as NumericType,
)

# Done just for brevity for all the RGB and XYZ tuples
FloatTripleType: TypeAlias = Tuple[float, float, float]
RGBAType: TypeAlias = Tuple[int, int, int, int]

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




#? NOTE: making smaller container dataclasses since it feels more manageable, extensible, and should help with more granular validation in the future
@dataclass(frozen=True)
class HUDSpec:
    enabled: bool = True
    anchor_xyz: FloatTripleType = (0.0, 0.0, 1.6)
    font_size: float = 12.0
    alpha: float = 0.55

@dataclass(frozen=True)
class ExploredSpec:
    edge_rgb: FloatTripleType = (0.55, 0.80, 1.0) # formerly `explored_line_rgb`
    edge_alpha: float = 0.45        # formerly `explored_line_alpha`
    edge_thickness: float = 0.03    # formerly `explored_line_thickness`
    node_rgba: RGBAType = (120, 190, 255, 170) # formerly `explored_rgba_u8`
    node_diam: float = 0.1 # formerly `explored_terminal_diam` in the config


@dataclass(frozen=True)
class StaticElementSpec:
    start_pose_diam: float = 0.36
    goal_marker_size: FloatTripleType = (0.50, 0.50, 0.35)
    goal_tol_height: float = 0.25
    # Grid and obstacles
    obstacles_rgba_u8: RGBAType = (220, 220, 220, 210)
    obstacles_height_m: float = 1.0
    # obstacles_stride_cells: int = 1
    max_obstacle_points: int = 50_000 #! UNUSED
    grid_flip_y: bool = True  # matches OccupancyGrid.world_to_grid / grid_to_world

@dataclass(frozen=True)
class TrajectorySpec:
    thickness: float = 0.08
    alpha: float = 1.00
    rgb: FloatTripleType = (0.95, 0.15, 0.90)
    pose_marker_diam: float = 0.10

@dataclass(frozen=True)
class PrunedSpec:
    thickness: float = 0.03
    alpha_grad: Tuple[float, float] = (0.55, 0.22)  # (near, far) alpha values for pruned trajectories based on distance from vehicle
    near_rgb: FloatTripleType = (0.80, 0.70, 0.18)
    far_rgb: FloatTripleType = (0.55, 0.90, 0.30)

@dataclass(frozen=True)
class AnalyticShotSpec:
    enabled: bool = False
    thickness: float = 0.08
    rgb: FloatTripleType = (0.80, 0.25, 0.95)
    alpha: float = 0.9

# TODO: determine unneeded parameters
@dataclass(frozen=True)
class VizConfig:
    frame_id: str = "map"
    # Scene
    trajectory: TrajectorySpec = field(default_factory=TrajectorySpec)
    # pruned path visualization - thin, semi-transparent lines fading out with distance from vehicle
    pruned: PrunedSpec = field(default_factory=PrunedSpec)
    # analytic shot visualization - single line from current pose to goal (if enabled and available), useful for visualizing the effect of the analytic shot in the planner
    analytic_shot: AnalyticShotSpec = field(default_factory=AnalyticShotSpec)
    # curvature "warms" the color (more orange) as |kappa| grows
    # kappa_ref: float = 0.25  # 1/m where warming saturates
    # Markers
    static: StaticElementSpec = field(default_factory=StaticElementSpec)
    current_pose_diam: float = 0.25
    best_pose_diam: float = 0.14
    # Vehicle cube footprint
    vehicle_height: float = 0.35
    # Point clouds
    explored: ExploredSpec = field(default_factory=ExploredSpec)
    collisions_rgba_u8: RGBAType = (255, 60, 60, 240)
    # HUD
    hud: HUDSpec = field(default_factory=HUDSpec)

    # TODO: [MAYBE] add static method to initialize some variables with an existing PlannerConfig
    # @staticmethod
    # def from_planner_config(cfg: PlannerConfig) -> "VizConfig":


# TODO: might consider making this a context manager that opens the MCAP file on init and closes on exit, and then just pass the `log_tick` method as the callback to the planner
class FoxgloveTickSink:
    """ Streaming writer - used like `planner.tick_callback` by passing to `planner.plan`, then cleaning up with `close` """
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
        self.occ_grid = occ_grid
        self.start_pose = start_pose
        self.goal = goal
        self.vehicle = vehicle
        # open the MCAP file and create channels
        self._ctx = foxglove.open_mcap(self.out_path, allow_overwrite=True)
        self._ctx.__enter__()
        # create channels inside the open_mcap context - Reference: https://foxglove.dev/blog/using-the-foxglove-sdk-to-generate-mcap
        self.scene_ch = SceneUpdateChannel("/planner/scene")
        self.tick_ch = Channel("/planner/tick", message_encoding="json")
        # TODO: change to VoxelGridChannel if we want to do better occupancy grid visualization with large maps
        self.explored_ch = PointCloudChannel("/planner/explored")
        self.collisions_ch = PointCloudChannel("/planner/collisions")
        # optional static channels for occupancy grid and obstacles
        self.grid_ch: Optional[GridChannel] = None
        # transform channel for flipping all y-values and shifting the origin of the world frame
        #! FIXME: giving errors in foxglove studio but it's showing up in the topics correctly
        # self.transform_ch = FrameTransformChannel("planner/world_to_viz")
        # self.obstacles_ch: Optional[PointCloudChannel] = None
        # static layers (grid, obstacles, start/goal)
        # t0 = _ts_from_time_s(0.0)
        self._static_logged = False
        if self.occ_grid is not None:
            self.grid_ch = GridChannel("/planner/grid")

    def __call__(self, tick: PlannerTick) -> None:
        #   since they don't change per tick and it would be wasteful to log them repeatedly
        stamp = _ts_from_time_s(float(tick.time_s))
        if not self._static_logged:
            #! FIXME: gives errors in the "Frame" and "Transforms" menus in the 3D panel in Foxglove Studio
            # self.transform_ch.log(
            #     FrameTransform(
            #         timestamp = stamp,
            #         parent_frame_id="map",
            #         child_frame_id=self.viz.frame_id,
            #         translation=Vector3(x=0.0, y=0.0, z=0.0),
            #         rotation=Quaternion(x=0.0, y=math.pi, z=0.0, w=1.0),
            #     )
            # )
            if self.grid_ch is not None and self.occ_grid is not None:
                self.grid_ch.log(
                    occupancy_to_grid_rgba(self.occ_grid, stamp, frame_id=self.viz.frame_id, flip_y=self.viz.static.grid_flip_y)
                )
            self.scene_ch.log(
                _static_scene(stamp=stamp, cfg=self.viz, start_pose=self.start_pose, goal=self.goal, occ_grid=self.occ_grid)
            )
            self._static_logged = True
        # "tick" JSON is useful for tables/raw inspection
        self.tick_ch.log(tick.to_scalar_dict())
        # log the scene update with current pose, best pose, and trajectory
        self.scene_ch.log(tick_to_scene_update(tick, cfg=self.viz, vehicle=self.vehicle)) #, show_text=True))
        fid = self.viz.frame_id
        ex_rgba, co_rgba = self.viz.explored.node_rgba, self.viz.collisions_rgba_u8
        self.explored_ch.log(poses_to_pointcloud(tick.explored_poses, stamp, frame_id=fid, rgba_u8=ex_rgba))
        self.collisions_ch.log(poses_to_pointcloud(tick.collision_poses, stamp, frame_id=fid, rgba_u8=co_rgba))

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)


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


# -----------------------------------------------------------------------------
# scene construction and static element helper functions
# -----------------------------------------------------------------------------

def _sphere(p: Pose, radius: float, rgb: FloatTripleType, alpha: float = 1.0, z: float = 0.0) -> SpherePrimitive:
    r, g, b = rgb
    return SpherePrimitive(
        pose = _to_fg_pose_xyyaw(p.x, p.y, yaw=0.0, z=z),
        size=Vector3(x = 2 * radius, y = 2 * radius, z = 2 * radius),
        color=Color(r=r, g=g, b=b, a=alpha),
    )


def _cube(pos: FloatTripleType, size: FloatTripleType, rgb: FloatTripleType, alpha: float = 1.0, z: float = 0.0) -> CubePrimitive:
    r, g, b = rgb
    x, y, yaw = pos
    return CubePrimitive(
        pose=_to_fg_pose_xyyaw(x, y, yaw=yaw, z=z),
        size=Vector3(x=size[0], y=size[1], z=size[2]),
        color=Color(r=r, g=g, b=b, a=alpha),
    )

def _goal_tolerance(goal: GoalSpec, height: float = 0.3, alpha: float = 0.18) -> CylinderPrimitive:
    return CylinderPrimitive(
        pose=FGPose(
            position=Vector3(x=goal.pose.x, y=goal.pose.y, z = height / 2.0),
            orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
        size=Vector3(x = 2 * goal.pos_tol, y = 2 * goal.pos_tol, z = height),
        bottom_scale=1.0,
        top_scale=1.0,
        color=Color(r=1.0, g=0.8, b=0.0, a=alpha),
    )


def _line_strip(points: Sequence[Pose], rgb: FloatTripleType, thickness: float, alpha: float, z: float = 0.0) -> Optional[LinePrimitive]:
    if len(points) < 2:
        return None
    return LinePrimitive(
        thickness=float(thickness),
        scale_invariant=False,
        points=[Point3(x=float(p.x), y=float(p.y), z=float(z)) for p in points],
        color=Color(r=rgb[0], g=rgb[1], b=rgb[2], a=float(alpha)),
    )


def _trajectory_pose_markers(traj: Sequence[Pose], cfg: VizConfig) -> List[SpherePrimitive]:
    if cfg.trajectory.pose_marker_diam <= 0.0:
        return []
    radius = 0.5 * cfg.trajectory.pose_marker_diam
    return [_sphere(p, radius=radius, rgb=cfg.trajectory.rgb, alpha=0.9, z=0.03) for p in traj]


def _explored_edge_lines(edges: Sequence[Tuple[Pose, Pose]], cfg: VizConfig) -> List[LinePrimitive]:
    lines: List[LinePrimitive] = []
    for p0, p1 in edges:
        line = _line_strip([p0, p1], rgb=cfg.explored.edge_rgb, thickness=cfg.explored.edge_thickness, alpha=cfg.explored.edge_alpha)
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
        rgb = _mix_rgb(cfg.pruned.near_rgb, cfg.pruned.far_rgb, t)
        alpha = cfg.pruned.alpha_grad[0] + t * (cfg.pruned.alpha_grad[1] - cfg.pruned.alpha_grad[0])
        line = _line_strip([p0, p1], rgb=rgb, thickness=cfg.pruned.thickness, alpha=alpha)
        if line is not None:
            lines.append(line)
    return lines


def _vehicle_footprint_cube(pose: Pose, vehicle: VehicleParams) -> CubePrimitive:
    length = float(vehicle.wheelbase + vehicle.front_overhang + vehicle.rear_overhang)
    return _cube(
        pos = [
            pose.x + 0.5 * length * math.cos(pose.theta),
            pose.y + 0.5 * length * math.sin(pose.theta),
            pose.theta
        ],
        size=(length, vehicle.width, 0.1),
        rgb=(0.55, 0.55, 0.55),
        alpha=0.6,
        z=0.03,
    )




# -----------------------------------------------------------------------------
# point cloud generation helper functions
# -----------------------------------------------------------------------------

def points_to_pointcloud(points_xyz: Sequence[FloatTripleType], stamp: Timestamp, frame_id: str, rgba_u8: RGBAType) -> PointCloud:
    r, g, b, a = [int(x) & 0xFF for x in rgba_u8]
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

def poses_to_pointcloud(poses: Sequence[Pose], stamp: Timestamp, frame_id: str, rgba_u8: RGBAType, z: float = 0.0) -> PointCloud:
    return points_to_pointcloud([(p.x, p.y, float(z)) for p in poses], stamp, frame_id=frame_id, rgba_u8=rgba_u8)


# -----------------------------------------------------------------------------
# occupancy grid and pose generation helper functions
# -----------------------------------------------------------------------------

def occupancy_to_grid_rgba(occ: OccupancyGrid, stamp: Timestamp, frame_id: str = "map", flip_y: bool = False) -> Grid:
    """ minimal occupancy visualization (occupied is dark & opaque; free is transparent) """
    # expects: occ.occ is 2D bool-like [rows, cols], occ.grid.resolution, occ.grid.origin_xy
    grid = occ.grid
    res = float(grid.resolution)
    xo, yo = grid.origin_xy
    rows, cols = occ.occ.shape
    buf = bytearray(4 * cols * rows)
    for y in range(rows):
        yy = (rows - 1 - y) if flip_y else y
        for x in range(cols):
            i = 4 * (y * cols + x)
            buf[i:i+4] = bytes((65, 65, 65, 190)) if occ.occ[yy, x] else bytes((0, 0, 0, 0))
    pose = FGPose(position=Vector3(x=xo, y=yo, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1))
    # pose = _to_fg_pose_xyyaw(xo, yo, yaw=0.0, z=0.0)
    return Grid(
        timestamp=stamp,
        frame_id=frame_id,
        pose=pose,
        column_count=cols,
        cell_size=Vector2(x=res, y=res),
        row_stride=int(4 * cols),
        cell_stride=4,
        fields=FIELDS_RGBA,
        data=bytes(buf),
    )


def _occupied_rectangles(occ: OccupancyGrid) -> List[Tuple[int, int, int, int]]:
    """ return merged occupied rectangles as (ix0, iy0, ix1, iy1) w/ right endpoints excluded """
    rows, cols = occ.occ.shape
    used = [[False] * cols for _ in range(rows)]
    rects: List[Tuple[int, int, int, int]] = []
    for iy in range(rows):
        for ix in range(cols):
            if used[iy][ix] or not bool(occ.occ[iy, ix]):
                continue
            x1 = ix + 1
            while x1 < cols and bool(occ.occ[iy, x1]) and not used[iy][x1]:
                x1 += 1
            y1 = iy + 1
            while y1 < rows:
                row_ok = True
                for x in range(ix, x1):
                    if used[y1][x] or not bool(occ.occ[y1, x]):
                        row_ok = False
                        break
                if not row_ok:
                    break
                y1 += 1
            for yy in range(iy, y1):
                for xx in range(ix, x1):
                    used[yy][xx] = True
            rects.append((ix, iy, x1, y1))
    return rects


def _obstacle_cubes(occ_grid: OccupancyGrid, cfg: VizConfig) -> List[CubePrimitive]:
    cubes: List[CubePrimitive] = []
    res = float(occ_grid.grid.resolution)
    ox, oy = occ_grid.grid.origin_xy
    height = float(cfg.static.obstacles_height_m)
    r, g, b, a = cfg.static.obstacles_rgba_u8
    rgb = (r / 255.0, g / 255.0, b / 255.0)
    # color = Color(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=a / 255.0)
    for ix0, iy0, ix1, iy1 in _occupied_rectangles(occ_grid):
        width = (ix1 - ix0) * res
        depth = (iy1 - iy0) * res
        cx = ox + (ix0 + ix1) * 0.5 * res
        cy = oy + (iy0 + iy1) * 0.5 * res
        cubes.append(_cube((cx, cy, 0.0), (width, depth, height), rgb=rgb, alpha=a / 255.0, z=height / 2.0))
    return cubes



# -----------------------------------------------------------------------------
# scene construction and static element helper functions
# -----------------------------------------------------------------------------


def _static_scene(stamp: Timestamp, cfg: VizConfig, start_pose: Optional[Pose], goal: Optional[GoalSpec], occ_grid: Optional[OccupancyGrid]) -> SceneUpdate:
    # static elements that don't change per tick (start/goal, maybe grid and obstacles too) can be logged once at the beginning instead of every tick
    zero_lifetime = Duration(sec=0, nsec=0)
    ents: List[SceneEntity] = []
    common_kwargs = dict(frame_id=cfg.frame_id, timestamp=stamp, lifetime=zero_lifetime, frame_locked=False, metadata=[])
    if start_pose is not None:
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/start_pose",
                spheres=[_sphere(start_pose, radius=0.5 * cfg.static.start_pose_diam, rgb=(0.2, 1.0, 0.2), alpha=1.0)],
            )
        )
    if goal is not None:
        # "diamond" goal: rotate cube 45 degrees in yaw (looks rhomboid-ish top-down)
        yaw = float(goal.pose.theta + math.pi / 4.0)
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/goal_pose",
                cubes=[
                    _cube(
                        pos=(goal.pose.x, goal.pose.y, yaw),
                        size=cfg.static.goal_marker_size,
                        rgb=(1.0, 0.35, 0.15),
                        alpha=0.95,
                        z=0.15,
                    )
                ],
            )
        )
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/goal_tolerance",
                cylinders=[_goal_tolerance(goal, height=cfg.static.goal_tol_height, alpha=0.25)]
            )
        )
    if occ_grid is not None:
        ents.append(SceneEntity(**common_kwargs, id="planner/obstacles", cubes=_obstacle_cubes(occ_grid, cfg)))
    return SceneUpdate(deletions=[], entities=ents)


#!! FIXME: needs work
def tick_scene_entity_ids(tick: PlannerTick, vehicle: Optional[VehicleParams] = None, show_text: bool = True, analytic_shot_enabled: bool = True) -> List[str]:
    ids = [
        "planner/current_pose", "planner/best_pose", "planner/trajectory", "planner/pruned_trajectories",
        "planner/pruned_terminals", "planner/explored_edges", "planner/explored_nodes", "planner/collision_nodes"
    ]
    if analytic_shot_enabled and tick.analytic_shot:
        ids.append("planner/analytic_shot")
    if vehicle is not None:
        ids.append("planner/vehicle")
    if show_text:
        ids.append("planner/hud")
    return ids


def tick_to_scene_update(
    tick: PlannerTick,
    cfg: VizConfig,
    vehicle: Optional[VehicleParams] = None,
    # show_text: bool = True,
) -> SceneUpdate:
    stamp = _ts_from_time_s(float(tick.time_s))
    zero_lifetime = Duration(sec=0, nsec=0)
    common_kwargs = dict(frame_id=cfg.frame_id, timestamp=stamp, lifetime=zero_lifetime, frame_locked=False, metadata=[])
    traj_line = _line_strip(tick.trajectory, rgb=cfg.trajectory.rgb, thickness=cfg.trajectory.thickness, alpha=cfg.trajectory.alpha)
    # vehicle current pose marker (sphere)
    #!!! FIXME: I think the majority of these ids may be invalid - or at least I can't find them when opening the MCAP file in Foxglove Studio
    ents: List[SceneEntity] = [
        SceneEntity(
            **common_kwargs, id="planner/current_pose",
            spheres=[_sphere(tick.pose, radius=0.5 * cfg.current_pose_diam, rgb=(0.2, 0.6, 1.0), alpha=1.0)]
        ),
        # TODO: consider dropping best_pose entirely - should be apparent when it's taken as the next "current pose" anyway
        SceneEntity(
            **common_kwargs, id="planner/best_pose",
            spheres=[_sphere(tick.best_pose, radius=0.5 * cfg.best_pose_diam, rgb=(0.0, 1.0, 0.2), alpha=0.8)]
        ),
        SceneEntity(
            **common_kwargs, id="planner/trajectory",
            # lines = _trajectory_segments(tick.trajectory, cfg=cfg),
            lines=[] if traj_line is None else [traj_line],
            spheres=_trajectory_pose_markers(tick.trajectory, cfg),
        ),
        SceneEntity(
            **common_kwargs, id="planner/pruned_trajectories",
            lines=_pruned_tendrils(tick.pruned_trajectories, cfg=cfg),
        ),
        SceneEntity(
            **common_kwargs,
            id="planner/pruned_terminals",
            spheres=[
                _sphere(path[-1], radius=0.5 * cfg.explored.node_diam, rgb=cfg.explored.node_rgba[:3], alpha=cfg.explored.node_rgba[3]/255.0)
                for path in tick.pruned_trajectories
                if path
            ],
        ),
        SceneEntity(
            **common_kwargs,
            id="planner/explored_edges",
            lines=_explored_edge_lines(tick.explored_edges, cfg),
        ),
        SceneEntity(
            **common_kwargs,
            id="planner/explored_nodes",
            spheres=[_sphere(p, radius=0.04, rgb=(0.55, 0.80, 1.0), alpha=0.65) for p in tick.explored_poses],
        ),
        SceneEntity(
            **common_kwargs,
            id="planner/collision_nodes",
            spheres=[_sphere(p, radius=0.05, rgb=(1.0, 0.2, 0.2), alpha=0.8) for p in tick.collision_poses],
        ),
    ]  # type: List[SceneEntity]  # current pose, best pose, trajectory, pruned trajectories, pruned terminals, explored edges, explored nodes, collision nodes

    if cfg.analytic_shot.enabled and tick.analytic_shot:
        analytic_line = _line_strip(
            tick.analytic_shot,
            rgb=cfg.analytic_shot.rgb,
            thickness=cfg.analytic_shot.thickness,
            alpha=cfg.analytic_shot.alpha,
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
                cubes=[_vehicle_footprint_cube(tick.pose, vehicle)],
            )
        )
    if cfg.hud.enabled:
        # lightweight HUD near current pose
        hud = f"it={tick.iteration}  open={tick.open_size}\nf={tick.best_f:.2f}  g={tick.best_g:.2f}"
        ents.append(
            SceneEntity(
                **common_kwargs, id="planner/hud",
                texts=[
                    TextPrimitive(
                        pose=_to_fg_pose_xyyaw(cfg.hud.anchor_xyz[0], cfg.hud.anchor_xyz[1], yaw=0.0, z=cfg.hud.anchor_xyz[2]),
                        billboard=True,
                        font_size=cfg.hud.font_size,
                        scale_invariant=True,
                        color=Color(r=1.0, g=1.0, b=1.0, a=cfg.hud.alpha),
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
) -> None:
    sink = FoxgloveTickSink(out_path, viz=viz, occ_grid=occ_grid, start_pose=start_pose, goal=goal, vehicle=vehicle)
    # iterate through ticks and log to channels
    try:
        for t in ticks:
            sink(t)
    finally:
        sink.close()
