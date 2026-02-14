# src/telemetry/foxglove.py
from pathlib import Path
import math
import struct
from dataclasses import dataclass #, asdict
from typing import Iterable, Sequence, Tuple, Optional, List, TypeAlias
# local imports
from src.structs import PlannerTick, Pose, GoalSpec, VehicleParams
from src.models.models import OccupancyGrid

try:
    import foxglove
except (ModuleNotFoundError, ImportError) as e:
    raise ImportError("foxglove-sdk is not installed. Install it with `pip install foxglove-sdk`") from e

from foxglove import Channel
from foxglove.channels import SceneUpdateChannel, PointCloudChannel, GridChannel
from foxglove.schemas import (
    SceneUpdate, SceneEntity, Timestamp, Duration,
    SpherePrimitive, LinePrimitive, CubePrimitive, TextPrimitive,
    Grid, Vector2, Vector3, Quaternion,
    Pose as FGPose, Color, Point3,
    PointCloud, PackedElementField, PackedElementFieldNumericType as NumericType,
)

# Done just for brevity for all the RGB and XYZ tuples
TripleType: TypeAlias = Tuple[float, float, float]

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


@dataclass(frozen=True)
class VizConfig:
    frame_id: str = "map"
    # Scene
    trajectory_thickness: float = 0.10
    trajectory_alpha: float = 0.90
    # distance-based gradient: dark near vehicle -> lighter far away
    traj_dark_rgb: TripleType = (0.05, 0.15, 0.35)
    traj_light_rgb: TripleType = (0.65, 0.85, 1.00)
    # curvature "warms" the color (more orange) as |kappa| grows
    kappa_ref: float = 0.25  # 1/m where warming saturates
    # Markers
    current_pose_diam: float = 0.25
    best_pose_diam: float = 0.30
    start_pose_diam: float = 0.45
    goal_marker_size: TripleType = (0.45, 0.45, 0.20)
    # Vehicle cube footprint
    vehicle_height: float = 0.35
    # Point clouds
    explored_rgba_u8: Tuple[int, int, int, int] = (120, 190, 255, 170)
    collisions_rgba_u8: Tuple[int, int, int, int] = (255, 60, 60, 240)
    obstacles_rgba_u8: Tuple[int, int, int, int] = (220, 220, 220, 210)
    obstacles_z_m: float = 1.0
    obstacles_stride_cells: int = 1
    max_obstacle_points: int = 50_000
    # Grid
    grid_flip_y: bool = False  # matches OccupancyGrid.world_to_grid / grid_to_world

    # @staticmethod
    # def from_planner_config(cfg: PlannerConfig) -> "VizConfig":
    # TODO: [MAYBE] add static method to initialize some variables with an existing PlannerConfig


# TODO: might consider making this a context manager that opens the MCAP file on init and closes on exit, and then just pass the `log_tick` method as the callback to the planner
class FoxgloveTickSink:
    """ Streaming writer - used like `planner.tick_callback` by passing to `planner.plan`, then cleaning up with `close` """
    def __init__(
        self,
        out_path: str | Path,
        *,
        viz: VizConfig = VizConfig(),
        occ_grid: Optional[OccupancyGrid] = None,
        start_pose: Optional[Pose] = None,
        goal: Optional[GoalSpec] = None,
        vehicle: Optional[VehicleParams] = None,
    ):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.viz = viz
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
        self.explored_ch = PointCloudChannel("/planner/explored")
        self.collisions_ch = PointCloudChannel("/planner/collisions")
        # optional static channels for occupancy grid and obstacles
        self.grid_ch: Optional[GridChannel] = None
        self.obstacles_ch: Optional[PointCloudChannel] = None
        # static layers (grid, obstacles, start/goal)
        t0 = ts_from_time_s(0.0)
        if self.occ_grid is not None:
            self.grid_ch = GridChannel("/planner/grid")
            self.grid_ch.log(
                occupancy_to_grid_rgba(self.occ_grid, t0, frame_id=self.viz.frame_id, flip_y=self.viz.grid_flip_y)
            )
            self.obstacles_ch = PointCloudChannel("/planner/obstacles")
            obs_pts = occupied_cells_to_points(
                self.occ_grid,
                z_m=self.viz.obstacles_z_m,
                stride_cells=self.viz.obstacles_stride_cells,
                max_points=self.viz.max_obstacle_points,
            )
            self.obstacles_ch.log(
                points_to_pointcloud(obs_pts, t0, frame_id=self.viz.frame_id, rgba_u8=self.viz.obstacles_rgba_u8)
            )
        self.scene_ch.log(
            _static_scene(stamp=t0, cfg=self.viz, start_pose=self.start_pose, goal=self.goal)
        )

    def __call__(self, tick: PlannerTick) -> None:
        stamp = ts_from_time_s(float(tick.time_s))
        # "tick" JSON is useful for tables/raw inspection
        self.tick_ch.log(tick.to_scalar_dict())
        # log the scene update with current pose, best pose, and trajectory
        self.scene_ch.log(tick_to_scene_update(tick, cfg=self.viz, vehicle=self.vehicle, show_text=True))
        self.explored_ch.log(
            poses_to_pointcloud(tick.explored_poses, stamp, frame_id=self.viz.frame_id, rgba_u8=self.viz.explored_rgba_u8)
        )
        self.collisions_ch.log(
            poses_to_pointcloud(tick.collision_poses, stamp, frame_id=self.viz.frame_id, rgba_u8=self.viz.collisions_rgba_u8)
        )

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)


# -----------------------------------------------------------------------------
# time and geometry resolution helper functions
# -----------------------------------------------------------------------------

def ts_from_time_s(t_s: float) -> Timestamp:
    sec = int(t_s)
    nsec = int(round((t_s - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec = 0
    return Timestamp(sec=sec, nsec=nsec)

def yaw_to_quaternion(yaw: float) -> Quaternion:
    # rotation about +Z
    return Quaternion(x=0, y=0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))

def to_fg_pose_xyyaw(x: float, y: float, yaw: float, z: float = 0.0) -> FGPose:
    pos_vector = Vector3(x=float(x), y=float(y), z=float(z))
    return FGPose(position=pos_vector, orientation=yaw_to_quaternion(float(yaw)))

def to_fg_pose(p: Pose, z: float = 0.0) -> FGPose:
    return to_fg_pose_xyyaw(p.x, p.y, p.theta, z=z)

def _mix_rgb(a: TripleType, b: TripleType, t: float) -> TripleType:
    """ linearly interpolate between two RGB colors a and b with parameter t in [0, 1] """
    return (ai + t * (bi - ai) for ai, bi in zip(a, b))

def _warm_by_curvature(rgb: TripleType, kappa: float, kappa_ref: float) -> TripleType:
    # push towards orange as |kappa| increases
    kn = abs(float(kappa)) / max(1e-9, float(kappa_ref))
    kn = min(max(kn, 0.0), 1.0)
    orange = (1.0, 0.55, 0.10) # orange target
    return _mix_rgb(rgb, orange, 0.55 * kn)


# -----------------------------------------------------------------------------
# point cloud generation helper functions
# -----------------------------------------------------------------------------

def points_to_pointcloud(
    points_xyz: Sequence[TripleType],
    stamp: Timestamp,
    frame_id: str,
    rgba_u8: Tuple[int, int, int, int],
) -> PointCloud:
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

def poses_to_pointcloud(
    poses: Sequence[Pose],
    stamp: Timestamp,
    frame_id: str,
    rgba_u8: Tuple[int, int, int, int],
    z: float = 0.0,
) -> PointCloud:
    return points_to_pointcloud([(p.x, p.y, float(z)) for p in poses], stamp, frame_id=frame_id, rgba_u8=rgba_u8)


# -----------------------------------------------------------------------------
# occupancy grid and pose generation helper functions
# -----------------------------------------------------------------------------

def occupancy_to_grid_rgba(occ: OccupancyGrid, stamp: Timestamp, frame_id: str = "map", flip_y: bool = True) -> Grid:
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
            buf[i:i+4] = bytes((30, 30, 30, 255)) if occ.occ[yy, x] else bytes((0, 0, 0, 0))
    pose = FGPose(position=Vector3(x=xo, y=yo, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1))
    # pose = to_fg_pose_xyyaw(xo, yo, yaw=0.0, z=0.0)
    return Grid(
        timestamp=stamp,
        frame_id=frame_id,
        pose=pose,
        column_count=cols,
        cell_size=Vector2(x=res, y=res),
        row_stride=int(4 * cols),
        cell_stride=4,
        fields=FIELDS_XYZ_RGBA[3:],  # only RGBA fields (Grid schema doesn't use XYZ)
        data=bytes(buf),
    )

def occupied_cells_to_points(occ: OccupancyGrid, z_m: float, stride_cells: int, max_points: int) -> List[TripleType]:
    """ Convert occupied cells to world XYZ points at fixed z using OccupancyGrid.grid_to_world scaling """
    stride = max(1, int(stride_cells))
    rows, cols = occ.occ.shape
    pts: List[TripleType] = []
    # simple deterministic subsampling (stride)
    # TODO: might want to keep a default stride of 1
    for iy in range(0, rows, stride):
        for ix in range(0, cols, stride):
            if bool(occ.occ[iy, ix]):
                x, y = occ.grid_to_world(ix, iy)
                pts.append((float(x), float(y), float(z_m)))
                if len(pts) >= int(max_points):
                    return pts
    return pts



# -----------------------------------------------------------------------------
# scene construction and static element helper functions
# -----------------------------------------------------------------------------

def _trajectory_segments(traj: Sequence[Pose], cfg: VizConfig) -> List[LinePrimitive]:
    if len(traj) < 2:
        return []
    # cumulative distance for gradient parameterization
    ds: List[float] = [0.0]
    total = 0.0
    for i in range(1, len(traj)):
        dx = float(traj[i].x - traj[i-1].x)
        dy = float(traj[i].y - traj[i-1].y)
        d = math.hypot(dx, dy)
        total += d
        ds.append(total)
    total = max(total, 1e-9)
    # create line segments with color based on distance along trajectory and curvature
    segs: List[LinePrimitive] = []
    for i in range(len(traj) - 1):
        # t at segment midpoint
        t = 0.5 * (ds[i] + ds[i+1]) / total
        rgb = _mix_rgb(cfg.traj_dark_rgb, cfg.traj_light_rgb, t)
        # warm by curvature (use midpoint kappa)
        kmid = 0.5 * (float(traj[i].kappa) + float(traj[i+1].kappa))
        rgb = _warm_by_curvature(rgb, kmid, cfg.kappa_ref)
        # add line segment from traj[i] to traj[i+1] with color and thickness
        segs.append(
            LinePrimitive(
                thickness=float(cfg.trajectory_thickness),
                scale_invariant=False,
                points=[
                    Point3(x=float(traj[i].x), y=float(traj[i].y), z=0.0),
                    Point3(x=float(traj[i+1].x), y=float(traj[i+1].y), z=0.0),
                ],
                color=Color(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(cfg.trajectory_alpha)),
            )
        )
    return segs


# TODO: might simplify this slightly so it's just a gray footprint shadow
def _vehicle_cube(pose: Pose, vehicle: VehicleParams, cfg: VizConfig) -> CubePrimitive:
    # pose is rear-axle center - cube expects center to offset along +x in vehicle frame
    length = float(vehicle.length)
    width = float(vehicle.width)
    height = float(cfg.vehicle_height)
    dx_center = 0.5 * (float(vehicle.wheelbase + vehicle.front_overhang - vehicle.rear_overhang))
    # center offset in world frame: (dx_center, 0) rotated by pose.theta
    cx = float(pose.x) + dx_center * math.cos(float(pose.theta))
    cy = float(pose.y) + dx_center * math.sin(float(pose.theta))
    cz = 0.5 * height
    return CubePrimitive(
        pose=to_fg_pose_xyyaw(cx, cy, yaw=float(pose.theta), z=cz),
        size=Vector3(x=length, y=width, z=height),
        color=Color(r=0.95, g=0.95, b=0.95, a=0.35),
    )


def _static_scene(stamp: Timestamp, cfg: VizConfig, start_pose: Optional[Pose], goal: Optional[GoalSpec]) -> SceneUpdate:
    # static elements that don't change per tick (start/goal, maybe grid and obstacles too) can be logged once at the beginning instead of every tick
    zero = Duration(sec=0, nsec=0)
    ents: List[SceneEntity] = []
    if start_pose is not None:
        ents.append(
            SceneEntity(
                frame_id=cfg.frame_id,
                id="planner/start_pose",
                timestamp=stamp,
                lifetime=zero,
                frame_locked=False,
                metadata=[],
                spheres=[
                    SpherePrimitive(
                        pose=to_fg_pose(start_pose),
                        size=Vector3(x=cfg.start_pose_diam, y=cfg.start_pose_diam, z=cfg.start_pose_diam),
                        color=Color(r=0.2, g=1.0, b=0.2, a=1.0),
                    )
                ],
            )
        )
    if goal is not None:
        # "diamond" goal: rotate cube 45 degrees in yaw (looks rhomboid-ish top-down)
        yaw = float(goal.pose.theta + math.pi / 4.0)
        ents.append(
            SceneEntity(
                frame_id=cfg.frame_id,
                id="planner/goal_pose",
                timestamp=stamp,
                lifetime=zero,
                frame_locked=False,
                metadata=[],
                cubes=[
                    CubePrimitive(
                        pose=to_fg_pose_xyyaw(goal.pose.x, goal.pose.y, yaw=yaw, z=0.10),
                        size=Vector3(
                            x=float(cfg.goal_marker_size[0]),
                            y=float(cfg.goal_marker_size[1]),
                            z=float(cfg.goal_marker_size[2]),
                        ),
                        color=Color(r=1.0, g=0.35, b=0.15, a=0.95),
                    )
                ],
            )
        )
    return SceneUpdate(deletions=[], entities=ents)




def tick_to_scene_update(
    tick: PlannerTick,
    cfg: VizConfig,
    vehicle: Optional[VehicleParams] = None,
    show_text: bool = True,
) -> SceneUpdate:
    stamp = ts_from_time_s(float(tick.time_s))
    zero_lifetime = Duration(sec=0, nsec=0)
    common_kwargs = dict(
        frame_id=cfg.frame_id,
        timestamp=stamp,
        lifetime=zero_lifetime,
        frame_locked=False,
        metadata=[],
    )
    # vehicle current pose marker (sphere)
    current = SceneEntity(
        **common_kwargs,
        id="planner/current_pose",
        spheres=[
            SpherePrimitive(
                pose=to_fg_pose(tick.pose),
                # size=Vector3(x=0.20, y=0.20, z=0.20),  # diameter
                size=Vector3(x=cfg.current_pose_diam, y=cfg.current_pose_diam, z=cfg.current_pose_diam), # diameter
                color=Color(r=0.2, g=0.6, b=1.0, a=1.0),
            )
        ],
    )
    best = SceneEntity(
        **common_kwargs,
        id="planner/best_pose",
        spheres=[
            SpherePrimitive(
                pose=to_fg_pose(tick.best_pose),
                # size=Vector3(x=0.25, y=0.25, z=0.25),
                size=Vector3(x=cfg.best_pose_diam, y=cfg.best_pose_diam, z=cfg.best_pose_diam),
                color=Color(r=0.0, g=1.0, b=0.2, a=1.0),
            )
        ],
    )
    # traj_pts = [Point3(x=float(p.x), y=float(p.y), z=0) for p in tick.trajectory]
    trajectory = SceneEntity(
        **common_kwargs,
        id="planner/trajectory",
        lines = _trajectory_segments(tick.trajectory, cfg=cfg),
    )
    ents = [current, best, trajectory]
    # add vehicle footprint cube if we have the needed params
    if vehicle is not None:
        ents.append(
            SceneEntity(
                **common_kwargs,
                id="planner/vehicle",
                cubes=[_vehicle_cube(tick.pose, vehicle, cfg)],
            )
        )
    if show_text:
        # lightweight HUD near current pose
        hud = f"it={tick.iteration}  open={tick.open_size}\nf={tick.best_f:.2f}  g={tick.best_g:.2f}"
        ents.append(
            SceneEntity(
                **common_kwargs,
                id="planner/hud",
                texts=[
                    TextPrimitive(
                        pose=to_fg_pose_xyyaw(tick.pose.x, tick.pose.y, yaw=0.0, z=1.2),
                        billboard=True,
                        font_size=18.0,
                        scale_invariant=True,
                        color=Color(r=1.0, g=1.0, b=1.0, a=0.95),
                        text=hud,
                    )
                ],
            )
        )
    return SceneUpdate(deletions=[], entities=ents)


# TODO: might be worth making this a static method of `FoxgloveTickSink` and just have it open/close the MCAP file internally, since that's really the main use case - need to pass ticks and output path
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
    # out_path = Path(out_path)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # # create channels inside the open_mcap context - Reference: https://foxglove.dev/blog/using-the-foxglove-sdk-to-generate-mcap
    # with foxglove.open_mcap(out_path, allow_overwrite=True):
    #     scene_ch = SceneUpdateChannel("/planner/scene")
    #     tick_ch = Channel("/planner/tick", message_encoding="json") # schemaless JSON should work
    #     explored_ch = PointCloudChannel("/planner/explored")
    #     collisions_ch = PointCloudChannel("/planner/collisions")
    #     if occ_grid is not None:
    #         grid_ch = GridChannel("/planner/grid")
    #         grid_ch.log(occupancy_to_grid_rgba(occ_grid, ts_from_time_s(0.0), frame_id=frame_id))
    #     # iterate through ticks and log to channels
    #     for t in ticks:
    #         stamp = ts_from_time_s(float(t.time_s))
    #         tick_ch.log(t.to_scalar_dict())  # stats/debug view
    #         scene_ch.log(tick_to_scene_update(t, frame_id=frame_id))
    #         explored_ch.log(poses_to_pointcloud(t.explored_poses, stamp, frame_id=frame_id, rgba=(153, 204, 255, 190)))
    #         collisions_ch.log(poses_to_pointcloud(t.collision_poses, stamp, frame_id=frame_id, rgba=(255, 50, 50, 242)))
    sink = FoxgloveTickSink(out_path, viz=viz, occ_grid=occ_grid, start_pose=start_pose, goal=goal, vehicle=vehicle)
    # iterate through ticks and log to channels
    try:
        for t in ticks:
            sink(t)
    finally:
        sink.close()

