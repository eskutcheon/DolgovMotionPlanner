# src/dolgov_cbmp/settings/world.py

from dataclasses import asdict
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import pickle
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field
# project imports
from dolgov_cbmp.structs import GoalSpec, GridSpec, Pose, WorldModel
from dolgov_cbmp.settings import VehicleParams, PlannerConfig
from dolgov_cbmp.models import OccupancyGrid

class OccupancyInlineSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: List[List[Union[int, bool]]]


class OccupancyFileSource(BaseModel):
    """ Reference to an external occupancy payload, which will be loaded according to the file extension
        Loader infers behavior from the file extension:
        - .npz: expects an array under `key` (default "occupancy")
        - .npy: ignores `key`
        - .h5/.hdf5: expects dataset path in `key`
        - image (.png/.pgm/...): loads grayscale and thresholds to produce bool occupancy;
    """
    model_config = ConfigDict(extra="forbid")
    path: str
    key: str = "occupancy"
    # image-specific options (ignored for non-image formats)
    # TODO: need to reuse some stuff from "scripts/npz_from_maze_images` to support this option
    occupied_thresh: float = Field(default=0.5, ge=0.0, le=1.0)  # occupied if value <= thresh


class WorldConfigSchema(BaseModel):
    """ Schema for world files consumed via `--world-cfg` CLI argument, which can contain both world state and planner config overrides
        Contains:
        - world state (grid spec, occupancy, start/goal)
        - optional vehicle params
        - optional planner config overrides (merged into passed PlannerConfig)
    """
    model_config = ConfigDict(extra="forbid")
    format_version: str = "1.0" # added primarily for dealing with any legacy grid files
    grid: GridSpec
    occupancy: Union[OccupancyInlineSource, OccupancyFileSource]
    start: Pose
    goal: GoalSpec
    # step_size: Optional[float] = Field(default=None, gt=0.0, le=50.0)
    vehicle: Optional[VehicleParams] = None
    planner_overrides: Dict[str, Any] = Field(default_factory=dict)


def _require_module(name: str, message: str):
    if importlib.util.find_spec(name) is None:
        raise ImportError(message)
    return importlib.import_module(name)

def _resolve_path(path: Union[str, Path], *, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    world_relative = (base_dir / candidate).resolve()
    if world_relative.is_file():
        return world_relative
    return candidate.resolve()


def _parse_doc_blob(blob: Any) -> Dict[str, Any]:
    if isinstance(blob, np.ndarray) and blob.shape == ():
        blob = blob.item()
    if isinstance(blob, dict):
        return blob
    if isinstance(blob, np.bytes_):
        blob = bytes(blob)
    if isinstance(blob, bytes):
        try:
            blob = blob.decode("utf-8").strip()
        except UnicodeDecodeError:
            payload = pickle.loads(blob)
            if not isinstance(payload, dict):
                raise ValueError("Serialized payload must decode to a mapping/object")
            return payload
    if not isinstance(blob, str):
        raise ValueError(f"Unsupported serialized world-doc type: {type(blob).__name__}")
    text = blob.strip()
    if not text:
        return {}
    if text[0] in "[{":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("World config root must be an object/mapping")
    return payload


def _load_occupancy_from_image(path: Path, *, occupied_thresh: float) -> np.ndarray:
    pil_image = _require_module("PIL.Image", "Pillow is required to load occupancy image files")
    with pil_image.open(path).convert("L") as img:
        values = np.asarray(img, dtype=np.float32) / 255.0
    return values <= float(occupied_thresh)


def _load_doc(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".jsonl":
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) != 1:
            raise ValueError("JSONL world config must contain exactly one JSON object line")
        payload = json.loads(lines[0])
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            payload = pickle.load(f)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            for key in ("world_cfg", "world", "config", "payload", "doc"):
                if key in npz.files:
                    return _parse_doc_blob(npz[key])
        raise ValueError("NPZ world config did not contain a serialized config document")
    elif suffix in {".hdf5", ".h5", ".hdf"}:
        h5py = _require_module("h5py", "h5py is required to load HDF5 world config files")
        with h5py.File(path, "r") as h5:
            for key in ("world_cfg", "world", "config", "payload", "doc"):
                if key in h5:
                    return _parse_doc_blob(h5[key][()])
                if key in h5.attrs:
                    return _parse_doc_blob(h5.attrs[key])
        raise ValueError("HDF5 world config did not contain a serialized config document")
    # elif suffix in {".pt", ".pth"}:
    #     torch = _require_module("torch", "torch is required to load .pt/.pth world config files")
    #     payload = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported world config extension: {suffix}")
    if not isinstance(payload, dict):
        raise ValueError("World config root must be an object/mapping")
    return payload


def _deep_merge(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _load_occupancy(source: Union[OccupancyInlineSource, OccupancyFileSource], base_path: Union[Path, str]) -> np.ndarray:
    if isinstance(source, OccupancyInlineSource):
        return np.asarray(source.data, dtype=bool)
    # occ_path = Path(source.path).expanduser()
    # if not occ_path.is_absolute():
    #     direct_path = occ_path.resolve()
    #     base_path = Path(base_path)
    #     occ_path = direct_path if direct_path.is_file() else (base_path / occ_path).resolve()
    #     print("Resolved relative occ_path to:", occ_path)
    occ_path = _resolve_path(source.path, base_dir=Path(base_path))
    if not occ_path.is_file():
        raise FileNotFoundError(f"Occupancy source file does not exist: {occ_path}")
    suffix = occ_path.suffix.lower()
    if suffix == ".npz":
        with np.load(occ_path, allow_pickle=True) as npz:
            if source.key not in npz.files:
                raise ValueError(f"Key '{source.key}' not found in {occ_path} (available: {npz.files})")
            return np.asarray(npz[source.key], dtype=bool)
    if suffix == ".npy":
        return np.asarray(np.load(occ_path), dtype=bool)
    if suffix in {".hdf5", ".h5", ".hdf"}:
        h5py = _require_module("h5py", "h5py is required to load HDF5 occupancy files")
        with h5py.File(occ_path, "r") as h5:
            if source.key not in h5:
                raise ValueError(f"Dataset '{source.key}' not found in {occ_path}")
            return np.asarray(h5[source.key][()], dtype=bool)
    if suffix in {".png", ".pgm", ".ppm", ".pbm", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}:
        return _load_occupancy_from_image(occ_path, occupied_thresh=source.occupied_thresh)
    raise ValueError(f"Unsupported occupancy file extension: {occ_path.suffix}")


def load_world_model(world_cfg_path: Union[str, Path], planner_config: PlannerConfig) -> Tuple[WorldModel, PlannerConfig]:
    """ load world state and optional planner overrides from a world config file """
    path = Path(world_cfg_path).expanduser().resolve()
    # suffix = path.suffix.lower()
    # # Backward compatibility for existing maze .npz files
    # # TODO: update for newer .npz schemas that may contain grid specs and other world parameters
    # if suffix == ".npz":
    raw_doc: Optional[Dict[str, Any]] = None
    load_error: Optional[Exception] = None
    try:
        raw_doc = _load_doc(path)
    except Exception as exc:
        load_error = exc
    if raw_doc is None and path.suffix.lower() == ".npz":
        occ_grid, start_xy, goal_xy = OccupancyGrid.grid_from_file(path, GridSpec())
        start = Pose(float(start_xy[0]), float(start_xy[1]), 0.0, 0.0)
        goal = GoalSpec(Pose(float(goal_xy[0]), float(goal_xy[1]), 0.0, 0.0))
        world = WorldModel(occupancy_grid=occ_grid, start=start, goal=goal, vehicle=planner_config.vehicle)
        world.validate()
        return world, planner_config
    # raw_doc = _load_doc(path)
    if raw_doc is None:
        assert load_error is not None, "Unexpected error: failed to load world config but no exception was raised"
        raise load_error
    world_cfg = WorldConfigSchema.model_validate(raw_doc)
    planner_overrides = dict(world_cfg.planner_overrides)
    if "grid" in planner_overrides:
        raise ValueError("planner_overrides.grid is not supported; world grid must be defined at world root")
    planner_payload = asdict(planner_config)
    if world_cfg.vehicle is not None and "vehicle" not in planner_overrides:
        planner_payload["vehicle"] = asdict(world_cfg.vehicle)
    _deep_merge(planner_payload, planner_overrides)
    updated_cfg = PlannerConfig(**planner_payload)
    occ = _load_occupancy(world_cfg.occupancy, base_path=path.parent)
    world = WorldModel(
        occupancy_grid=OccupancyGrid(occ, world_cfg.grid),
        start=world_cfg.start,
        goal=world_cfg.goal,
        vehicle=updated_cfg.vehicle,
    )
    world.validate()
    return world, updated_cfg