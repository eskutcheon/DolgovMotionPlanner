# src/dolgov_cbmp/settings/world.py

from dataclasses import asdict
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
    model_config = ConfigDict(extra="forbid")
    path: str
    key: str = "occupancy"


class WorldConfigSchema(BaseModel):
    """ Schema for world files consumed via `--world-cfg` CLI argument, which can contain both world state and planner config overrides """
    model_config = ConfigDict(extra="forbid")
    format_version: str = "1.0" # added primarily for dealing with any legacy grid files
    grid: GridSpec
    occupancy: Union[OccupancyInlineSource, OccupancyFileSource]
    start: Pose
    goal: GoalSpec
    # step_size: Optional[float] = Field(default=None, gt=0.0, le=50.0)
    vehicle: Optional[VehicleParams] = None
    planner_overrides: Dict[str, Any] = Field(default_factory=dict)



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
    elif suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
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
    occ_path = Path(source.path).expanduser()
    if not occ_path.is_absolute():
        direct_path = occ_path.resolve()
        base_path = Path(base_path)
        occ_path = direct_path if direct_path.is_file() else (base_path / occ_path).resolve()
        print("Resolved relative occ_path to:", occ_path)
    if occ_path.suffix.lower() == ".npz":
        npz = np.load(occ_path)
        if source.key not in npz:
            raise ValueError(f"Key '{source.key}' not found in {occ_path}")
        return np.asarray(npz[source.key], dtype=bool)
    if occ_path.suffix.lower() == ".npy":
        return np.asarray(np.load(occ_path), dtype=bool)
    raise ValueError(f"Unsupported occupancy file extension: {occ_path.suffix}")


def load_world_model(world_cfg_path: Union[str, Path], planner_config: PlannerConfig) -> Tuple[WorldModel, PlannerConfig]:
    """ load world state and optional planner overrides from a world config file """
    path = Path(world_cfg_path).expanduser().resolve()
    suffix = path.suffix.lower()
    # Backward compatibility for existing maze .npz files
    # TODO: update for newer .npz schemas that may contain grid specs and other world parameters
    if suffix == ".npz":
        occ_grid, start_xy, goal_xy = OccupancyGrid.grid_from_file(path, GridSpec())
        start = Pose(float(start_xy[0]), float(start_xy[1]), 0.0, 0.0)
        goal = GoalSpec(Pose(float(goal_xy[0]), float(goal_xy[1]), 0.0, 0.0))
        world = WorldModel(occupancy_grid=occ_grid, start=start, goal=goal, vehicle=planner_config.vehicle)
        world.validate()
        return world, planner_config
    raw_doc = _load_doc(path)
    world_cfg = WorldConfigSchema.model_validate(raw_doc)
    planner_overrides = dict(world_cfg.planner_overrides)
    if "grid" in planner_overrides:
        raise ValueError("planner_overrides.grid is not supported; world grid must be defined at world root")
    planner_payload = asdict(planner_config)
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