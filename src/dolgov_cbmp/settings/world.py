# src/dolgov_cbmp/settings/world.py

from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import pickle
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field
# from dolgov_cbmp.settings.config import PlannerConfig
from dolgov_cbmp.models import OccupancyGrid
from dolgov_cbmp.structs import GoalSpec, GridSpec, PlannerConfig, Pose, WorldModel


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
    step_size: Optional[float] = Field(default=None, gt=0.0, le=50.0)
    # TODO: will be adding more supported parameters from here like vehicle params stuff and other (sort of) world state stuff



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


def _load_occupancy(source: Union[OccupancyInlineSource, OccupancyFileSource]) -> np.ndarray:
    if isinstance(source, OccupancyInlineSource):
        return np.asarray(source.data, dtype=bool)
    occ_path = Path(source.path).expanduser()
    # if not occ_path.is_absolute():
    #     occ_path = (base_path / occ_path).resolve()
    #     print("Resolved relative occ_path to:", occ_path)
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
    # Backward compatibility for existing maze .npz files.
    if suffix == ".npz":
        occ_grid, start_xy, goal_xy = OccupancyGrid.grid_from_file(path, planner_config.grid)
        start = Pose(float(start_xy[0]), float(start_xy[1]), 0.0, 0.0)
        goal = GoalSpec(Pose(float(goal_xy[0]), float(goal_xy[1]), 0.0, 0.0))
        world = WorldModel(occupancy_grid=occ_grid, start=start, goal=goal, vehicle=planner_config.vehicle)
        world.validate()
        return world, planner_config
    raw_doc = _load_doc(path)
    cfg = WorldConfigSchema.model_validate(raw_doc)
    occ = _load_occupancy(cfg.occupancy)
    world_grid = cfg.grid
    updated_cfg = planner_config
    if planner_config.grid != world_grid:
        planner_kwargs = asdict(planner_config)
        planner_kwargs["grid"] = world_grid.to_dict()
        # planner_kwargs["curvature"]["kappa_bins"] = world_grid.kappa_bins
        # planner_kwargs["curvature"]["kappa_max"] = world_grid.kappa_max
        if cfg.step_size is not None:
            planner_kwargs["step_size"] = cfg.step_size
        updated_cfg = PlannerConfig(**planner_kwargs)
    elif cfg.step_size is not None:
        planner_kwargs = asdict(planner_config)
        planner_kwargs["step_size"] = cfg.step_size
        updated_cfg = PlannerConfig(**planner_kwargs)
    world = WorldModel(
        occupancy_grid=OccupancyGrid(occ, world_grid),
        start=cfg.start,
        goal=cfg.goal,
        vehicle=updated_cfg.vehicle,
    )
    world.validate()
    return world, updated_cfg