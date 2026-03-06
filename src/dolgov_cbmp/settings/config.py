# src/dolgov_cbmp/settings/config.py

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator #, model_validator

from dolgov_cbmp.structs import GoalSpec, GridSpec, PlannerConfig, Pose, VehicleParams


# TODO: really considering moving most of these methods to cli.py and moving all things related to PlannerConfig to this file

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

#------------------------------------------------------------------------------
# helper methods used in CLI parsing
#   - defined here to cut down imports since this is where config models for those classes are defined
#------------------------------------------------------------------------------

class DotPathOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    value: Any

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if not value or value.startswith(".") or value.endswith("."):
            raise ValueError("override path must be a dotted path like 'weights.reverse_penalty'")
        return value


def default_planner_config_dict() -> dict[str, Any]:
    return asdict(
        PlannerConfig(
            grid=GridSpec(resolution=0.5, theta_bins=36, origin_xy=(0.0, 0.0), kappa_bins=11),
            vehicle=VehicleParams(),
        )
    )


def deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def apply_dotted_override(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    node = target
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config root must be a mapping/object")
    return data

