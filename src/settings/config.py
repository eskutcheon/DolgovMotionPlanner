# src/settings/config.py

from dataclasses import asdict
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator #, model_validator

from src.structs import GoalSpec, GridSpec, PlannerConfig, Pose, VehicleParams


class PlanningRunConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = Field(default="python", pattern=r"^(python|cpp)$")
    max_expansions: int = Field(default=100_000, ge=1000, le=1_000_000)
    planner: PlannerConfig #Model
    start: Pose #Model
    goal: GoalSpec #Model


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

