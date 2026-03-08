# src/dolgov_cbmp/settings/cli.py

import argparse
import copy
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Sequence, Optional, List, Dict
from pydantic import BaseModel, ConfigDict, Field, field_validator
# project imports
from dolgov_cbmp.structs import GoalSpec, Pose
from dolgov_cbmp.settings.config import PlanningRunConfigModel, VehicleParams, PlannerConfig



WHOLE_PI = 3.14159265 #35897932384626433832795028841971693993751058209749445923
HALF_PI = 1.57079632 #67948966192313216916397514420985846996875529104874722961
# QUARTER_PI = 0.78539816 #33974483096156608458198757210492923498437764552437361480
# THIRD_PI = 1.04719755 # 11965977461542144610931676280657231331250352736583148641
TWELFTH_PI = 0.26179938 #77991494365385536152732919070164307832812588184145787160

class CLIOverridesModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = Field(default="python", pattern=r"^(python|cpp)$")
    max_expansions: int = Field(default=100_000, ge=1, le=1_000_000)
    # TODO: need to change the precedence and conditional dependence of other arguments with this
    #   e.g., don't require start, goal, or grid spec if world_cfg_path is provided
    #   also need some conditional logic in the OccupancyGrid loading and instantiation for a full grid
    world_cfg_path: Optional[str] = Field(default=None, pattern=r".*\.(yaml|yml|json|jsonl|pkl|npz|hdf5)$")
    config_file: Optional[str] = None
    start_x: float = 5.0
    start_y: float = 5.0
    start_theta: float = 0.0
    start_kappa: float = 0.0
    goal_x: float = 80.0
    goal_y: float = 80.0
    goal_theta: float = HALF_PI
    goal_kappa: float = 0.0
    goal_pos_tol: float = Field(default=0.5, ge=0.0, le=10.0)
    goal_theta_tol: float = Field(default=TWELFTH_PI, ge=0.0, le=WHOLE_PI)
    set_values: List[str] = Field(default_factory=list)


@dataclass(frozen=True)
class PlanningInputs:
    backend: str
    max_expansions: int
    world_cfg_path: Optional[str]
    planner_config: PlannerConfig
    start: Pose
    goal: GoalSpec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DolgovMotionPlanner CLI with YAML config and dot-path parameter overrides",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", dest="config_file", help="Path to YAML config file")
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--max-expansions", type=int, default=100_000)
    # TODO: still need to work on this for currently unsupported file types and the precedence logic with other args like start/goal/grid specs
    parser.add_argument("--world-cfg", dest="world_cfg_path", help="Path to world config file (YAML/JSON/PKL) that can override planner config values and provide map data")
    # start pose
    parser.add_argument("--start.x", dest="start_x", type=float, default=5.0)
    parser.add_argument("--start.y", dest="start_y", type=float, default=5.0)
    parser.add_argument("--start.theta", dest="start_theta", type=float, default=0.0)
    parser.add_argument("--start.kappa", dest="start_kappa", type=float, default=0.0)
    # goal pose and tolerances
    parser.add_argument("--goal.x", dest="goal_x", type=float, default=80.0)
    parser.add_argument("--goal.y", dest="goal_y", type=float, default=80.0)
    parser.add_argument("--goal.theta", dest="goal_theta", type=float, default=HALF_PI)
    parser.add_argument("--goal.kappa", dest="goal_kappa", type=float, default=0.0)
    parser.add_argument("--goal.pos_tol", dest="goal_pos_tol", type=float, default=0.5)
    parser.add_argument("--goal.theta_tol", dest="goal_theta_tol", type=float, default=TWELFTH_PI)
    # allows for arbitrary overrides of config values via dotted paths, e.g. --set weights.reverse_penalty=2.0
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="group.param=value",
        help="Override planner config values with dotted keys (example: --set weights.reverse_penalty=2.0)",
    )
    return parser



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


def default_planner_config_dict() -> Dict[str, Any]:
    return asdict(
        PlannerConfig(
            # grid=GridSpec(resolution=0.5, theta_bins=36, origin_xy=(0.0, 0.0), kappa_bins=11),
            vehicle=VehicleParams(),
        )
    )


def deep_merge(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def apply_dotted_override(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    node = target
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def load_yaml_dict(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config root must be a mapping/object")
    return data


def _coerce_value(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        lowered = raw_value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return raw_value


def _parse_overrides(set_values: list[str]) -> list[DotPathOverride]:
    overrides: list[DotPathOverride] = []
    for item in set_values:
        if "=" not in item:
            raise ValueError(f"Invalid --set override '{item}'; expected path=value")
        path, raw_value = item.split("=", 1)
        overrides.append(DotPathOverride(path=path.strip(), value=_coerce_value(raw_value.strip())))
    return overrides


def parse_planning_inputs(argv: Optional[Sequence[str]] = None) -> PlanningInputs:
    parser = build_arg_parser()
    parsed = parser.parse_args(argv)
    cli = CLIOverridesModel(
        backend=parsed.backend,
        max_expansions=parsed.max_expansions,
        config_file=parsed.config_file,
        world_cfg_path=parsed.world_cfg_path,
        start_x=parsed.start_x,
        start_y=parsed.start_y,
        start_theta=parsed.start_theta,
        start_kappa=parsed.start_kappa,
        goal_x=parsed.goal_x,
        goal_y=parsed.goal_y,
        goal_theta=parsed.goal_theta,
        goal_kappa=parsed.goal_kappa,
        goal_pos_tol=parsed.goal_pos_tol,
        goal_theta_tol=parsed.goal_theta_tol,
        set_values=parsed.set,
    )
    planner_dict = default_planner_config_dict()
    yaml_blob: Dict[str, Any] = {}
    if cli.config_file:
        yaml_blob = load_yaml_dict(cli.config_file)
        if "planner" in yaml_blob:
            deep_merge(planner_dict, yaml_blob["planner"])
        else:
            deep_merge(planner_dict, yaml_blob)
    for override in _parse_overrides(cli.set_values):
        apply_dotted_override(planner_dict, override.path, override.value)
    run_payload: Dict[str, Any] = {
        "backend": cli.backend,
        "max_expansions": cli.max_expansions,
        "world_cfg_path": cli.world_cfg_path,
        "planner": planner_dict,
        "start": {
            "x": cli.start_x,
            "y": cli.start_y,
            "theta": cli.start_theta,
            "kappa": cli.start_kappa,
        },
        "goal": {
            "pose": {
                "x": cli.goal_x,
                "y": cli.goal_y,
                "theta": cli.goal_theta,
                "kappa": cli.goal_kappa,
            },
            "pos_tol": cli.goal_pos_tol,
            "theta_tol": cli.goal_theta_tol,
        },
    }
    planner_snapshot = copy.deepcopy(planner_dict)
    deep_merge(run_payload, yaml_blob)
    run_payload["planner"] = planner_snapshot
    run_model = PlanningRunConfigModel.model_validate(run_payload)
    return PlanningInputs(
        backend=run_model.backend,
        max_expansions=run_model.max_expansions,
        world_cfg_path=run_model.world_cfg_path,
        planner_config=run_model.planner,
        start=run_model.start,
        goal=run_model.goal,
    )
