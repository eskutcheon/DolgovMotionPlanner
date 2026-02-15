# src/telemetry/telemetry.py
from pathlib import Path
import json
from typing import Dict, Any, Iterable
# local imports
from src.structs import PlannerTick, Pose

try:  # optional dependency
    from mcap.writer import Writer as McapWriter
except Exception:  # pragma: no cover
    McapWriter = None  # type: ignore[assignment]



def _pose_to_marker_point(pose: Pose) -> Dict[str, float]:
    return {"x": float(pose.x), "y": float(pose.y), "z": 0.0}


def tick_to_markers(tick: PlannerTick) -> Dict[str, Any]:
    """ convert a planner tick into marker-like JSON payloads for visualization """
    return {
        "tick": {
            "iteration": tick.iteration,
            "time_s": tick.time_s,
            "expanded": tick.expanded,
            "pushed": tick.pushed,
            "open_size": tick.open_size,
            "collision_checks": tick.collision_checks,
            "failed_rollouts": tick.failed_rollouts,
            "best_f": tick.best_f,
            "best_g": tick.best_g,
        },
        "markers": [
            {
                "namespace": "planner/current",
                "type": "SPHERE",
                "pose": _pose_to_marker_point(tick.pose),
                "scale": {"x": 0.2, "y": 0.2, "z": 0.2},
                "color": {"r": 0.2, "g": 0.6, "b": 1.0, "a": 1.0},
            },
            {
                "namespace": "planner/best",
                "type": "SPHERE",
                "pose": _pose_to_marker_point(tick.best_pose),
                "scale": {"x": 0.25, "y": 0.25, "z": 0.25},
                "color": {"r": 0.0, "g": 1.0, "b": 0.2, "a": 1.0},
            },
            {
                "namespace": "planner/trajectory",
                "type": "LINE_STRIP",
                "points": [_pose_to_marker_point(p) for p in tick.trajectory],
                "scale": {"x": 0.08},
                "color": {"r": 1.0, "g": 0.65, "b": 0.1, "a": 0.9},
            },
            {
                "namespace": "planner/explored",
                "type": "POINTS",
                "points": [_pose_to_marker_point(p) for p in tick.explored_poses],
                "scale": {"x": 0.05, "y": 0.05},
                "color": {"r": 0.6, "g": 0.8, "b": 1.0, "a": 0.75},
            },
            {
                "namespace": "planner/collisions",
                "type": "POINTS",
                "points": [_pose_to_marker_point(p) for p in tick.collision_poses],
                "scale": {"x": 0.07, "y": 0.07},
                "color": {"r": 1.0, "g": 0.2, "b": 0.2, "a": 0.95},
            },
            {
                "namespace": "planner/pruned_trajectories",
                "type": "LINE_LIST",
                "segments": [
                    [_pose_to_marker_point(path[i]), _pose_to_marker_point(path[i + 1])]
                    for path in tick.pruned_trajectories for i in range(max(0, len(path) - 1))
                ],
                "scale": {"x": 0.04},
                "color": {"r": 0.35, "g": 0.55, "b": 1.0, "a": 0.55},
            },
            {
                "namespace": "planner/analytic_shot",
                "type": "LINE_STRIP",
                "points": [_pose_to_marker_point(p) for p in tick.analytic_shot],
                "scale": {"x": 0.08},
                "color": {"r": 0.9, "g": 0.3, "b": 0.95, "a": 0.9},
            },
        ],
    }


class JsonlTickSink:
    """ simple sink for offline debugging and conversion pipelines """
    def __init__(self, out_path: str | Path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, tick: PlannerTick) -> None:
        payload = tick_to_markers(tick)
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


class McapTickSink:
    """ write planner tick marker payloads directly to MCAP as JSON messages """
    def __init__(self, out_path: str | Path, topic: str = "/planner/markers"):
        if McapWriter is None:
            raise ImportError("mcap is not installed. Install with `pip install mcap`.")
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_path.open("wb")
        self._writer = McapWriter(self._fh)
        #? NOTE: important to start the writer or the generated MCAP file will be unreadable/corrupted (missing header, index, etc)
        self._writer.start()
        with open(Path(__file__).parent / "schemas.json", "r", encoding="utf-8") as f:
            schema = json.load(f)["RAW_MCAP"]
        self._schema_id = self._writer.register_schema(
            name="dolgov.planner_markers",
            encoding="jsonschema",
            data=json.dumps(schema).encode("utf-8"),
        )
        self._channel_id = self._writer.register_channel(topic=topic, message_encoding="json", schema_id=self._schema_id,)

    def __call__(self, tick: PlannerTick) -> None:
        payload = tick_to_markers(tick)
        t_ns = int(tick.time_s * 1e9)
        self._writer.add_message(
            channel_id=self._channel_id,
            log_time=t_ns,
            publish_time=t_ns,
            data=json.dumps(payload).encode("utf-8"),
        )

    def close(self) -> None:
        self._writer.finish()
        self._fh.close()


def write_ticks_jsonl(ticks: Iterable[PlannerTick], out_path: str | Path) -> None:
    sink = JsonlTickSink(out_path)
    for tick in ticks:
        sink(tick)


def write_ticks_mcap(ticks: Iterable[PlannerTick], out_path: str | Path, topic: str = "/planner/markers") -> None:
    sink = McapTickSink(out_path, topic=topic)
    try:
        for tick in ticks:
            sink(tick)
    finally:
        sink.close()



