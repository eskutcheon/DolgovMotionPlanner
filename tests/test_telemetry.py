import json
from dataclasses import replace
# from tracemalloc import start
from uuid import uuid4
from pathlib import Path
import pytest
import math
from typing import List, Tuple, Any
# local imports
from src.structs import PlannerTick, Pose, GoalSpec, PlannerConfig
from src.utils import goal_reached



def _read_mcap_messages(path) -> List[Tuple[Any, Any, Any]]:
    """ read all messages from an MCAP file and return a list of (schema, channel, message) tuples """
    pytest.importorskip("mcap")
    from mcap.reader import make_reader
    out = []
    with open(path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            out.append((schema, channel, message))
    return out


def _read_mcap_by_topic(path):
    """ for Foxglove MCAP schemas - read from an MCAP file and return a dict (topic -> list) of (schema, channel, message) tuples """
    pytest.importorskip("mcap")
    from mcap.reader import make_reader
    by_topic = {}
    with open(path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            by_topic.setdefault(channel.topic, []).append((schema, channel, message))
    return by_topic


#? NOTE: trying to use the common prefix "test_tick_" for all telemetry tests to make it easy to search and run `pytest -k test_tick_`


def test_tick_to_markers_includes_trajectory_and_collision_points(sample_tick: PlannerTick):
    from src.telemetry import tick_to_markers
    payload = tick_to_markers(sample_tick)
    assert payload["tick"]["iteration"] == 10
    assert len(payload["markers"]) == 7
    assert payload["markers"][2]["namespace"] == "planner/trajectory"
    assert len(payload["markers"][2]["points"]) == 2
    assert payload["markers"][4]["namespace"] == "planner/collisions"
    assert len(payload["markers"][4]["points"]) == 1
    assert payload["markers"][5]["namespace"] == "planner/pruned_trajectories"
    assert payload["markers"][6]["namespace"] == "planner/analytic_shot"


def test_tick_write_mcap(tmp_path: Path, sample_tick: PlannerTick):
    pytest.importorskip("mcap")
    from src.telemetry import write_ticks_mcap
    pytest.importorskip("mcap")
    out = tmp_path / "ticks.mcap"
    write_ticks_mcap([sample_tick], out)
    assert out.exists()
    assert out.stat().st_size > 0
    # assert that magic header is present (indicates it's a valid MCAP file and not just random bytes)
    assert out.read_bytes()[:8] == bytes([0x89, 0x4D, 0x43, 0x41, 0x50, 0x30, 0x0D, 0x0A])


def test_tick_write_mcap_roundtrip_json(tmp_path: Path, sample_tick):
    pytest.importorskip("mcap")
    from src.telemetry import write_ticks_mcap
    # spoof some ticks with different iterations for testing
    ticks = [
        sample_tick,
        replace(sample_tick, iteration=11, time_s=0.30, expanded=11),
        replace(sample_tick, iteration=12, time_s=0.35, expanded=12),
    ]
    out_path = tmp_path / f"raw_{uuid4().hex}.mcap"
    write_ticks_mcap(ticks, out_path)
    records = _read_mcap_messages(out_path)
    # ensure at least one message per tick and that all messages are on the expected topic
    assert len(records) == len(ticks)
    topics = {ch.topic for _, ch, _ in records}
    assert topics == {"/planner/markers"}
    # payload should decode as JSON and include "markers" and "tick"
    for (_, ch, msg), t in zip(records, ticks):
        assert ch.message_encoding == "json"
        payload = json.loads(msg.data.decode("utf-8"))
        assert "tick" in payload
        assert "markers" in payload
        assert payload["tick"]["iteration"] == t.iteration


def test_tick_write_mcap_ns_time(tmp_path: Path, sample_tick):
    pytest.importorskip("mcap")
    from src.telemetry import write_ticks_mcap
    ticks = [
        replace(sample_tick, time_s=0.25),
        replace(sample_tick, time_s=1.00),
    ]
    out_path: Path = tmp_path / f"raw_time_{uuid4().hex}.mcap"
    write_ticks_mcap(ticks, out_path)
    records = _read_mcap_messages(out_path)
    assert len(records) == len(ticks)
    # log_time/publish_time should be int(time_s * 1e9)
    for (_, _, msg), t in zip(records, ticks):
        expected_ns = int(t.time_s * 1e9)
        assert msg.log_time == expected_ns
        assert msg.publish_time == expected_ns


def test_tick_write_jsonl_lines(tmp_path: Path, sample_tick):
    from src.telemetry import write_ticks_jsonl
    ticks = [
        sample_tick,
        replace(sample_tick, iteration=11, time_s=0.30, expanded=11),
    ]
    out_path: Path = tmp_path / f"ticks_{uuid4().hex}.jsonl"
    write_ticks_jsonl(ticks, out_path)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    # ensure one line per tick and that each line is valid JSON with expected keys
    assert len(lines) == len(ticks)
    row0 = json.loads(lines[0])
    assert "tick" in row0
    assert "markers" in row0


def test_tick_foxglove_mcap_topics(tmp_path, sample_tick):
    """ test that the Foxglove MCAP writer publishes to expected topics, with expected encodings and payload structure """
    pytest.importorskip("foxglove")
    from src.telemetry import write_ticks_mcap_foxglove as write_fg_mcap
    ticks = [
        sample_tick,
        replace(sample_tick, iteration=11, time_s=0.30, expanded=11),
    ]
    out_path = tmp_path / f"fg_{uuid4().hex}.mcap"
    write_fg_mcap(ticks, out_path)
    by_topic = _read_mcap_by_topic(out_path)
    # emsure presence of topics the Foxglove writer logs: scene + two pointclouds + raw tick-as-json
    assert "/planner/scene" in by_topic
    # TODO: check these against a schema built from the VizConfig objects later
    # assert "/planner/explored" in by_topic
    assert "/planner/collisions" in by_topic
    assert "/planner/tick" in by_topic
    # expect at least one message per tick on these streams
    assert len(by_topic["/planner/scene"]) - 1 == len(ticks) # need to subtract 1 because the writer emits an initial scene message before any ticks are processed
    assert len(by_topic["/planner/tick"]) == len(ticks)
    # /planner/tick is JSON so we make sure it decodes and has an iteration
    for schema, ch, msg in by_topic["/planner/tick"]:
        assert ch.message_encoding == "json"
        tick_payload = json.loads(msg.data.decode("utf-8"))
        assert "iteration" in tick_payload


@pytest.mark.slow
def test_tick_planner_logs_and_writes_mcap(tmp_path, empty_grid, planner_config):
    from src.planners import planner_factory
    # tick_callback is only supported in the python backend in your code today
    planner = planner_factory(empty_grid, planner_config, backend="python")
    # TODO: might replace with conftest fixtures later
    start = Pose(10.0, 10.0, 0.0)
    goal = GoalSpec(Pose(50.0, 50.0, 0.0), pos_tol=2.0, theta_tol=math.radians(30.0))
    ticks = []

    def on_tick(t):
        ticks.append(t)

    # keep this moderate so test isn't too slow, but likely to emit multiple ticks
    path, stats = planner.plan(
        start,
        goal,
        max_expansions=20_000,
        tick_callback=on_tick,
        tick_stride=250,
    )
    # sanity check that we got some ticks with expected content
    assert stats.expanded > 0
    assert len(ticks) > 2  # "several" ticks
    pytest.importorskip("mcap")
    from mcap.reader import make_reader
    from src.telemetry import write_ticks_mcap as write_raw_mcap
    raw_path = tmp_path / f"sim_raw_{uuid4().hex}.mcap"
    write_raw_mcap(ticks, raw_path)
    pytest.importorskip("foxglove")
    from src.telemetry import write_ticks_mcap_foxglove
    fg_path = tmp_path / f"sim_fg_{uuid4().hex}.mcap"
    write_ticks_mcap_foxglove(ticks, fg_path)
    # check that both files are readable, i.e. have the magic header and at least one message
    for p in (raw_path, fg_path):
        with open(p, "rb") as f:
            reader = make_reader(f)
            n = sum(1 for _ in reader.iter_messages())
        assert n > 0


@pytest.mark.slow
def test_tick_planner_logs_mazes_and_writes_mcap(tmp_path, maze_grid_and_poses, planner_config: PlannerConfig, start_pose: Pose, goal_spec: GoalSpec):
    from src.planners import planner_factory
    grid, start, goal = maze_grid_and_poses
    # update start and goal poses with those from the maze file (necessary since Pose dataclasses are frozen)
    s_pose = Pose(start[0], start[1], start_pose.theta, start_pose.kappa)
    g_pose = Pose(goal[0], goal[1], goal_spec.pose.theta, goal_spec.pose.kappa)
    # create new GoalSpec with updated goal pose
    g_spec = GoalSpec(g_pose, goal_spec.pos_tol, goal_spec.theta_tol, goal_spec.kappa_tol)
    planner = planner_factory(grid, planner_config, backend="python")
    ticks = []

    def on_tick(t):
        ticks.append(t)

    path, stats = planner.plan(
        s_pose,
        g_spec,
        max_expansions=200_000,
        tick_callback=on_tick,
    )
    # sanity check that we got some ticks with expected content
    assert stats.expanded > 0
    assert len(ticks) > 2
    assert goal_reached(
            path[-1].as_tuple(), g_spec.pose.as_tuple(), g_spec.pos_tol, g_spec.theta_tol
        ), "Final pose does not reach the goal tolerances"
    grid.view_grid(path)
    pytest.importorskip("mcap")
    from mcap.reader import make_reader
    pytest.importorskip("foxglove")
    from src.telemetry import write_ticks_mcap_foxglove
    fg_path = tmp_path / f"full_sim_fg_{uuid4().hex}.mcap"
    write_ticks_mcap_foxglove(ticks, fg_path, occ_grid=grid, start_pose=s_pose, goal=g_spec, vehicle=planner_config.vehicle)
    # check that both files are readable, i.e. have the magic header and at least one message
    for p in (fg_path, fg_path):
        with open(p, "rb") as f:
            reader = make_reader(f)
            n = sum(1 for _ in reader.iter_messages())
        assert n > 0