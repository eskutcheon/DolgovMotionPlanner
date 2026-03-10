# tests/test_telemetry.py
import json
from dataclasses import replace
# from tracemalloc import start
from uuid import uuid4
from pathlib import Path
import pytest
import math
from typing import List, Tuple, Any
# local imports
from dolgov_cbmp.structs import PlannerTick, PlannerStats, Pose, GoalSpec
from dolgov_cbmp.settings import PlannerConfig
from dolgov_cbmp.utils import goal_reached



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
    from dolgov_cbmp.telemetry import tick_to_markers
    payload = tick_to_markers(sample_tick)
    assert payload["tick"]["iteration"] == 10
    assert len(payload["markers"]) == 7
    assert payload["markers"][2]["namespace"] == "planner/trajectory"
    assert len(payload["markers"][2]["points"]) == 2
    assert payload["markers"][4]["namespace"] == "planner/collisions"
    assert len(payload["markers"][4]["points"]) == 1
    assert payload["markers"][5]["namespace"] == "planner/pruned_trajectories"
    assert payload["markers"][6]["namespace"] == "planner/analytic_shot"


def test_tick_write_mcap(mcap_out_dir: Path, sample_tick: PlannerTick):
    pytest.importorskip("mcap")
    from dolgov_cbmp.telemetry import write_ticks_mcap
    pytest.importorskip("mcap")
    out = mcap_out_dir / "ticks.mcap"
    write_ticks_mcap([sample_tick], out)
    assert out.exists()
    assert out.stat().st_size > 0
    # assert that magic header is present (indicates it's a valid MCAP file and not just random bytes)
    assert out.read_bytes()[:8] == bytes([0x89, 0x4D, 0x43, 0x41, 0x50, 0x30, 0x0D, 0x0A])


def test_tick_write_mcap_roundtrip_json(mcap_out_dir: Path, sample_tick):
    pytest.importorskip("mcap")
    from dolgov_cbmp.telemetry import write_ticks_mcap
    # spoof some ticks with different iterations for testing
    ticks = [
        sample_tick,
        replace(sample_tick, iteration=11, time_s=0.30, expanded=11),
        replace(sample_tick, iteration=12, time_s=0.35, expanded=12),
    ]
    out_path = mcap_out_dir / f"raw_{uuid4().hex}.mcap"
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


def test_tick_write_mcap_ns_time(mcap_out_dir: Path, sample_tick):
    pytest.importorskip("mcap")
    from dolgov_cbmp.telemetry import write_ticks_mcap
    ticks = [
        replace(sample_tick, time_s=0.25),
        replace(sample_tick, time_s=1.00),
    ]
    out_path: Path = mcap_out_dir / f"raw_time_{uuid4().hex}.mcap"
    write_ticks_mcap(ticks, out_path)
    records = _read_mcap_messages(out_path)
    assert len(records) == len(ticks)
    # log_time/publish_time should be int(time_s * 1e9)
    for (_, _, msg), t in zip(records, ticks):
        expected_ns = int(t.time_s * 1e9)
        assert msg.log_time == expected_ns
        assert msg.publish_time == expected_ns


def test_tick_write_jsonl_lines(mcap_out_dir: Path, sample_tick):
    from dolgov_cbmp.telemetry import write_ticks_jsonl
    ticks = [
        sample_tick,
        replace(sample_tick, iteration=11, time_s=0.30, expanded=11),
    ]
    out_path: Path = mcap_out_dir / f"ticks_{uuid4().hex}.jsonl"
    write_ticks_jsonl(ticks, out_path)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    # ensure one line per tick and that each line is valid JSON with expected keys
    assert len(lines) == len(ticks)
    row0 = json.loads(lines[0])
    assert "tick" in row0
    assert "markers" in row0


def test_tick_async_dispatcher_fanout(sample_tick: PlannerTick):
    from dolgov_cbmp.telemetry import PlannerLogDispatcher
    tick_events = []
    stats_events = []

    def on_tick(t: PlannerTick):
        tick_events.append(t.iteration)

    def on_stats(s: PlannerStats):
        stats_events.append(s.expanded)

    with PlannerLogDispatcher(tick_sinks=[on_tick], stats_sinks=[on_stats], drop_when_full=False) as dispatcher:
        dispatcher.publish_tick(sample_tick)
        dispatcher.publish_stats(PlannerStats(expanded=sample_tick.expanded))
    assert tick_events == [sample_tick.iteration]
    assert stats_events == [sample_tick.expanded]


def test_tick_planner_telemetry_session_writes_live_jsonl(mcap_out_dir: Path, empty_grid, planner_config):
    from dolgov_cbmp.planners import planner_factory
    from dolgov_cbmp.telemetry import PlannerTelemetrySession
    planner = planner_factory(empty_grid, planner_config, backend="python")
    start = Pose(10.0, 10.0, 0.0)
    goal = GoalSpec(Pose(50.0, 50.0, 0.0), pos_tol=2.0, theta_tol=math.radians(30.0))
    tick_path = mcap_out_dir / f"live_ticks_{uuid4().hex}.jsonl"
    stats_path = mcap_out_dir / f"live_stats_{uuid4().hex}.jsonl"
    with PlannerTelemetrySession(tick_jsonl_path=tick_path, stats_jsonl_path=stats_path, drop_when_full=False) as session:
        path, stats = planner.plan(
            start, goal, max_expansions=20_000,
            tick_stride=100, stats_stride=250,
            telemetry_session=session,
            close_telemetry_session=False,
        )
    assert stats.expanded > 0
    assert path
    tick_lines = tick_path.read_text(encoding="utf-8").splitlines()
    stats_lines = stats_path.read_text(encoding="utf-8").splitlines()
    assert len(tick_lines) > 2
    assert len(stats_lines) >= 1
    last_stats = json.loads(stats_lines[-1])
    assert last_stats["expanded"] == stats.expanded



def test_tick_foxglove_mcap_topics(mcap_out_dir, sample_tick):
    """ test that the Foxglove MCAP writer publishes to expected topics, with expected encodings and payload structure """
    pytest.importorskip("foxglove")
    from dolgov_cbmp.telemetry import write_ticks_mcap_foxglove as write_fg_mcap
    ticks = [
        sample_tick,
        replace(sample_tick, iteration=11, time_s=0.30, expanded=11),
    ]
    out_path = mcap_out_dir / f"fg_{uuid4().hex}.mcap"
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
def test_tick_planner_logs_and_writes_mcap(mcap_out_dir, empty_grid, planner_config):
    from dolgov_cbmp.planners import planner_factory
    # tick_callback is only supported in the python backend in your code today
    planner = planner_factory(empty_grid, planner_config, backend="python")
    # TODO: might replace with conftest fixtures later
    start = Pose(10.0, 10.0, 0.0)
    goal = GoalSpec(Pose(50.0, 50.0, 0.0), pos_tol=2.0, theta_tol=math.radians(30.0))
    from dolgov_cbmp.telemetry import PlannerTelemetrySession
    tick_jsonl: Path = mcap_out_dir / f"sim_ticks_{uuid4().hex}.jsonl"
    stats_jsonl: Path = mcap_out_dir / f"sim_stats_{uuid4().hex}.jsonl"
    raw_path = mcap_out_dir / f"sim_raw_{uuid4().hex}.mcap"
    pytest.importorskip("mcap")
    with PlannerTelemetrySession(
        tick_jsonl_path=tick_jsonl,
        stats_jsonl_path=stats_jsonl,
        tick_mcap_path=raw_path,
        drop_when_full=False,
    ) as telemetry:
        path, stats = planner.plan(
            start,
            goal,
            max_expansions=20_000,
            tick_stride=100, # keep this moderate so test isn't too slow, but likely to emit multiple ticks
            stats_stride=250,
            telemetry_session=telemetry,
            close_telemetry_session=False,
        )
    # if not path:
    #     pytest.skip("Planner failed to find a path in the maze, so skipping the rest of the MCAP writing/logging test")
    # sanity check that we got some ticks with expected content
    assert stats.expanded > 0
    # assert len(ticks) > 2  # "several" ticks
    assert path
    assert len(tick_jsonl.read_text(encoding="utf-8").splitlines()) > 2
    assert len(stats_jsonl.read_text(encoding="utf-8").splitlines()) >= 1
    pytest.importorskip("mcap")
    from mcap.reader import make_reader
    #& UPDATE: no longer writes MCAP directly and instead relies on the PlannerTelemetrySession to dispatch to the McapTickSink
    #   so we just check that the file is a valid MCAP with expected topic and at least one message
    # from dolgov_cbmp.telemetry import write_ticks_mcap as write_raw_mcap
    # raw_path = mcap_out_dir / f"sim_raw_{uuid4().hex}.mcap"
    # write_raw_mcap(ticks, raw_path)
    # pytest.importorskip("foxglove")
    # from dolgov_cbmp.telemetry import write_ticks_mcap_foxglove
    # fg_path = mcap_out_dir / f"sim_fg_{uuid4().hex}.mcap"
    # write_ticks_mcap_foxglove(ticks, fg_path)
    # # check that both files are readable, i.e. have the magic header and at least one message
    with open(raw_path, "rb") as f:
        reader = make_reader(f)
        n = sum(1 for _ in reader.iter_messages())
    assert n > 0


@pytest.mark.slow
def test_tick_planner_logs_mazes_and_writes_mcap(
    mcap_out_dir: Path,
    # maze_grid_and_poses: Tuple[Any, List[float], List[float]],
    maze_world_model,
    planner_config: PlannerConfig,
    start_pose: Pose,
    goal_spec: GoalSpec,
    require_success: bool = True,
):
    from dolgov_cbmp.planners import planner_factory
    # grid, start, goal = maze_grid_and_poses
    world = maze_world_model
    grid = world.occupancy_grid
    # update start and goal poses with those from the maze file (necessary since Pose dataclasses are frozen)
    s_pose = Pose(world.start.x, world.start.y, start_pose.theta, start_pose.kappa)
    g_pose = Pose(world.goal.pose.x, world.goal.pose.y, goal_spec.pose.theta, goal_spec.pose.kappa)
    # create new GoalSpec with updated goal pose
    g_spec = GoalSpec(g_pose, goal_spec.pos_tol, goal_spec.theta_tol) #, goal_spec.kappa_tol)
    planner = planner_factory(grid, planner_config, backend="python")
    from dolgov_cbmp.telemetry import PlannerTelemetrySession
    fg_path = mcap_out_dir / f"full_sim_fg_{uuid4().hex}.mcap"
    # path, stats = planner.plan(s_pose, g_spec, max_expansions=200_000, tick_callback=on_tick)
    with PlannerTelemetrySession(drop_when_full=False) as telemetry:
        pytest.importorskip("foxglove")
        from dolgov_cbmp.telemetry import FoxgloveTickSink
        telemetry.subscribe_tick_sink(FoxgloveTickSink(fg_path, occ_grid=grid, start_pose=s_pose, goal=g_spec, vehicle=planner_config.vehicle))
        path, stats = planner.plan(
            s_pose,
            g_spec,
            max_expansions=200_000,
            tick_stride=100,
            telemetry_session=telemetry,
            close_telemetry_session=False,
        )
    # sanity check that we got some ticks with expected content
    assert stats.expanded > 0
    # assert len(ticks) > 2
    if require_success:
        assert path, "Planner failed to find a path in the maze"
        assert goal_reached(
                path[-1].as_tuple(), g_spec.pose.as_tuple(), g_spec.pos_tol, g_spec.theta_tol
            ), "Final pose does not reach the goal tolerances"
    elif not path:
        pytest.skip("Planner failed to find a path in the maze, so skipping remaining testing of tick logging and MCAP writing")
    grid.view_grid(path)
    pytest.importorskip("mcap")
    from mcap.reader import make_reader
    #& UPDATE: no longer writes MCAP directly and instead relies on the PlannerTelemetrySession to dispatch to the McapTickSink
    # pytest.importorskip("foxglove")
    # from dolgov_cbmp.telemetry import write_ticks_mcap_foxglove
    # fg_path = mcap_out_dir / f"full_sim_fg_{uuid4().hex}.mcap"
    # # write_ticks_mcap_foxglove(ticks, fg_path, occ_grid=grid, start_pose=s_pose, goal=g_spec, vehicle=planner_config.vehicle)
    # write_ticks_mcap_foxglove(ticks, fg_path, world=world, vehicle=planner_config.vehicle)
    # # check that both files are readable, i.e. have the magic header and at least one message
    with open(fg_path, "rb") as f:
        reader = make_reader(f)
        n = sum(1 for _ in reader.iter_messages())
    assert n > 0