# dolgov_cbmp/telemetry/logging.py

from dataclasses import asdict, dataclass, field
from pathlib import Path
from enum import Enum
from queue import Empty, Full, Queue
import threading
from typing import Callable, Optional, Protocol, Sequence, Union
# local imports - planner structs
from dolgov_cbmp.structs import PlannerStats, PlannerTick
from dolgov_cbmp.telemetry import JsonlStatsSink, JsonlTickSink, McapTickSink


class LogRecordType(str, Enum):
    TICK = "tick"
    STATS = "stats"


@dataclass(frozen=True)
class PlannerLogRecord:
    type: LogRecordType
    payload: dict

    @staticmethod
    def from_tick(tick: PlannerTick) -> "PlannerLogRecord":
        return PlannerLogRecord(type=LogRecordType.TICK, payload=asdict(tick))

    @staticmethod
    def from_stats(stats: PlannerStats) -> "PlannerLogRecord":
        return PlannerLogRecord(type=LogRecordType.STATS, payload=asdict(stats))


class TickSink(Protocol):
    def __call__(self, tick: PlannerTick) -> None: ...

class StatsSink(Protocol):
    def __call__(self, stats: PlannerStats) -> None: ...


class PlannerLogDispatcher:
    """ producer-consumer patterned dispatcher for planner telemetry with subscriber fan-out and back-pressure handling """
    def __init__(
        self,
        tick_sinks: Optional[Sequence[TickSink]] = None,
        stats_sinks: Optional[Sequence[StatsSink]] = None,
        max_queue_size: int = 2048,
        drop_when_full: bool = True,
    ):
        self.tick_sinks = list(tick_sinks or [])
        self.stats_sinks = list(stats_sinks or [])
        self.drop_when_full = bool(drop_when_full)
        self._queue: Queue[Optional[PlannerLogRecord]] = Queue(maxsize=max(1, int(max_queue_size)))
        self._stopped = threading.Event()
        self._worker = threading.Thread(target=self._run, name="planner-log-dispatcher", daemon=True)
        self._worker.start()

    def subscribe_tick_sink(self, sink: TickSink) -> None:
        self.tick_sinks.append(sink)

    def subscribe_stats_sink(self, sink: StatsSink) -> None:
        self.stats_sinks.append(sink)

    def publish_tick(self, tick: PlannerTick) -> None:
        self._publish(PlannerLogRecord.from_tick(tick))

    def publish_stats(self, stats: PlannerStats) -> None:
        self._publish(PlannerLogRecord.from_stats(stats))

    # TODO: kind of unnecessary but made things a bit easier in the planner - might remove later
    @property
    def tick_callback(self) -> Callable[[PlannerTick], None]:
        return self.publish_tick

    @property
    def stats_callback(self) -> Callable[[PlannerStats], None]:
        return self.publish_stats


    def _publish(self, record: PlannerLogRecord) -> None:
        if self._stopped.is_set():
            return
        if self.drop_when_full:
            try:
                self._queue.put_nowait(record)
            except Full:
                return
        else:
            self._queue.put(record)


    def _run(self) -> None:
        while True:
            try:
                rec = self._queue.get(timeout=0.1)
            except Empty:
                if self._stopped.is_set():
                    return
                continue
            if rec is None:
                self._queue.task_done()
                return
            try:
                if rec.type == LogRecordType.TICK:
                    tick = PlannerTick(**rec.payload)
                    for sink in self.tick_sinks:
                        sink(tick)
                elif rec.type == LogRecordType.STATS:
                    stats = PlannerStats(**rec.payload)
                    for sink in self.stats_sinks:
                        sink(stats)
            finally:
                self._queue.task_done()


    def close(self, timeout_s: float = 5.0) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._queue.put(None)
        self._worker.join(timeout=max(0.0, float(timeout_s)))
        for sink in [*self.tick_sinks, *self.stats_sinks]:
            close_fn = getattr(sink, "close", None)
            if callable(close_fn):
                close_fn()

    def __enter__(self) -> "PlannerLogDispatcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()



class PlannerTelemetrySession:
    """ high-level telemetry session with threaded producer-consumer dispatch and support for multiple sinks and back-pressure handling """
    def __init__(
        self,
        tick_jsonl_path: Optional[Union[str, Path]] = None,
        stats_jsonl_path: Optional[Union[str, Path]] = None,
        tick_mcap_path: Optional[Union[str, Path]] = None,
        tick_mcap_topic: str = "/planner/markers",
        max_queue_size: int = 2048,
        drop_when_full: bool = True,
        _dispatcher: PlannerLogDispatcher = field(init=False),
    ):
        self.tick_jsonl_path = tick_jsonl_path
        self.stats_jsonl_path = stats_jsonl_path
        self.tick_mcap_path = tick_mcap_path
        self.tick_mcap_topic = tick_mcap_topic
        self.max_queue_size = max_queue_size
        self.drop_when_full = drop_when_full
        self._dispatcher = _dispatcher
        tick_sinks = []
        stats_sinks = []
        if self.tick_jsonl_path is not None:
            tick_sinks.append(JsonlTickSink(self.tick_jsonl_path))
        if self.tick_mcap_path is not None:
            tick_sinks.append(McapTickSink(self.tick_mcap_path, topic=self.tick_mcap_topic))
        if self.stats_jsonl_path is not None:
            stats_sinks.append(JsonlStatsSink(self.stats_jsonl_path))
        self._dispatcher = PlannerLogDispatcher(
            tick_sinks=tick_sinks,
            stats_sinks=stats_sinks,
            max_queue_size=self.max_queue_size,
            drop_when_full=self.drop_when_full,
        )


    # TODO: might make these proper class methods later for transparency's sake
    @property
    def tick_callback(self):
        return self._dispatcher.tick_callback

    @property
    def stats_callback(self):
        return self._dispatcher.stats_callback


    def subscribe_tick_sink(self, sink) -> None:
        self._dispatcher.subscribe_tick_sink(sink)

    def subscribe_stats_sink(self, sink) -> None:
        self._dispatcher.subscribe_stats_sink(sink)

    def publish_tick(self, tick: PlannerTick) -> None:
        self._dispatcher.publish_tick(tick)

    def publish_stats(self, stats: PlannerStats) -> None:
        self._dispatcher.publish_stats(stats)

    def close(self) -> None:
        self._dispatcher.close()

    def __enter__(self) -> "PlannerTelemetrySession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()