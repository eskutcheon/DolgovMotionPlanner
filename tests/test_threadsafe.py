
from concurrent.futures import ThreadPoolExecutor

from planners import planner_factory

#^ WARNING: this test may expose shared-state bugs in the planner implementation
#^ WARNING: also one of the slowest tests in the suite
    # BUT it's not a final implementation and is mainly meant for testing that there aren't barriers to future concurrent implementations
def test_planner_can_be_called_concurrently(empty_grid, planner_config, start_pose, goal_spec):
    # test for shared-state bugs (even though Python backend is mostly serialized by the GIL).
    planner = planner_factory(empty_grid, planner_config, backend="python")

    def run_once():
        path, stats = planner.plan(start_pose, goal_spec, max_expansions=50_000)
        return len(path), stats.expanded

    with ThreadPoolExecutor(max_workers=2) as ex:
        a = ex.submit(run_once)
        b = ex.submit(run_once)
        la, ea = a.result()
        lb, eb = b.result()
    assert la > 1 and lb > 1, "One of the concurrent runs failed to find a path"
    assert ea > 0 and eb > 0,  "One of the concurrent runs did not expand any nodes"

