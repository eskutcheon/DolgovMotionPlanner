
import math
import numpy as np
import pytest
hybrid_core = pytest.importorskip("hybrid_core", reason="C++ extension not built")
pytestmark = pytest.mark.cpp


@pytest.mark.cpp
def test_cpp_extension_import_and_expand_primitives():
    try:
        import hybrid_core  # type: ignore
    except Exception as e:
        pytest.skip(f"C++ extension not built/importable: {e}")
    if not hasattr(hybrid_core, "expand_primitives"):
        pytest.skip("hybrid_core does not expose expand_primitives in this build")

    # Empty map
    occ = np.zeros((30, 30), dtype=np.uint8)
    steer = np.asarray([-0.2, 0.0, 0.2], dtype=np.float64)
    out = hybrid_core.expand_primitives(
        5.0, 5.0, 0.0,
        1,
        steer,
        True,
        1.0, 4,
        2.7,
        occ,
        0.0, 0.0, 1.0,
        72,
    )

    assert "x" in out and "y" in out and "th" in out
    x = np.asarray(out["x"])
    y = np.asarray(out["y"])
    th = np.asarray(out["th"])
    assert x.shape == y.shape == th.shape
    assert x.ndim == 1
    # should produce at least one valid successor on an empty map
    assert x.size > 0


@pytest.mark.cpp
@pytest.mark.slow
def test_cpp_backend_planner_smoke(empty_grid, planner_config, start_pose, goal_spec):
    # This is an integration smoke test: requires a compiled extension that exposes run_search.
    try:
        import hybrid_core  # type: ignore
    except Exception as e:
        pytest.skip(f"C++ extension not built/importable: {e}")
    if not hasattr(hybrid_core, "run_search"):
        pytest.skip("hybrid_core does not expose run_search in this build")
    try:
        from src.cpp_kernels import CPP_AVAILABLE
    except Exception:
        pytest.skip("cpp_kernels import failed")
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")
    from src.planners import HybridAStarPlannerCpp

    planner_cpp = HybridAStarPlannerCpp(empty_grid, planner_config)
    path_cpp, stats_cpp = planner_cpp.plan(start_pose, goal_spec, max_expansions=50_000)

    assert stats_cpp.expanded > 0
    assert len(path_cpp) > 1

    # goal reached within tolerance
    dx = path_cpp[-1].x - goal_spec.pose.x
    dy = path_cpp[-1].y - goal_spec.pose.y
    assert dx * dx + dy * dy <= goal_spec.pos_tol * goal_spec.pos_tol
    dth = (path_cpp[-1].theta - goal_spec.pose.theta + math.pi) % (2 * math.pi) - math.pi
    assert abs(dth) <= goal_spec.theta_tol

