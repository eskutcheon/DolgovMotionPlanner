

""" CLI entry point for running a simple planning example """

import numpy as np

from dolgov_cbmp.structs import PlannerConfig
from dolgov_cbmp.models import OccupancyGrid
from dolgov_cbmp.planners import planner_factory
from dolgov_cbmp.settings import parse_planning_inputs


    # TODO: might actually just make a new file in `settings` for ingesting inputs from world config files
    #   - mostly since this is getting pretty unwieldy and the CLI parsing is really just one way to provide those inputs
    #   - eventually want to support a more programmatic API for running the planner with different configs as well
    #   - should be able to remove and reuse a couple methods currently in OccupancyGrid
    # TODO: really considering making a new major object like `WorldModel` that encapsulates the OccupancyGrid, the GridSpec, and the start and goal poses
    #   - also considering adding something new like `CurvatureParams` to either the PlannerConfig or to GridSpec for defining curvature kinematics
    #       (e.g. `kappa_max`, `kappa_bins`, `kappa_min`) since this is really a core part of the problem definition and is currently duplicated in some places
    #       though we'd need to differentiate between what are physical world specifications vs planner constraints (e.g. `kappa_rate_max`, `kappa_rate_samples`)
    #   - might pitch this to Copilot since it touches so many files - remember to get it to update the README
    #   - this could also contain the Indexer model used by the planners, or if nothing else a factory method for creating an Indexer object


def get_toggles_from_cli(cfg: PlannerConfig) -> dict[str, bool]:
    return {
        "analytic_connections": cfg.use_analytic_connector,
        "adaptive_analytic_connections": cfg.analytic.use_adaptive_schedule,
        "path_smoothing": cfg.use_path_smoothing,
        "objective_smoothing": cfg.smoother.use_objective_smoother,
        "use_nh_heuristic": cfg.heuristics.use_nonholonomic,
        "use_h2d_voronoi": cfg.heuristics.use_voronoi,
        "variable_step_policy": cfg.step_policy.use_variable_step,
        "allow_reverse_motion": cfg.allow_reverse, #TODO: unit test this being False - not sure if I've tried that at all so far
    }


def main() -> None:
    args = parse_planning_inputs()
    # TODO: replace with loading a map from file, e.g., via world_cfg_path
    occ = np.zeros((200, 200), dtype=bool)
    occ[80:120, 100] = True
    # TODO: also consider implementing the new major WorldConfig to replace the instantiations below
    og = OccupancyGrid(occ, args.planner_config.grid)

    planner = planner_factory(og, args.planner_config, backend=args.backend)
    print(
        "planning with settings:"
        f"\n  backend={args.backend}, max_expansions={args.max_expansions}, step_size={args.planner_config.step_size}, "
        # TODO: honestly really need to add the shape here, but it requires refactoring how and when we create these and the OccupancyGrid together
        f"\n  grid resolution={args.planner_config.grid.resolution}, grid size={(og.height, og.width)}, "
        f"\n  start=({args.start.x}, {args.start.y}, {args.start.theta}), "
        f"\t  goal=({args.goal.pose.x}, {args.goal.pose.y}, {args.goal.pose.theta})"
    )
    for toggle_name, enabled in get_toggles_from_cli(args.planner_config).items():
        print(f"  {toggle_name}: {'ON' if enabled else 'OFF'}")
    # run planner
    path, stats = planner.plan(args.start, args.goal, max_expansions=args.max_expansions)
    # print from returned path and stats
    print(
        f"path poses={len(path)} | expanded={stats.expanded} | "
        f"pushed={stats.pushed} | elapsed={stats.elapsed_s():.3f}s"
    )


if __name__ == "__main__":
    main()