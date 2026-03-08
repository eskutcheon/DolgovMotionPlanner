# src/dolgov_cbmp/__main__.py
""" CLI entry point for running a simple planning example """

from dolgov_cbmp.structs import GridSpec, WorldModel
from dolgov_cbmp.settings import parse_planning_inputs, load_world_model, PlannerConfig
from dolgov_cbmp.models import OccupancyGrid
from dolgov_cbmp.planners import planner_factory



# TODO: might actually just make a new file in `settings` for ingesting inputs from world config files
#   - mostly since this is getting pretty unwieldy and the CLI parsing is really just one way to provide those inputs
#   - eventually want to support a more programmatic API for running the planner with different configs as well
#   - should be able to remove and reuse a couple methods currently in OccupancyGrid


def _get_toggles_from_cli(cfg: PlannerConfig) -> dict[str, bool]:
    return {
        "analytic_connections": cfg.use_analytic_connector,
        "adaptive_analytic_connections": cfg.analytic.use_adaptive_schedule,
        "path_smoothing": cfg.use_path_smoothing,
        "objective_smoothing": cfg.smoother.use_objective_smoother,
        "use_nh_heuristic": cfg.heuristics.use_nonholonomic,
        "use_h2d_voronoi": cfg.heuristics.use_voronoi,
        "variable_step_policy": cfg.step_policy.use_variable_step,
        "allow_reverse_motion": cfg.allow_reverse, #TODO: unit test this when set to False - not sure if I've tried that at all so far
    }


def _default_world(cfg: PlannerConfig, start, goal) -> WorldModel:
    import numpy as np
    occ = np.zeros((200, 200), dtype=bool)
    occ[80:120, 100] = True
    og = OccupancyGrid(occ, GridSpec())
    world = WorldModel(occupancy_grid=og, start=start, goal=goal, vehicle=cfg.vehicle)
    world.validate()
    return world


def main() -> None:
    args = parse_planning_inputs()
    planner_cfg = args.planner_config
    world: WorldModel
    if args.world_cfg_path:
        world, planner_cfg = load_world_model(args.world_cfg_path, args.planner_config)
    else:
        world = _default_world(args.planner_config, args.start, args.goal)
    planner = planner_factory(world, planner_cfg, backend=args.backend)
    print(
        "planning with settings:"
        f"\n  backend={args.backend}, max_expansions={args.max_expansions}, step_size={planner_cfg.step_size}, "
        f"\n  grid resolution={world.grid.resolution}, grid size={(world.occupancy_grid.height, world.occupancy_grid.width)}, "
        f"\n  start=({world.start.x}, {world.start.y}, {world.start.theta}), "
        f"\t  goal=({world.goal.pose.x}, {world.goal.pose.y}, {world.goal.pose.theta})"
    )
    for toggle_name, enabled in _get_toggles_from_cli(planner_cfg).items():
        print(f"  {toggle_name}: {'ON' if enabled else 'OFF'}")
    # run planner
    path, stats = planner.plan(world.start, world.goal, max_expansions=args.max_expansions)
    # print from returned path and stats
    print(
        f"path poses={len(path)} | expanded={stats.expanded} | "
        f"pushed={stats.pushed} | elapsed={stats.elapsed_s():.3f}s"
    )


if __name__ == "__main__":
    main()