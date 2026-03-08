""" generate small deterministic world-config test assets for local development """

from pathlib import Path
import json
import numpy as np


if __name__ == "__main__":
    out = Path("tests/world_configs/generated")
    out.mkdir(parents=True, exist_ok=True)
    # create dummy occupancy grid and save in multiple formats
    occ = np.zeros((8, 8), dtype=np.uint8)
    occ[3:5, 3:5] = 1
    np.save(out / "occ.npy", occ)
    np.savez(out / "occ.npz", occupancy=occ)
    # define world config doc with path to occupancy grid npy file
    doc = {
        "format_version": "1.0",
        "grid": {"resolution": 1.0, "theta_bins": 36, "origin_xy": [0.0, 0.0], "kappa_bins": 11},
        "occupancy": {"path": str((out / "occ.npy").resolve())},
        "start": {"x": 1.0, "y": 1.0, "theta": 0.0, "kappa": 0.0},
        "goal": {"pose": {"x": 6.0, "y": 6.0, "theta": 0.0, "kappa": 0.0}, "pos_tol": 0.5, "theta_tol": 0.26},
        "planner_overrides": {"step_size": 1.5},
    }
    (out / "world.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    np.savez(out / "world_doc.npz", world_cfg=np.array(json.dumps(doc), dtype=np.str_))
    print(f"Wrote assets to: {out}")