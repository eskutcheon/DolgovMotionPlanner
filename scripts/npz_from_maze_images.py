"""
Python script to accept simple image files of mazes
1. binarize images (assuming black and white input images) and invert to occupancy grid format (0=free, 1=obstacle)
2. ~~ensure no empty space on borders (for square mazes)~~ - REMOVED - REVISIT LATER
3. perform erosion + dilation to clean up noise at the edges
4. resize to various resolutions (using nearest neighbor interpolation to preserve binary nature of grid) while preserving aspect ratio
5. find free midpoints on top and bottom edges for start and goal poses (assuming entrances/exits are on these edges)
6. save as numpy arrays (occupancy grid + start/goal poses) in .npz format for use in tests
"""

from typing import List
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation


def view_grid(occ: np.ndarray) -> None:
    """ print grid to console while highlighting 1s as border cells """
    H, W = occ.shape
    OBS_FMT = "\033[93m1\033[0m"
    for r in range(H):
        for c in range(W):
            if occ[r, c]:
                print(OBS_FMT, end="")
            else:
                print("0", end="")
        print()
    print()


def create_maze_grid(image_path: Path) -> np.ndarray:
    with Image.open(image_path).convert("L") as img:  # convert to grayscale
        maze = np.array(img, dtype=np.uint8)
    # binarize image (with arbitrary threshold=32)
    maze = np.where(maze > 32, 0, 1).astype(bool)  # free space=0, obstacles=1
    # perform erosion + dilation (morphological opening) to clean up noise at edges and to widen paths slightly to ensure connectivity given vehicle footprint
    maze_cleaned = binary_erosion(maze, structure=np.ones((5,5), dtype=bool))
    maze_cleaned = binary_dilation(maze_cleaned, structure=np.ones((3, 3), dtype=bool))
    return maze_cleaned.astype(np.uint8) # convert to byte for nearest neighbor resizing


def resize_maze_grid(maze: np.ndarray, target_size: int) -> np.ndarray:
    """ Resize maze grid to target size while preserving aspect ratio using nearest neighbor interpolation """
    h, w = maze.shape
    if h < w:
        new_h = target_size
        new_w = int(target_size * w / h)
    else:
        new_w = target_size
        new_h = int(target_size * h / w)
    resized_maze = Image.fromarray(maze).resize((new_w, new_h), resample=Image.NEAREST)
    return np.array(resized_maze, dtype=np.uint8)


def find_free_midpoint(edge: np.ndarray) -> List[int]:
    """ Find midpoints of free (0) segments on the edge
        WARNING: assumes the edge lies exactly on the border of the maze, so this search is a simple 1D search along the image's edge
            If this differs in the future, this logic will need to be updated to find free segments in a 2D band along the edge and find their midpoints accordingly
    """
    free_indices = np.where(edge == 0)[0]
    midpoints = []
    if len(free_indices) == 0:
        raise ValueError("No free areas (without obstacles) found on edge")
    # group consecutive free space indices along the edge and find their midpoints
    groups = np.split(free_indices, np.where(np.diff(free_indices) != 1)[0] + 1)
    for group in groups:
        mid_idx = group[len(group) // 2]
        midpoints.append(mid_idx)
    if len(midpoints) > 1:
        print("[WARNING] This script assumes entrances/exits on top and bottom edges, found", len(midpoints), "candidate regions")
        print("\t Choosing the second empty space as the entrance/exit by default, but please verify this is correct for your maze and adjust the logic if necessary")
    idx = int(len(midpoints) > 1)  # choose second if exists
    return midpoints[idx]



if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    default_output_dir = str(project_root / "tests" / "grids")
    parser = argparse.ArgumentParser(description="Convert maze images to .npz files with start/goal poses")
    parser.add_argument("input_dir", type=str, help="Directory containing input maze images")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Directory to save output .npz files")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[50, 100], help="List of target resolutions for resizing (minimum side length in pixels)")
    args = parser.parse_args()
    # handle input and output directories and find all eligible image files before processing
    input_dir = Path(args.input_dir)
    assert input_dir.is_dir(), f"Input directory {input_dir} does not exist or is not a directory"
    target_dir = Path(args.output_dir)
    target_dir.mkdir(exist_ok=True)
    img_files = [f for f in input_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    # print("img files found:", img_files)
    for img in img_files:
        print("\nProcessing file:", img)
        maze_grid = create_maze_grid(img)
        # view_grid(maze_grid)
        for res in args.resolutions:
            resized_grid = resize_maze_grid(maze_grid, res)
            view_grid(resized_grid)
            start_midpoint = find_free_midpoint(resized_grid[0, :]) # find free midpoint on top edge for start pose
            goal_midpoint = find_free_midpoint(resized_grid[-1, :]) # find free midpoint on bottom edge for goal pose
            # shifting both start and goal poses away from edges by 2 pixels to ensure they begin in free space given the vehicle footprint
            start_coords = (start_midpoint, 2)  # (x, y) "indices" for start pose
            goal_coords = (goal_midpoint, resized_grid.shape[0] - 3)  # (x, y) "indices" for goal pose
            #? NOTE: might come back and separate start and goal as separate npz fields, but I don't feel like changing the existing 'poses' key yet
            print(f"Start midpoint on top edge: {start_coords}, Goal midpoint on bottom edge: {goal_coords}")
            out_path = target_dir / (img.stem + f"_res{res}.npz")
            np.savez(out_path, occupancy=resized_grid, poses=np.array([start_coords, goal_coords], dtype=np.int32))
            print("Saved resized grid and start/goal poses to:", out_path)