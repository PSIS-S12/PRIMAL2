
import os
import argparse
import numpy as np


def load_movingai_map(map_path):
    """
    Read a MovingAI .map file and convert it to a state_map:
      - free cell  -> 0
      - obstacle   -> -1
    """
    with open(map_path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    try:
        start_idx = lines.index("map") + 1
    except ValueError:
        raise ValueError(f"'map' section not found in {map_path}")

    grid = lines[start_idx:]
    if not grid:
        raise ValueError(f"No grid data found in {map_path}")

    h = len(grid)
    w = len(grid[0])

    state_map = np.zeros((h, w), dtype=int)

    for i, row in enumerate(grid):
        if len(row) != w:
            raise ValueError(f"Inconsistent row length in {map_path} at row {i}")
        for j, ch in enumerate(row):
            if ch != '.':
                state_map[i, j] = -1

    return state_map


def make_goals_map(state_map, num_agents, seed=None):
    """
    Create a goal map with agent IDs placed on random free cells.
    """
    rng = np.random.default_rng(seed)
    free_cells = np.argwhere(state_map == 0)

    if len(free_cells) < num_agents:
        raise ValueError(
            f"Not enough free cells ({len(free_cells)}) for {num_agents} agents"
        )

    chosen = rng.choice(len(free_cells), size=num_agents, replace=False)
    goals_map = np.zeros_like(state_map, dtype=int)

    for agent_id, idx in enumerate(chosen, start=1):
        x, y = free_cells[idx]
        goals_map[x, y] = agent_id

    return goals_map


def convert_single_map(map_path, out_path, num_agents, seed=None):
    """
    Convert one MovingAI .map file into a simple npy with:
        [state_map, goals_map]
    This is intentionally separate from any PRIMAL world logic.
    """
    state_map = load_movingai_map(map_path)
    goals_map = make_goals_map(state_map, num_agents=num_agents, seed=seed)

    np.save(out_path, np.array([[state_map, goals_map]], dtype=object))
    return out_path


def convert_folder(input_dir, output_dir, num_agents, seed=None):
    os.makedirs(output_dir, exist_ok=True)

    for name in os.listdir(input_dir):
        if not name.endswith(".map"):
            continue
        map_path = os.path.join(input_dir, name)
        base = os.path.splitext(name)[0]
        out_path = os.path.join(output_dir, base + ".npy")
        convert_single_map(map_path, out_path, num_agents=num_agents, seed=seed)
        print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a MovingAI .map file or folder")
    parser.add_argument("--output", required=True, help="Output .npy file or folder")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents/goals to place")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        convert_folder(args.input, args.output, num_agents=args.agents, seed=args.seed)
    else:
        convert_single_map(args.input, args.output, num_agents=args.agents, seed=args.seed)
        print(f"saved {args.output}")