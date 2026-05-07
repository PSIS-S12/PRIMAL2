import numpy as np

files = [
    "./8agents_same50_011_3.npy",
    "./16agents_same50_011_3.npy",
    "./32agents_same50_011_3.npy",
    "./64agents_same50_011_3.npy",
    "./128agents_same50_011_3.npy",
]

base_walls = None
base_shape = None

for file in files:
    data = np.load(file, allow_pickle=True)
    state = data[0][0]

    walls = state == -1
    shape = state.shape
    num_agents = int(state.max())
    num_goals = int(data[0][1].max())

    print(f"\nFile: {file}")
    print(f"Shape: {shape}")
    print(f"Agents: {num_agents}")
    print(f"Goals: {num_goals}")

    if base_walls is None:
        base_walls = walls
        base_shape = shape
        print("Map check: BASE")
    else:
        same_shape = shape == base_shape
        same_walls = np.array_equal(base_walls, walls)

        print(f"Same shape: {same_shape}")
        print(f"Same walls: {same_walls}")

        if not same_walls:
            diff = np.sum(base_walls != walls)
            print(f"Different wall cells: {diff}")

print("\nDone.")
