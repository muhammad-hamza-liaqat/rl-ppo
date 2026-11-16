"""
Maze — a small Gymnasium environment for tabular Q-learning.

The agent starts in the top-left corner and must reach the goal in the
bottom-right corner. Walls block movement (bumping a wall = stay put).

    Observation: one integer, the cell index  row * size + col
    Actions (4): 0 = up, 1 = down, 2 = left, 3 = right
    Rewards:
        +1.00  reach the goal   (episode ends)
        -0.01  every other step (encourages short paths)
    Episode is cut off after `max_steps` moves.

Unlike Treasure Grid, the layout stays the same between episodes: Q-learning
learns *this* maze, one table row per cell.
"""

import random
from collections import deque

import gymnasium as gym
from gymnasium import spaces

MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def generate_maze(size, seed=None, loop_chance=0.12):
    """
    Randomised depth-first search on an odd-sized grid, then knock out a few
    extra walls so there is more than one route (more interesting to learn).
    Returns the set of wall cells.
    """
    if size % 2 == 0 or size < 5:
        raise ValueError("size must be odd and at least 5")
    rng = random.Random(seed)
    walls = {(r, c) for r in range(size) for c in range(size)}

    # Rooms live on odd coordinates; carve passages between them.
    start = (1, 1)
    walls.discard(start)
    stack = [start]
    while stack:
        r, c = stack[-1]
        options = [
            (r + dr * 2, c + dc * 2, r + dr, c + dc)
            for dr, dc in MOVES.values()
            if 0 < r + dr * 2 < size - 1 and 0 < c + dc * 2 < size - 1 and (r + dr * 2, c + dc * 2) in walls
        ]
        if not options:
            stack.pop()
            continue
        nr, nc, wr, wc = rng.choice(options)
        walls.discard((wr, wc))
        walls.discard((nr, nc))
        stack.append((nr, nc))

    # Braid: open some inner walls that sit between two open cells.
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if (r, c) not in walls:
                continue
            horizontal = (r, c - 1) not in walls and (r, c + 1) not in walls
            vertical = (r - 1, c) not in walls and (r + 1, c) not in walls
            if (horizontal ^ vertical) and rng.random() < loop_chance:
                walls.discard((r, c))
    return walls


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=9, seed=None, max_steps=None):
        super().__init__()
        self.size = size
        self.maze_seed = seed
        self.walls = generate_maze(size, seed)
        self.start = (1, 1)
        self.goal = (size - 2, size - 2)
        self.max_steps = max_steps or size * size * 3

        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)
        self.pos = self.start
        self.steps = 0

    def state_of(self, cell):
        return cell[0] * self.size + cell[1]

    def cell_of(self, state):
        return divmod(int(state), self.size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        self.steps = 0
        return self.state_of(self.pos), {}

    def step(self, action):
        dr, dc = MOVES[int(action)]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        bumped = nxt in self.walls
        if not bumped:
            self.pos = nxt
        self.steps += 1

        terminated = self.pos == self.goal
        reward = 1.0 if terminated else -0.01
        truncated = (not terminated) and self.steps >= self.max_steps
        result = "goal" if terminated else "timeout" if truncated else None
        info = {"bumped": bumped}
        if result:
            info["result"] = result
        return self.state_of(self.pos), reward, terminated, truncated, info

    def shortest_path_length(self):
        """Breadth-first search: the fewest moves from start to goal."""
        seen = {self.start: 0}
        queue = deque([self.start])
        while queue:
            cell = queue.popleft()
            if cell == self.goal:
                return seen[cell]
            for dr, dc in MOVES.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt not in self.walls and nxt not in seen:
                    seen[nxt] = seen[cell] + 1
                    queue.append(nxt)
        return None

    def render(self):
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) == self.pos:
                    row.append("A")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) in self.walls:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)


if __name__ == "__main__":
    env = MazeEnv(size=9, seed=0)
    env.reset()
    print(env.render())
    print("shortest path:", env.shortest_path_length(), "moves")
