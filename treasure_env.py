"""
Treasure Grid — a tiny custom Gymnasium environment.

The agent (A) walks on a grid. It must reach the treasure (T) and avoid traps (X).
Every new episode places A, T and the traps at random, so the agent has to learn
a general strategy instead of memorising one path.

    Observation (8 numbers, all scaled to 0..1):
        [agent_row, agent_col, treasure_row, treasure_col,
         trap_up, trap_down, trap_left, trap_right]

    Actions (4):  0 = up, 1 = down, 2 = left, 3 = right

    Rewards:
        +1.00  reach the treasure   (episode ends)
        -1.00  step on a trap       (episode ends)
        -0.01  every other step     (encourages short paths)
    Episode is cut off ("truncated") after `max_steps` moves.
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# action id -> (row change, col change)
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


class TreasureGridEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=6, n_traps=4, max_steps=50, render_mode=None):
        super().__init__()
        self.size = size
        self.n_traps = n_traps
        self.max_steps = max_steps
        self.render_mode = render_mode

        # What the agent sees and what it can do — required by Gymnasium.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random

        # Place agent, treasure and traps on distinct random cells.
        # Re-roll until the treasure is actually reachable.
        while True:
            cells = self.np_random.choice(self.size * self.size, size=2 + self.n_traps, replace=False)
            positions = [divmod(int(c), self.size) for c in cells]
            self.agent = positions[0]
            self.treasure = positions[1]
            self.traps = set(positions[2:])
            if self._path_exists():
                break

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        dr, dc = MOVES[int(action)]
        r, c = self.agent[0] + dr, self.agent[1] + dc
        if 0 <= r < self.size and 0 <= c < self.size:  # walls: bumping = stay put
            self.agent = (r, c)
        self.steps += 1

        terminated = False
        result = None
        reward = -0.01
        if self.agent in self.traps:
            reward, terminated, result = -1.0, True, "trap"
        elif self.agent == self.treasure:
            reward, terminated, result = 1.0, True, "win"

        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated:
            result = "timeout"

        info = {"result": result} if result else {}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) == self.agent:
                    row.append("A")
                elif (r, c) == self.treasure:
                    row.append("T")
                elif (r, c) in self.traps:
                    row.append("X")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        scale = self.size - 1
        ar, ac = self.agent
        trap_flags = [float((ar + dr, ac + dc) in self.traps) for dr, dc in MOVES.values()]
        return np.array(
            [ar / scale, ac / scale, self.treasure[0] / scale, self.treasure[1] / scale, *trap_flags],
            dtype=np.float32,
        )

    def _path_exists(self):
        """Breadth-first search from agent to treasure, avoiding traps."""
        seen = {self.agent}
        queue = deque([self.agent])
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.treasure:
                return True
            for dr, dc in MOVES.values():
                nxt = (r + dr, c + dc)
                if (
                    0 <= nxt[0] < self.size
                    and 0 <= nxt[1] < self.size
                    and nxt not in self.traps
                    and nxt not in seen
                ):
                    seen.add(nxt)
                    queue.append(nxt)
        return False


if __name__ == "__main__":
    # Quick sanity check: play one episode with random actions.
    env = TreasureGridEnv()
    obs, _ = env.reset(seed=0)
    print(env.render(), "\nobs:", obs, "\n")
    done, total = False, 0.0
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        done = terminated or truncated
    print(env.render())
    print(f"random agent finished: {info['result']}, total reward {total:.2f}")
