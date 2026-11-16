"""
Tabular Q-learning.

The whole "brain" is a table Q[state, action]: how much total future reward
the agent expects if it takes `action` in `state` and plays well afterwards.

After every move it nudges one entry toward a better guess (the Bellman update):

    target   = reward + gamma * max_a' Q[next_state, a']     (0 future if the episode ended)
    Q[s, a] += alpha * (target - Q[s, a])

Exploration is epsilon-greedy: with probability epsilon take a random move,
otherwise take the best-known move. Epsilon shrinks after each episode.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class QConfig:
    alpha: float = 0.5           # learning rate: how far each update moves the estimate
    gamma: float = 0.95          # discount: how much future reward matters
    epsilon: float = 1.0         # starting exploration rate
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.93  # multiply epsilon by this after every episode
    decay: bool = True


class QLearningAgent:
    def __init__(self, n_states, n_actions=4, config=None, seed=None):
        self.cfg = config or QConfig()
        self.n_actions = n_actions
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.visits = np.zeros(n_states, dtype=np.int64)
        self.epsilon = self.cfg.epsilon
        self.rng = np.random.default_rng(seed)

    def best_action(self, state):
        """Highest Q-value, breaking ties at random (early on, every value is 0)."""
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def act(self, state, greedy=False):
        """Returns (action, explored?)."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions)), True
        return self.best_action(state), False

    def update(self, state, action, reward, next_state, terminated):
        old = float(self.q[state, action])
        max_next = 0.0 if terminated else float(self.q[next_state].max())
        target = reward + self.cfg.gamma * max_next
        new = old + self.cfg.alpha * (target - old)
        self.q[state, action] = new
        self.visits[next_state] += 1
        return {"old": old, "new": new, "target": target, "max_next": max_next}

    def end_episode(self):
        if self.cfg.decay:
            self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def reset(self):
        self.q[:] = 0.0
        self.visits[:] = 0
        self.epsilon = self.cfg.epsilon


def run_episode(env, agent, on_step=None):
    """Play and learn from one episode. Returns (steps, total_reward, reached_goal)."""
    state, _ = env.reset()
    agent.visits[state] += 1
    total, done = 0.0, False
    while not done:
        action, explored = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        upd = agent.update(state, action, reward, next_state, terminated)
        total += reward
        done = terminated or truncated
        if on_step:
            on_step(state, action, explored, reward, next_state, done, info, upd)
        state = next_state
    agent.end_episode()
    return env.steps, total, env.pos == env.goal


if __name__ == "__main__":
    from maze_env import MazeEnv

    env = MazeEnv(size=9, seed=0)
    agent = QLearningAgent(env.observation_space.n, seed=0)
    print(f"shortest path: {env.shortest_path_length()} moves")
    for ep in range(1, 101):
        steps, total, reached = run_episode(env, agent)
        if ep % 10 == 0 or ep == 1:
            print(f"episode {ep:3d} | steps {steps:4d} | return {total:+.2f} | epsilon {agent.epsilon:.2f}")
