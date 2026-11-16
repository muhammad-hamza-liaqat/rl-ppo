"""
WebSocket session for the Q-learning maze page (/maze in the web UI).

Client -> server
    {"type": "new_maze", "size": 11, "seed": 42}    seed optional
    {"type": "settings", "alpha": .5, "gamma": .95, "epsilon": .3, "decay": true}
    {"type": "speed", "delay_ms": 80}
    {"type": "learn"}                    play and learn, streaming every move
    {"type": "pause"}
    {"type": "train_fast", "episodes": 50}
    {"type": "walk"}                     follow the best-known move from the start, no learning
    {"type": "reset"}                    forget everything (Q-table back to zeros)

Server -> client
    maze, snapshot, step, episode, episodes, walk_step, status, error
"""

import asyncio
import random

from maze_env import MazeEnv
from qlearning import QConfig, QLearningAgent, run_episode

SIZES = (7, 9, 11, 13, 15)


def r4(x):
    return round(float(x), 4)


class MazeSession:
    def __init__(self, send):
        self.send = send  # async callable(dict)
        self.task: asyncio.Task | None = None
        self.delay = 0.08
        self.status = "idle"
        self.cfg = QConfig()
        self.new_maze(11, None)

    # ------------------------------------------------------------------ #
    def new_maze(self, size, seed):
        size = size if size in SIZES else 11
        seed = seed if seed is not None else random.randrange(1_000_000)
        self.env = MazeEnv(size=size, seed=seed)
        epsilon = self.cfg.epsilon
        self.agent = QLearningAgent(self.env.observation_space.n, config=self.cfg)
        self.agent.epsilon = epsilon
        self.episode = 0
        self.history = []

    def maze_msg(self):
        e = self.env
        return {
            "type": "maze",
            "size": e.size,
            "seed": e.maze_seed,
            "walls": sorted([list(w) for w in e.walls]),
            "start": list(e.start),
            "goal": list(e.goal),
            "shortest": e.shortest_path_length(),
            "max_steps": e.max_steps,
        }

    def snapshot_msg(self):
        return {
            "type": "snapshot",
            "q": [[r4(v) for v in row] for row in self.agent.q],
            "visits": self.agent.visits.tolist(),
            "epsilon": r4(self.agent.epsilon),
            "episode": self.episode,
            "history": self.history,
            "settings": {
                "alpha": self.cfg.alpha,
                "gamma": self.cfg.gamma,
                "epsilon": r4(self.agent.epsilon),
                "decay": self.cfg.decay,
            },
            "status": self.status,
        }

    async def set_status(self, status):
        self.status = status
        await self.send({"type": "status", "status": status})

    def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None

    async def stop_and_idle(self):
        self.stop()
        await self.set_status("idle")

    # ------------------------------------------------------------------ #
    def record(self, steps, total, reached):
        point = {
            "episode": self.episode,
            "steps": int(steps),
            "return": r4(total),
            "reached": bool(reached),
            "epsilon": r4(self.agent.epsilon),
        }
        self.history.append(point)
        return point

    async def learn_loop(self):
        """Play episodes forever, one streamed move at a time."""
        env, agent = self.env, self.agent
        try:
            await self.set_status("learning")
            while True:
                self.episode += 1
                state, _ = env.reset()
                agent.visits[state] += 1
                await self.send({"type": "step", "episode": self.episode, "step": 0,
                                 "agent": list(env.start), "start_of_episode": True,
                                 "epsilon": r4(agent.epsilon)})
                await asyncio.sleep(self.delay)
                total, done = 0.0, False
                while not done:
                    from_cell = env.pos
                    action, explored = agent.act(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    upd = agent.update(state, action, reward, next_state, terminated)
                    total += reward
                    done = terminated or truncated
                    await self.send({
                        "type": "step",
                        "episode": self.episode,
                        "step": env.steps,
                        "from": list(from_cell),
                        "agent": list(env.pos),
                        "action": action,
                        "explored": explored,
                        "bumped": info["bumped"],
                        "reward": reward,
                        "update": {"cell": list(from_cell), "action": action,
                                   "old": r4(upd["old"]), "new": r4(upd["new"]),
                                   "target": r4(upd["target"]), "max_next": r4(upd["max_next"]),
                                   "terminal": terminated},
                        "q_row": {"state": int(state), "values": [r4(v) for v in agent.q[state]]},
                        "total": r4(total),
                        "done": done,
                        "result": info.get("result"),
                        "epsilon": r4(agent.epsilon),
                    })
                    state = next_state
                    await asyncio.sleep(self.delay)
                agent.end_episode()
                point = self.record(env.steps, total, env.pos == env.goal)
                await self.send({"type": "episode", "point": point, "epsilon": r4(agent.epsilon)})
                await asyncio.sleep(max(0.4, self.delay * 4))
        except asyncio.CancelledError:
            raise

    async def train_fast(self, episodes):
        await self.stop_and_idle()
        await self.set_status("training")
        points = []
        for _ in range(episodes):
            self.episode += 1
            steps, total, reached = run_episode(self.env, self.agent)
            points.append(self.record(steps, total, reached))
        await self.send({"type": "episodes", "points": points})
        await self.set_status("idle")
        await self.send(self.snapshot_msg())

    async def walk_loop(self):
        """Follow the greedy policy from the start. Stops at the goal or when it loops."""
        env, agent = self.env, self.agent
        await self.set_status("walking")
        state, _ = env.reset()
        seen = {state}
        await self.send({"type": "walk_step", "agent": list(env.pos), "step": 0, "done": False})
        await asyncio.sleep(max(self.delay, 0.12))
        while True:
            action = agent.best_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            looped = (state in seen) and not terminated
            seen.add(state)
            done = terminated or truncated or looped
            result = "goal" if terminated else "loop" if looped else "timeout" if truncated else None
            await self.send({"type": "walk_step", "agent": list(env.pos), "action": action,
                             "step": env.steps, "done": done, "result": result,
                             "shortest": env.shortest_path_length()})
            if done:
                break
            await asyncio.sleep(max(self.delay, 0.12))
        await self.set_status("idle")

    # ------------------------------------------------------------------ #
    async def start(self):
        await self.send(self.maze_msg())
        await self.send(self.snapshot_msg())

    async def handle(self, msg):
        kind = msg.get("type")
        if kind == "new_maze":
            await self.stop_and_idle()
            seed = msg.get("seed")
            self.new_maze(int(msg.get("size", self.env.size)), int(seed) if seed is not None else None)
            await self.start()
        elif kind == "settings":
            self.cfg.alpha = min(max(float(msg.get("alpha", self.cfg.alpha)), 0.01), 1.0)
            self.cfg.gamma = min(max(float(msg.get("gamma", self.cfg.gamma)), 0.0), 0.999)
            self.cfg.decay = bool(msg.get("decay", self.cfg.decay))
            if "epsilon" in msg:
                self.agent.epsilon = min(max(float(msg["epsilon"]), 0.0), 1.0)
            await self.send({"type": "settings", "alpha": self.cfg.alpha, "gamma": self.cfg.gamma,
                             "epsilon": r4(self.agent.epsilon), "decay": self.cfg.decay})
        elif kind == "speed":
            self.delay = min(max(float(msg.get("delay_ms", 80)) / 1000, 0.0), 2.0)
        elif kind == "learn":
            self.stop()
            self.task = asyncio.create_task(self.learn_loop())
        elif kind == "pause":
            await self.stop_and_idle()
            await self.send(self.snapshot_msg())
        elif kind == "train_fast":
            await self.train_fast(min(max(int(msg.get("episodes", 50)), 1), 2000))
        elif kind == "walk":
            self.stop()
            self.task = asyncio.create_task(self.walk_loop())
        elif kind == "reset":
            await self.stop_and_idle()
            self.agent.reset()
            self.episode = 0
            self.history = []
            await self.send(self.snapshot_msg())
