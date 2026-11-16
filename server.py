"""
Live web server for Treasure Grid.

    uvicorn server:app --port 8000

The browser (web/) connects to ws://localhost:8000/ws and gets a stream of
"frames": the board, what the agent sees, and what it is thinking (move
probabilities from the actor, value estimate from the critic).

Client -> server messages
    {"type": "watch", "agent": "trained" | "random"}   stream episodes forever
    {"type": "human_reset"}                            start a game you play
    {"type": "human_move", "action": 0..3}             0 up, 1 down, 2 left, 3 right
    {"type": "speed", "delay_ms": 300}                 time between streamed moves
    {"type": "stop"}                                   stop streaming
    {"type": "train", "updates": 150}                  train PPO (shared by all tabs)
    {"type": "stop_train"}

Server -> client messages
    hello, frame, train_start, train_progress, train_preview, train_done, error
"""

import asyncio
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from maze_session import MazeSession
from ppo import PPO, PPOConfig, load_policy
from rlhf_session import RLHFSession
from vision_session import VisionSession
from vision_session import hub as vision_hub
from train import evaluate
from treasure_env import TreasureGridEnv

MODEL_PATH = Path(__file__).with_name("ppo_treasure.pt")

# These networks are tiny: one CPU thread is faster than several (less coordination overhead).
torch.set_num_threads(1)
UNIFORM = [0.25, 0.25, 0.25, 0.25]

app = FastAPI(title="Treasure Grid live server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------- #
# Helpers: turn the environment + network into JSON the browser can draw
# ---------------------------------------------------------------------- #
@torch.no_grad()
def think(net, obs):
    """What the network thinks about this state: move probabilities and value."""
    if net is None:
        return UNIFORM, None
    t = torch.as_tensor(obs)
    probs = torch.softmax(net.actor(t), dim=-1).tolist()
    return probs, net.value(t).item()


def make_frame(env, obs, net, *, source, episode, step, total,
               action=None, reward=None, done=False, result=None):
    probs, value = (None, None) if done else think(net, obs)
    if net is None and not done:
        probs = UNIFORM
    return {
        "type": "frame",
        "source": source,  # trained | random | human | preview
        "episode": episode,
        "step": step,
        "board": {
            "size": env.size,
            "agent": list(env.agent),
            "treasure": list(env.treasure),
            "traps": sorted([list(t) for t in env.traps]),
        },
        "obs": [round(float(x), 3) for x in obs],
        "probs": probs,    # chance of each next move (up, down, left, right)
        "value": value,    # critic: expected future reward from here
        "action": action,  # the move that led to this frame
        "reward": reward,
        "total": round(total, 2),
        "done": done,
        "result": result,  # win | trap | timeout
        "max_steps": env.max_steps,
    }


def play_episode_frames(net, seed=None, source="preview", episode=1):
    """Play one greedy episode and return every frame (used for training previews)."""
    env = TreasureGridEnv()
    obs, _ = env.reset(seed=seed)
    frames = [make_frame(env, obs, net, source=source, episode=episode, step=0, total=0.0)]
    total, step, done = 0.0, 0, False
    while not done:
        probs, _ = think(net, obs)
        action = int(np.argmax(probs))
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        step += 1
        done = terminated or truncated
        frames.append(make_frame(env, obs, net, source=source, episode=episode, step=step,
                                 total=total, action=action, reward=reward, done=done,
                                 result=info.get("result")))
    return frames


# ---------------------------------------------------------------------- #
# Shared state: connected browsers, the trained model, the training run
# ---------------------------------------------------------------------- #
class Client:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.lock = asyncio.Lock()  # one message at a time per socket

    async def send(self, msg):
        async with self.lock:
            await self.ws.send_json(msg)


class Hub:
    def __init__(self):
        self.clients: set[Client] = set()
        self.trained_net = load_policy(MODEL_PATH) if MODEL_PATH.exists() else None
        self.train_task: asyncio.Task | None = None
        self.stop_requested = False
        self.train_state = {"running": False, "update": 0, "updates": 0}
        self.history: list[dict] = []  # chart points of the latest run
        self.before = None
        self.after = None

    async def broadcast(self, msg):
        for c in list(self.clients):
            try:
                await c.send(msg)
            except Exception:
                self.clients.discard(c)

    def hello(self):
        return {
            "type": "hello",
            "model_available": self.trained_net is not None,
            "training": self.train_state,
            "history": self.history,
            "before": self.before,
            "after": self.after,
            "config": {"rollout_steps": PPOConfig.rollout_steps},
        }


hub = Hub()


async def run_training(updates: int):
    env = TreasureGridEnv()
    agent = PPO(env, PPOConfig(), seed=0)
    hub.stop_requested = False
    hub.history, hub.after = [], None
    hub.train_state = {"running": True, "update": 0, "updates": updates}

    try:
        hub.before = await asyncio.to_thread(evaluate, agent.net, 300)
        await hub.broadcast({"type": "train_start", "updates": updates, "before": hub.before})

        recent = deque(maxlen=200)
        start = time.time()
        done_updates = 0
        for update in range(1, updates + 1):
            if hub.stop_requested:
                break
            stats = await asyncio.to_thread(agent.train_step)
            recent.extend(agent.finished_episodes)
            agent.finished_episodes.clear()
            done_updates = update

            n = max(len(recent), 1)
            point = {
                "update": update,
                "steps": update * agent.cfg.rollout_steps,
                "avg_reward": sum(r for r, _ in recent) / n,
                "win": sum(res == "win" for _, res in recent) / n,
                "trap": sum(res == "trap" for _, res in recent) / n,
                "entropy": stats["entropy"],
                "kl": stats["approx_kl"],
                "clip_frac": stats["clip_frac"],
                "elapsed": time.time() - start,
            }
            hub.history.append(point)
            hub.train_state = {"running": True, "update": update, "updates": updates}
            await hub.broadcast({"type": "train_progress", "point": point, "updates": updates})

            # Let the browser watch the current policy play one game.
            frames = await asyncio.to_thread(play_episode_frames, agent.net, None, "preview", update)
            await hub.broadcast({"type": "train_preview", "update": update, "frames": frames})

        stopped = done_updates < updates
        if not stopped:
            agent.save(MODEL_PATH)
            hub.trained_net = load_policy(MODEL_PATH)
            hub.after = await asyncio.to_thread(evaluate, agent.net, 500)
        await hub.broadcast({
            "type": "train_done",
            "stopped": stopped,
            "updates_done": done_updates,
            "before": hub.before,
            "after": hub.after,
            "model_available": hub.trained_net is not None,
        })
    except Exception as e:  # report, don't crash the server
        await hub.broadcast({"type": "error", "message": f"Training failed: {e}"})
    finally:
        hub.train_state = {**hub.train_state, "running": False}


# ---------------------------------------------------------------------- #
# One browser tab
# ---------------------------------------------------------------------- #
class Session:
    def __init__(self, client: Client):
        self.client = client
        self.task: asyncio.Task | None = None
        self.delay = 0.3
        self.env = TreasureGridEnv()
        self.human = None  # {"obs", "episode", "step", "total", "done"}
        self.human_episodes = 0

    def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None

    async def watch(self, agent: str):
        net = hub.trained_net if agent == "trained" else None
        if agent == "trained" and net is None:
            await self.client.send({"type": "error", "code": "no_model",
                                    "message": "No trained model yet. Train one first."})
            return
        env, episode = self.env, 0
        while True:
            episode += 1
            obs, _ = env.reset()
            total, step, done = 0.0, 0, False
            await self.client.send(make_frame(env, obs, net, source=agent, episode=episode,
                                              step=0, total=0.0))
            await asyncio.sleep(self.delay)
            while not done:
                if net is None:
                    action = int(env.action_space.sample())
                else:
                    action = int(np.argmax(think(net, obs)[0]))
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                step += 1
                done = terminated or truncated
                await self.client.send(make_frame(env, obs, net, source=agent, episode=episode,
                                                  step=step, total=total, action=action,
                                                  reward=reward, done=done,
                                                  result=info.get("result")))
                await asyncio.sleep(self.delay)
            await asyncio.sleep(max(1.2, self.delay * 3))  # pause on the result

    async def human_reset(self):
        self.stop()
        self.human_episodes += 1
        obs, _ = self.env.reset()
        self.human = {"obs": obs, "step": 0, "total": 0.0, "done": False}
        # In "play yourself" the probabilities are the trained AI's advice.
        await self.client.send(make_frame(self.env, obs, hub.trained_net, source="human",
                                          episode=self.human_episodes, step=0, total=0.0))

    async def human_move(self, action: int):
        if self.human is None or self.human["done"]:
            return
        obs, reward, terminated, truncated, info = self.env.step(action)
        h = self.human
        h["obs"], h["step"], h["total"] = obs, h["step"] + 1, h["total"] + reward
        h["done"] = terminated or truncated
        await self.client.send(make_frame(self.env, obs, hub.trained_net, source="human",
                                          episode=self.human_episodes, step=h["step"],
                                          total=h["total"], action=action, reward=reward,
                                          done=h["done"], result=info.get("result")))

    async def handle(self, msg: dict):
        kind = msg.get("type")
        if kind == "watch":
            self.stop()
            self.task = asyncio.create_task(self.watch(msg.get("agent", "trained")))
        elif kind == "stop":
            self.stop()
        elif kind == "speed":
            self.delay = min(max(float(msg.get("delay_ms", 300)) / 1000, 0.02), 2.0)
        elif kind == "human_reset":
            await self.human_reset()
        elif kind == "human_move":
            action = int(msg.get("action", -1))
            if action in (0, 1, 2, 3):
                await self.human_move(action)
        elif kind == "train":
            if hub.train_state["running"]:
                await self.client.send({"type": "error", "message": "Training is already running."})
                return
            updates = min(max(int(msg.get("updates", 150)), 1), 1000)
            hub.train_task = asyncio.create_task(run_training(updates))
        elif kind == "stop_train":
            hub.stop_requested = True


@app.websocket("/ws/maze")
async def maze_endpoint(ws: WebSocket):
    """Q-learning maze page. Each browser tab gets its own maze and Q-table."""
    await ws.accept()
    client = Client(ws)
    session = MazeSession(client.send)
    try:
        await session.start()
        while True:
            await session.handle(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        session.stop()


@app.websocket("/ws/rlhf")
async def rlhf_endpoint(ws: WebSocket):
    """RLHF palette page. Each browser tab has its own picks, reward model and generator."""
    await ws.accept()
    client = Client(ws)
    session = RLHFSession(client.send)
    try:
        await session.start()
        while True:
            await session.handle(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        session.stop()


@app.websocket("/ws/vision")
async def vision_endpoint(ws: WebSocket):
    """Anomaly-search page. Watching is per tab; training is shared by all tabs."""
    await ws.accept()
    client = Client(ws)
    vision_hub.clients.add(client.send)
    session = VisionSession(client.send)
    try:
        await client.send(vision_hub.hello())
        while True:
            await session.handle(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        session.stop()
        vision_hub.clients.discard(client.send)


@app.get("/health")
def health():
    return {"ok": True, "model_available": hub.trained_net is not None,
            "training": hub.train_state}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    client = Client(ws)
    hub.clients.add(client)
    session = Session(client)
    try:
        await client.send(hub.hello())
        while True:
            await session.handle(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        session.stop()
        hub.clients.discard(client)
