"""
WebSocket session for the anomaly-search page (/vision in the web UI).

Client -> server
    {"type": "watch", "agent": "trained" | "random"}
    {"type": "human_reset"} / {"type": "human_move", "action": 0..6}
    {"type": "speed", "delay_ms": 200}
    {"type": "stop"}
    {"type": "train", "updates": 150} / {"type": "stop_train"}    shared by all tabs

Server -> client
    hello, frame, train_start, train_progress, train_preview, train_done, error

Every frame carries the box, the 12x12 fovea the agent sees, its move
probabilities and value. The image itself (and the 8x8 context view) is only
sent on the first frame of each episode.
"""

import asyncio
import base64
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from ppo import PPOConfig, load_policy
from ppo_vec import VecPPO
from vision_env import ACTION_NAMES, FOVEA, HIT_IOU, VisionEnv

MODEL_PATH = Path(__file__).with_name("ppo_vision.pt")
HIDDEN = 128
N_ACTIONS = len(ACTION_NAMES)
UNIFORM = [1 / N_ACTIONS] * N_ACTIONS


@torch.no_grad()
def think(net, obs):
    if net is None:
        return UNIFORM, None
    t = torch.as_tensor(obs)
    return torch.softmax(net.actor(t), dim=-1).tolist(), net.value(t).item()


def r3(x):
    return round(float(x), 3)


def make_frame(env, obs, net, *, source, episode, step, total, action=None, reward=None,
               done=False, result=None, first=False):
    probs, value = (None, None) if done else think(net, obs)
    msg = {
        "type": "frame",
        "source": source,  # trained | random | human | preview
        "episode": episode,
        "step": step,
        "size": env.size,
        "box": [r3(v) for v in env.box],
        "target": [r3(v) for v in env.target],
        "iou": r3(env.current_iou()),
        "fovea": [r3(v) for v in obs[: FOVEA * FOVEA]],
        "probs": probs,
        "value": value,
        "action": action,
        "reward": reward,
        "total": r3(total),
        "done": done,
        "result": result,  # found | wrong | timeout
        "max_steps": env.max_steps,
    }
    if first:
        pixels = (env.image * 255).astype(np.uint8).tobytes()
        msg["image"] = base64.b64encode(pixels).decode()
        msg["context"] = [r3(v) for v in env.context().ravel()]
    return msg


def play_episode_frames(net, source="preview", episode=1, seed=None):
    """One greedy episode, every frame (used for training previews)."""
    env = VisionEnv()
    obs, _ = env.reset(seed=seed)
    frames = [make_frame(env, obs, net, source=source, episode=episode, step=0, total=0.0, first=True)]
    total, done = 0.0, False
    while not done:
        action = int(np.argmax(think(net, obs)[0]))
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        done = term or trunc
        frames.append(make_frame(env, obs, net, source=source, episode=episode, step=env.steps, total=total,
                                 action=action, reward=r3(reward), done=done, result=info.get("result")))
    return frames


@torch.no_grad()
def evaluate(net, episodes=300, seed=777):
    env = VisionEnv()
    results, steps = Counter(), []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(net.actor(torch.as_tensor(obs)).argmax().item())
            done = term or trunc
        results[info["result"]] += 1
        steps.append(env.steps)
    out = {k: results[k] / episodes for k in ("found", "wrong", "timeout")}
    out["steps"] = float(np.mean(steps))
    return out


# ---------------------------------------------------------------------- #
class VisionHub:
    def __init__(self):
        self.clients = set()  # send callables
        self.trained_net = load_policy(MODEL_PATH) if MODEL_PATH.exists() else None
        self.train_state = {"running": False, "update": 0, "updates": 0}
        self.history, self.before, self.after = [], None, None
        self.stop_requested = False
        self.task = None

    async def broadcast(self, msg):
        for send in list(self.clients):
            try:
                await send(msg)
            except Exception:
                self.clients.discard(send)

    def hello(self):
        return {"type": "hello", "model_available": self.trained_net is not None, "training": self.train_state,
                "history": self.history, "before": self.before, "after": self.after,
                "actions": ACTION_NAMES, "hit_iou": HIT_IOU}


hub = VisionHub()


async def run_training(updates):
    # 16 scans searched side by side: same PPO, much faster on a CPU
    agent = VecPPO(VisionEnv, num_envs=16, config=PPOConfig(), seed=0, hidden=HIDDEN)
    hub.stop_requested = False
    hub.history, hub.after = [], None
    hub.train_state = {"running": True, "update": 0, "updates": updates}
    try:
        hub.before = await asyncio.to_thread(evaluate, agent.net, 200)
        await hub.broadcast({"type": "train_start", "updates": updates, "before": hub.before})
        recent = deque(maxlen=300)
        start, done_updates = time.time(), 0
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
                "found": sum(res == "found" for _, res in recent) / n,
                "entropy": stats["entropy"],
                "kl": stats["approx_kl"],
                "elapsed": time.time() - start,
            }
            hub.history.append(point)
            hub.train_state = {"running": True, "update": update, "updates": updates}
            await hub.broadcast({"type": "train_progress", "point": point, "updates": updates})
            frames = await asyncio.to_thread(play_episode_frames, agent.net, "preview", update)
            await hub.broadcast({"type": "train_preview", "update": update, "frames": frames})

        stopped = done_updates < updates
        if not stopped:
            agent.save(MODEL_PATH)
            hub.trained_net = load_policy(MODEL_PATH)
            hub.after = await asyncio.to_thread(evaluate, agent.net, 300)
        await hub.broadcast({"type": "train_done", "stopped": stopped, "updates_done": done_updates,
                             "before": hub.before, "after": hub.after,
                             "model_available": hub.trained_net is not None})
    except Exception as e:
        await hub.broadcast({"type": "error", "message": f"Training failed: {e}"})
    finally:
        hub.train_state = {**hub.train_state, "running": False}


# ---------------------------------------------------------------------- #
class VisionSession:
    def __init__(self, send):
        self.send = send
        self.task = None
        self.delay = 0.25
        self.env = VisionEnv()
        self.human = None
        self.human_episodes = 0

    def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None

    async def watch(self, agent):
        net = hub.trained_net if agent == "trained" else None
        if agent == "trained" and net is None:
            await self.send({"type": "error", "code": "no_model", "message": "No trained model yet. Train one first."})
            return
        env, episode = self.env, 0
        rng = np.random.default_rng()
        while True:
            episode += 1
            obs, _ = env.reset()
            total, done = 0.0, False
            await self.send(make_frame(env, obs, net, source=agent, episode=episode, step=0, total=0.0, first=True))
            await asyncio.sleep(self.delay * 2)
            while not done:
                if net is None:
                    action = int(rng.integers(N_ACTIONS))
                else:
                    action = int(np.argmax(think(net, obs)[0]))
                obs, reward, term, trunc, info = env.step(action)
                total += reward
                done = term or trunc
                await self.send(make_frame(env, obs, net, source=agent, episode=episode, step=env.steps,
                                           total=total, action=action, reward=r3(reward), done=done,
                                           result=info.get("result")))
                await asyncio.sleep(self.delay)
            await asyncio.sleep(max(1.5, self.delay * 4))

    async def human_reset(self):
        self.stop()
        self.human_episodes += 1
        obs, _ = self.env.reset()
        self.human = {"obs": obs, "total": 0.0, "done": False}
        await self.send(make_frame(self.env, obs, hub.trained_net, source="human", episode=self.human_episodes,
                                   step=0, total=0.0, first=True))

    async def human_move(self, action):
        h = self.human
        if h is None or h["done"]:
            return
        obs, reward, term, trunc, info = self.env.step(action)
        h["obs"], h["total"], h["done"] = obs, h["total"] + reward, term or trunc
        await self.send(make_frame(self.env, obs, hub.trained_net, source="human", episode=self.human_episodes,
                                   step=self.env.steps, total=h["total"], action=action, reward=r3(reward),
                                   done=h["done"], result=info.get("result")))

    async def handle(self, msg):
        kind = msg.get("type")
        if kind == "watch":
            self.stop()
            self.task = asyncio.create_task(self.watch(msg.get("agent", "trained")))
        elif kind == "stop":
            self.stop()
        elif kind == "speed":
            self.delay = min(max(float(msg.get("delay_ms", 250)) / 1000, 0.02), 2.0)
        elif kind == "human_reset":
            await self.human_reset()
        elif kind == "human_move":
            a = int(msg.get("action", -1))
            if 0 <= a < N_ACTIONS:
                await self.human_move(a)
        elif kind == "train":
            if hub.train_state["running"]:
                await self.send({"type": "error", "message": "Training is already running."})
                return
            updates = min(max(int(msg.get("updates", 150)), 1), 1000)
            hub.task = asyncio.create_task(run_training(updates))
        elif kind == "stop_train":
            hub.stop_requested = True
