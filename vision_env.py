"""
Anomaly Search — RL for image processing ("active vision").

Instead of looking at the whole image at once, the agent controls a square
search box. It starts covering the whole image, then moves and resizes the box
step by step until it says "found it". This is the idea behind papers such as
"Active Object Localization with Deep Reinforcement Learning" (Caicedo & Lazebnik, 2015).

The images are synthetic grayscale "scans": smooth background texture, a few
dim normal structures, and one bright, irregular anomaly to find.

    Observation (212 numbers, 0..1):
        fovea    12x12  sharp view of what is inside the box
        context   8x8   blurry view of the whole image (peripheral vision)
        box       3     centre x, centre y, size
        time      1     fraction of the step budget used

    Actions (7):
        0 up, 1 down, 2 left, 3 right, 4 bigger, 5 smaller, 6 found it

    Rewards:
        moves:     +0.1 if the box overlaps the anomaly better (IoU went up), -0.1 otherwise
        found it:  +1 + IoU if IoU >= 0.5 (episode ends), else -1 (episode ends)
        timeout after `max_steps`: episode cut off
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_NAMES = ["up", "down", "left", "right", "bigger", "smaller", "found it"]
FOVEA = 12
CONTEXT = 8
HIT_IOU = 0.5


# ---------------------------------------------------------------------- #
# Synthetic scans
# ---------------------------------------------------------------------- #
def _blob(yy, xx, cy, cx, ry, rx):
    return np.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2))


def make_scan(rng, size=64):
    """Returns (image float32 in 0..1, target box (cx, cy, half_size))."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size), dtype=np.float32)

    # smooth background texture
    for _ in range(6):
        cy, cx = rng.uniform(0, size, 2)
        r = rng.uniform(8, 20)
        img += rng.uniform(0.05, 0.15) * _blob(yy, xx, cy, cx, r, r)
    img += 0.12

    # normal structures: dim rings and soft elongated shapes
    for _ in range(rng.integers(2, 5)):
        cy, cx = rng.uniform(6, size - 6, 2)
        if rng.random() < 0.5:
            r = rng.uniform(4, 8)
            d = np.hypot(yy - cy, xx - cx)
            img += rng.uniform(0.18, 0.3) * np.exp(-(((d - r) / 1.3) ** 2))
        else:
            ry, rx = rng.uniform(2, 4), rng.uniform(5, 9)
            if rng.random() < 0.5:
                ry, rx = rx, ry
            img += rng.uniform(0.15, 0.28) * _blob(yy, xx, cy, cx, ry, rx)

    # the anomaly: bright, compact, irregular (2-3 overlapping blobs)
    r = rng.uniform(3, 6.5)
    cy, cx = rng.uniform(r + 4, size - r - 4, 2)
    for i in range(rng.integers(2, 4)):
        oy, ox = (rng.normal(0, r * 0.35, 2) if i else (0, 0))
        img += rng.uniform(0.55, 0.75) * _blob(yy, xx, cy + oy, cx + ox, r * rng.uniform(0.6, 1), r * rng.uniform(0.6, 1))

    img += rng.normal(0, 0.035, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    return img, (float(cx), float(cy), float(r * 1.5))


def iou(a, b):
    """IoU of two squares given as (cx, cy, half_size)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2], a[1] - a[2], a[0] + a[2], a[1] + a[2]
    bx0, by0, bx1, by1 = b[0] - b[2], b[1] - b[2], b[0] + b[2], b[1] + b[2]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (2 * a[2]) ** 2 + (2 * b[2]) ** 2 - inter
    return inter / union if union > 0 else 0.0


def resize_area(img, out):
    """Resample a 2-D array to out x out by bilinear sampling (cheap, no dependencies)."""
    h, w = img.shape
    ys = (np.arange(out) + 0.5) * h / out - 0.5
    xs = (np.arange(out) + 0.5) * w / out - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    a = img[y0][:, x0]
    b = img[y0][:, x1]
    c = img[y1][:, x0]
    d = img[y1][:, x1]
    return (a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + c * wy * (1 - wx) + d * wy * wx).astype(np.float32)


def block_mean(img, out):
    s = img.shape[0] // out
    return img[: s * out, : s * out].reshape(out, s, out, s).mean(axis=(1, 3))


# ---------------------------------------------------------------------- #
# Environment
# ---------------------------------------------------------------------- #
class VisionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, size=64, max_steps=40, min_half=3.0):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.min_half = min_half
        self.observation_space = spaces.Box(0.0, 1.0, shape=(FOVEA * FOVEA + CONTEXT * CONTEXT + 4,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

    # box helpers ------------------------------------------------------- #
    def _clamp(self, cx, cy, s):
        s = float(np.clip(s, self.min_half, self.size / 2))
        cx = float(np.clip(cx, s, self.size - s))
        cy = float(np.clip(cy, s, self.size - s))
        return cx, cy, s

    def fovea(self):
        cx, cy, s = self.box
        x0, x1 = int(round(cx - s)), int(round(cx + s))
        y0, y1 = int(round(cy - s)), int(round(cy + s))
        crop = self.image[max(y0, 0): max(y1, y0 + 1), max(x0, 0): max(x1, x0 + 1)]
        return resize_area(crop, FOVEA)

    def context(self):
        return self._context

    def _obs(self):
        cx, cy, s = self.box
        return np.concatenate([
            self.fovea().ravel(),
            self._context.ravel(),
            np.array([cx / self.size, cy / self.size, s / (self.size / 2), self.steps / self.max_steps], dtype=np.float32),
        ]).astype(np.float32)

    def current_iou(self):
        return iou(self.box, self.target)

    # gym API ----------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.image, self.target = make_scan(self.np_random, self.size)
        self._context = block_mean(self.image, CONTEXT).astype(np.float32)
        # start: the whole image (the agent always starts here). A person playing gets a
        # smaller box in the middle via options={"start_half": 12}, so the arrows move it at once.
        start_half = (options or {}).get("start_half", self.size / 2)
        self.box = self._clamp(self.size / 2, self.size / 2, start_half)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        cx, cy, s = self.box
        before = self.current_iou()
        self.steps += 1
        info = {}

        if action == 6:  # found it
            if before >= HIT_IOU:
                reward, result = 1.0 + before, "found"
            else:
                reward, result = -1.0, "wrong"
            info.update(result=result, iou=before)
            return self._obs(), reward, True, False, info

        step = max(1.0, 0.25 * s)
        if action == 0:
            cy -= step
        elif action == 1:
            cy += step
        elif action == 2:
            cx -= step
        elif action == 3:
            cx += step
        elif action == 4:
            s *= 1.25
        elif action == 5:
            s *= 0.8
        self.box = self._clamp(cx, cy, s)
        after = self.current_iou()
        reward = 0.1 if after > before + 1e-9 else -0.1

        truncated = self.steps >= self.max_steps
        if truncated:
            info.update(result="timeout", iou=after)
        return self._obs(), reward, False, truncated, info


if __name__ == "__main__":
    env = VisionEnv()
    obs, _ = env.reset(seed=0)
    print("obs", obs.shape, "target", np.round(env.target, 1))
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        done = term or trunc
    print("random agent:", info, f"reward {total:+.2f}")
