"""
Colour palettes for the RLHF demo.

A palette is 5 colours picked from a fixed set of swatches. The generator
(a PPO policy) builds a palette one colour at a time, like a language model
writing one token at a time.

    SWATCHES      the "vocabulary": 12 hues x 3 lightness x 2 saturation + 3 neutrals
    features()    8 readable traits of a palette (warmth, lightness, contrast, ...)
    PaletteEnv    Gymnasium env: 5 steps, reward from a reward function at the end
"""

import colorsys
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PALETTE_LEN = 5


# ---------------------------------------------------------------------- #
# The vocabulary of colours
# ---------------------------------------------------------------------- #
def _make_swatches():
    out = []  # (hex, hue 0..1, saturation 0..1, lightness 0..1, neutral?)
    for i in range(12):
        hue = i / 12
        for light in (0.3, 0.52, 0.76):
            for sat in (0.38, 0.85):
                r, g, b = colorsys.hls_to_rgb(hue, light, sat)
                out.append((_hex(r, g, b), hue, sat, light, False))
    for light in (0.12, 0.5, 0.94):
        out.append((_hex(light, light, light), 0.0, 0.0, light, True))
    return out


def _hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


SWATCHES = _make_swatches()
N_SWATCHES = len(SWATCHES)  # 75


# ---------------------------------------------------------------------- #
# Readable traits the reward model scores
# ---------------------------------------------------------------------- #
FEATURES = [
    ("light", "Light colours", "Average lightness of the five colours."),
    ("contrast", "Strong contrast", "Gap between the lightest and darkest colour."),
    ("vivid", "Vivid colours", "Average saturation. Low means muted and dusty."),
    ("warm", "Warm, not cool", "Reds, oranges and yellows score high; blues low."),
    ("variety", "Many different hues", "Low when all colours come from one hue family."),
    ("gradient", "Ordered light to dark", "High when lightness steps steadily in one direction."),
    ("neutral", "Greys and neutrals", "Share of white, grey or black colours."),
    ("repeats", "Repeated colours", "Share of colours that appear more than once."),
]
N_FEATURES = len(FEATURES)


def features(palette):
    """Map a palette (list of swatch ids) to 8 numbers, mostly in 0..1 (warmth is -1..1)."""
    sw = [SWATCHES[i] for i in palette]
    lights = np.array([s[3] for s in sw])
    sats = np.array([s[2] for s in sw])
    hues = np.array([s[1] for s in sw])
    neutral = np.array([s[4] for s in sw], dtype=float)
    chroma = sats * (1 - neutral)

    # warmth: cos distance from orange (30 deg), weighted by how colourful each one is
    warm = float(np.sum(chroma * np.cos(2 * math.pi * (hues - 30 / 360))) / max(chroma.sum(), 1e-6))
    # hue variety: 1 - length of the mean hue vector (0 = one hue, ~1 = spread around the wheel)
    if chroma.sum() > 1e-6:
        vx = np.sum(chroma * np.cos(2 * math.pi * hues)) / chroma.sum()
        vy = np.sum(chroma * np.sin(2 * math.pi * hues)) / chroma.sum()
        variety = 1 - math.hypot(vx, vy)
    else:
        variety = 0.0
    steps = np.sign(np.diff(lights))
    gradient = abs(steps.sum()) / (PALETTE_LEN - 1)
    repeats = 1 - len(set(palette)) / len(palette)

    return np.array(
        [
            lights.mean(),
            lights.max() - lights.min(),
            sats.mean(),
            warm,
            variety,
            gradient,
            neutral.mean(),
            repeats,
        ],
        dtype=np.float64,
    )


def hexes(palette):
    return [SWATCHES[i][0] for i in palette]


def random_palette(rng):
    return [int(x) for x in rng.integers(N_SWATCHES, size=PALETTE_LEN)]


# A pretend person, used by "answer for me" and for testing.
SIMULATED_TASTE = np.array([1.2, 0.6, -0.4, 1.6, -0.6, 0.8, -0.8, -2.0])
SIMULATED_DESCRIPTION = "likes warm, light palettes ordered light to dark, and dislikes repeats and greys"


def simulated_choice(feat_a, feat_b, rng, temperature=0.35):
    """Bradley-Terry choice: prefers higher hidden score, with some noise like a real person."""
    diff = (SIMULATED_TASTE @ (feat_a - feat_b)) / temperature
    p_a = 1 / (1 + math.exp(-diff))
    return "a" if rng.random() < p_a else "b"


# ---------------------------------------------------------------------- #
# Environment: pick 5 colours, get scored by the reward model at the end
# ---------------------------------------------------------------------- #
_SW_FEATS = np.array(
    [[1.0, math.sin(2 * math.pi * s[1]) * s[2], math.cos(2 * math.pi * s[1]) * s[2], s[2], s[3]] for s in SWATCHES],
    dtype=np.float32,
)


class PaletteEnv(gym.Env):
    """
    Observation: for each of the 5 slots [filled?, hue sin, hue cos, saturation, lightness]
                 plus a one-hot of which slot is next  -> 30 numbers
    Action:      which swatch to put in the next slot (75 choices)
    Reward:      0 until the palette is complete, then reward_fn(palette)
    """

    def __init__(self, reward_fn):
        super().__init__()
        self.reward_fn = reward_fn
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(PALETTE_LEN * 5 + PALETTE_LEN,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_SWATCHES)
        self.palette = []

    def _obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        for i, sw in enumerate(self.palette):
            obs[i * 5 : i * 5 + 5] = _SW_FEATS[sw]
        if len(self.palette) < PALETTE_LEN:
            obs[PALETTE_LEN * 5 + len(self.palette)] = 1.0
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.palette = []
        return self._obs(), {}

    def step(self, action):
        self.palette.append(int(action))
        done = len(self.palette) == PALETTE_LEN
        reward = float(self.reward_fn(self.palette)) if done else 0.0
        info = {"result": "done", "palette": list(self.palette)} if done else {}
        return self._obs(), reward, done, False, info
