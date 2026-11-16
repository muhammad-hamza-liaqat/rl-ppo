"""
RLHF in miniature: Reinforcement Learning from Human Feedback, on colour palettes.

The same three steps used to fine-tune chat assistants:

  1. Collect comparisons: show a person two outputs, they pick the better one.
  2. Train a reward model so that   P(A preferred over B) = sigmoid(r(A) - r(B))
     (the Bradley-Terry model). Here r is a weighted sum of 8 readable traits,
     so you can read exactly what it learned; real reward models are big networks.
  3. Tune the generator with PPO to score well with the reward model, while a
     KL penalty keeps it close to the original generator:

        reward per step = -beta * (log pi(a|s) - log pi_original(a|s))
        reward at end   += normalised reward-model score

     Without the KL penalty the generator tends to collapse onto one palette
     the reward model happens to love ("reward hacking").
"""

import copy
from collections import deque

import numpy as np
import torch
from torch.distributions import Categorical

from palettes import N_FEATURES, PaletteEnv, features, random_palette
from ppo import PPO, PPOConfig


# ---------------------------------------------------------------------- #
# Reward model
# ---------------------------------------------------------------------- #
class RewardModel:
    def __init__(self, l2=0.01):
        self.w = np.zeros(N_FEATURES)
        self.l2 = l2
        self.mean, self.std = 0.0, 1.0  # for normalising PPO rewards

    def score(self, feat):
        return float(self.w @ feat)

    def fit(self, pairs, steps=1500, lr=0.5):
        """pairs: list of (features_of_winner, features_of_loser). Logistic regression on differences."""
        if not pairs:
            return
        d = np.array([fw - fl for fw, fl in pairs])
        w = self.w.copy()
        for _ in range(steps):
            z = d @ w
            # loss = mean(-log sigmoid(z)) + l2 * |w|^2
            grad = -(d * (1 / (1 + np.exp(z)))[:, None]).mean(0) + 2 * self.l2 * w
            w -= lr * grad
        self.w = w

    def agreement(self, pairs):
        if not pairs:
            return None
        return float(np.mean([self.w @ (fw - fl) > 0 for fw, fl in pairs]))

    def calibrate(self, rng, n=1000):
        """Score random palettes so PPO rewards are roughly mean 0, std 1."""
        scores = np.array([self.score(features(random_palette(rng))) for _ in range(n)])
        self.mean, self.std = float(scores.mean()), float(scores.std() + 1e-6)

    def reward(self, palette):
        return float(np.clip((self.score(features(palette)) - self.mean) / self.std, -4, 4))


# ---------------------------------------------------------------------- #
# PPO with a KL penalty to the original generator
# ---------------------------------------------------------------------- #
RLHF_CONFIG = PPOConfig(
    rollout_steps=1280,   # 256 palettes per update
    gamma=1.0,            # 5-step episodes: no need to discount
    gae_lambda=0.95,
    lr=1e-3,
    epochs=4,
    minibatch_size=256,
    entropy_coef=0.0,     # the KL penalty does this job
)


class RLHFTrainer(PPO):
    def __init__(self, reward_model: RewardModel, beta=0.05, seed=0):
        self.rm = reward_model
        env = PaletteEnv(reward_fn=lambda p: self.rm.reward(p))
        super().__init__(env, RLHF_CONFIG, seed=seed)
        self.reference = copy.deepcopy(self.net)  # the frozen "original" generator
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.beta = beta
        self.last_kl = 0.0
        self.last_palettes = []

    @torch.no_grad()
    def collect_rollout(self):
        """Same as PPO.collect_rollout, plus a per-step KL penalty against the reference policy."""
        n = self.cfg.rollout_steps
        obs_dim = self.env.observation_space.shape[0]
        buf = {
            "obs": np.zeros((n, obs_dim), dtype=np.float32),
            "actions": np.zeros(n, dtype=np.int64),
            "logprobs": np.zeros(n, dtype=np.float32),
            "values": np.zeros(n, dtype=np.float32),
            "rewards": np.zeros(n, dtype=np.float32),
            "dones": np.zeros(n, dtype=np.float32),
        }
        kls, palettes = [], []
        for t in range(n):
            obs_t = torch.as_tensor(self.obs)
            action, logprob, _, value = self.net.act(obs_t)
            ref_logprob = Categorical(logits=self.reference.actor(obs_t)).log_prob(action)
            kl = (logprob - ref_logprob).item()
            kls.append(kl)

            next_obs, reward, terminated, truncated, info = self.env.step(action.item())
            self.ep_reward += reward

            buf["obs"][t] = self.obs
            buf["actions"][t] = action.item()
            buf["logprobs"][t] = logprob.item()
            buf["values"][t] = value.item()
            buf["rewards"][t] = reward - self.beta * kl
            buf["dones"][t] = float(terminated or truncated)

            if terminated or truncated:
                self.finished_episodes.append((self.ep_reward, info.get("result")))
                palettes.append(info["palette"])
                self.ep_reward = 0.0
                next_obs, _ = self.env.reset()
            self.obs = next_obs

        # KL per palette = sum over its 5 choices
        self.last_kl = float(np.mean(kls) * 5)
        self.last_palettes = palettes
        last_value = self.net.value(torch.as_tensor(self.obs)).item()
        return buf, last_value

    @torch.no_grad()
    def sample(self, n, net=None, greedy=False):
        """Generate n palettes from the tuned (or another) policy."""
        net = net or self.net
        env = PaletteEnv(reward_fn=lambda p: 0.0)
        out = []
        for _ in range(n):
            obs, _ = env.reset()
            for _ in range(5):
                logits = net.actor(torch.as_tensor(obs))
                a = int(logits.argmax()) if greedy else int(Categorical(logits=logits).sample())
                obs, _, done, _, info = env.step(a)
            out.append(info["palette"])
        return out


if __name__ == "__main__":
    # Terminal demo with the simulated rater: does the tuned generator please it more?
    from palettes import SIMULATED_TASTE, simulated_choice

    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(40):
        a, b = random_palette(rng), random_palette(rng)
        fa, fb = features(a), features(b)
        pairs.append((fa, fb) if simulated_choice(fa, fb, rng) == "a" else (fb, fa))
    rm = RewardModel()
    rm.fit(pairs)
    rm.calibrate(rng)
    print("learned weights:", np.round(rm.w, 2))
    print("hidden taste:   ", SIMULATED_TASTE)
    print(f"agreement with the rater's picks: {rm.agreement(pairs):.0%}")

    trainer = RLHFTrainer(rm, beta=0.05)
    taste = lambda ps: np.mean([SIMULATED_TASTE @ features(p) for p in ps])
    print(f"hidden-taste score before: {taste(trainer.sample(300)):+.2f}")
    recent = deque(maxlen=256)
    for u in range(1, 41):
        stats = trainer.train_step()
        if u % 5 == 0:
            ps = trainer.last_palettes
            uniq = len({tuple(p) for p in ps}) / len(ps)
            print(f"update {u:2d} | rm reward {np.mean([r for r, _ in trainer.finished_episodes[-256:]]):+.2f} "
                  f"| kl {trainer.last_kl:.2f} | unique {uniq:.0%} | hidden taste {taste(ps):+.2f}")
        trainer.finished_episodes.clear()
