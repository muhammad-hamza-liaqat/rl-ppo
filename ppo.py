"""
PPO (Proximal Policy Optimization) from scratch in PyTorch.

The loop, in one sentence:
    play with the current policy -> score every action ("advantage")
    -> nudge the policy toward good actions, but never too far (the clip).

Pieces in this file:
    ActorCritic  — the neural network ("brain"): actor picks moves, critic predicts value
    PPOConfig    — all hyperparameters in one place
    PPO          — collect_rollout() + compute_gae() + update()
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------------- #
# 1. The network
# ---------------------------------------------------------------------- #
def layer(in_dim, out_dim, std=np.sqrt(2)):
    """Linear layer with orthogonal init — a standard PPO trick that helps stability."""
    lin = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.constant_(lin.bias, 0.0)
    return lin


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.hidden = hidden
        # Actor: observation -> one score (logit) per action
        self.actor = nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(),
            layer(hidden, hidden), nn.Tanh(),
            layer(hidden, n_actions, std=0.01),  # tiny init => near-uniform start (explores)
        )
        # Critic: observation -> single number "how good is this state?"
        self.critic = nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(),
            layer(hidden, hidden), nn.Tanh(),
            layer(hidden, 1, std=1.0),
        )

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, action=None):
        """Sample an action (or score a given one). Returns action, log-prob, entropy, value."""
        dist = Categorical(logits=self.actor(obs))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(obs)


# ---------------------------------------------------------------------- #
# 2. Hyperparameters
# ---------------------------------------------------------------------- #
@dataclass
class PPOConfig:
    rollout_steps: int = 2048   # steps of experience collected before each update
    gamma: float = 0.99         # discount: how much future reward matters
    gae_lambda: float = 0.95    # GAE smoothing (bias/variance trade-off)
    clip_eps: float = 0.2       # PPO clip range: policy ratio kept in [0.8, 1.2]
    lr: float = 3e-4            # learning rate
    epochs: int = 10            # passes over each rollout
    minibatch_size: int = 256
    value_coef: float = 0.5     # weight of critic loss
    entropy_coef: float = 0.01  # weight of exploration bonus
    max_grad_norm: float = 0.5  # gradient clipping


# ---------------------------------------------------------------------- #
# 3. The algorithm
# ---------------------------------------------------------------------- #
class PPO:
    def __init__(self, env, config=PPOConfig(), seed=0, hidden=64):
        self.env = env
        self.cfg = config
        torch.manual_seed(seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.net = ActorCritic(obs_dim, n_actions, hidden=hidden)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)

        self.obs, _ = env.reset(seed=seed)
        self.ep_reward = 0.0
        self.finished_episodes = []  # (total_reward, result) for logging

    # ---- Step A: play -------------------------------------------------- #
    @torch.no_grad()
    def collect_rollout(self):
        n = self.cfg.rollout_steps
        obs_dim = self.env.observation_space.shape[0]
        buf = {
            "obs": np.zeros((n, obs_dim), dtype=np.float32),
            "actions": np.zeros(n, dtype=np.int64),
            "logprobs": np.zeros(n, dtype=np.float32),
            "values": np.zeros(n, dtype=np.float32),
            "rewards": np.zeros(n, dtype=np.float32),
            "dones": np.zeros(n, dtype=np.float32),  # 1 = episode ended after this step
        }

        for t in range(n):
            obs_t = torch.as_tensor(self.obs)
            action, logprob, _, value = self.net.act(obs_t)
            next_obs, reward, terminated, truncated, info = self.env.step(action.item())
            self.ep_reward += reward

            # Time-limit cut-off isn't a real "game over": the agent would have kept
            # earning value, so bootstrap with the critic's estimate of where it was.
            if truncated and not terminated:
                reward += self.cfg.gamma * self.net.value(torch.as_tensor(next_obs)).item()

            buf["obs"][t] = self.obs
            buf["actions"][t] = action.item()
            buf["logprobs"][t] = logprob.item()
            buf["values"][t] = value.item()
            buf["rewards"][t] = reward
            buf["dones"][t] = float(terminated or truncated)

            if terminated or truncated:
                self.finished_episodes.append((self.ep_reward, info.get("result")))
                self.ep_reward = 0.0
                next_obs, _ = self.env.reset()
            self.obs = next_obs

        last_value = self.net.value(torch.as_tensor(self.obs)).item()
        return buf, last_value

    # ---- Step B: score every action ----------------------------------- #
    def compute_gae(self, buf, last_value):
        """
        Advantage = "how much better was this action than the critic expected?"
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)          (one-step surprise)
            A_t     = delta_t + gamma * lambda * A_{t+1}          (smoothed over future)
        Returns = A_t + V(s_t) are the targets the critic learns to predict.
        """
        rewards, values, dones = buf["rewards"], buf["values"], buf["dones"]
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * not_done - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * not_done * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    # ---- Step C: learn ------------------------------------------------- #
    def update(self, buf, advantages, returns):
        obs = torch.as_tensor(buf["obs"])
        actions = torch.as_tensor(buf["actions"])
        old_logprobs = torch.as_tensor(buf["logprobs"])
        advantages = torch.as_tensor(advantages)
        returns = torch.as_tensor(returns)

        n = len(obs)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}

        for _ in range(self.cfg.epochs):
            for idx in torch.randperm(n).split(self.cfg.minibatch_size):
                _, new_logprob, entropy, new_value = self.net.act(obs[idx], actions[idx])

                # Normalise advantages per minibatch (keeps step sizes sane)
                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # ratio = pi_new(a|s) / pi_old(a|s)
                log_ratio = new_logprob - old_logprobs[idx]
                ratio = log_ratio.exp()

                # ---- The PPO clipped objective ----
                # Take the more pessimistic of (unclipped, clipped) so the policy
                # gets no extra credit for moving outside [1-eps, 1+eps].
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = ((new_value - returns[idx]) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy_bonus.item())
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clip_frac"].append(((ratio - 1).abs() > self.cfg.clip_eps).float().mean().item())

        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ---- One full iteration ------------------------------------------- #
    def train_step(self):
        buf, last_value = self.collect_rollout()
        advantages, returns = self.compute_gae(buf, last_value)
        return self.update(buf, advantages, returns)

    # ---- Save / load --------------------------------------------------- #
    def save(self, path):
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                # plain ints: gymnasium gives numpy ints, which torch.load(weights_only=True) rejects
                "obs_dim": int(self.env.observation_space.shape[0]),
                "n_actions": int(self.env.action_space.n),
                "hidden": int(self.net.hidden),
            },
            path,
        )


def load_policy(path):
    ckpt = torch.load(path, weights_only=True)
    net = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"], hidden=ckpt.get("hidden", 64))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net
