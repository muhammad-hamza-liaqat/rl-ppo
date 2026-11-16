"""
PPO with several environments running side by side ("vectorised" PPO).

Same algorithm as ppo.py. The only change is in collecting experience: instead
of one game at a time, N games advance together and the network evaluates all
N observations in one batch. For small networks on a CPU, that per-call
overhead dominates, so this is several times faster.

Buffers are shaped (T steps, N envs). GAE runs along T for each env separately,
then everything is flattened for the usual PPO update.
"""

import numpy as np
import torch
from torch.distributions import Categorical

from ppo import PPO, PPOConfig


class VecPPO(PPO):
    def __init__(self, make_env, num_envs=16, config=PPOConfig(), seed=0, hidden=64):
        self.envs = [make_env() for _ in range(num_envs)]
        super().__init__(self.envs[0], config, seed=seed, hidden=hidden)
        self.num_envs = num_envs
        self.obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])
        self.ep_rewards = np.zeros(num_envs)

    @torch.no_grad()
    def collect_rollout(self):
        n_envs = self.num_envs
        steps = self.cfg.rollout_steps // n_envs
        obs_dim = self.obs.shape[1]
        buf = {
            "obs": np.zeros((steps, n_envs, obs_dim), dtype=np.float32),
            "actions": np.zeros((steps, n_envs), dtype=np.int64),
            "logprobs": np.zeros((steps, n_envs), dtype=np.float32),
            "values": np.zeros((steps, n_envs), dtype=np.float32),
            "rewards": np.zeros((steps, n_envs), dtype=np.float32),
            "dones": np.zeros((steps, n_envs), dtype=np.float32),
        }
        for t in range(steps):
            obs_t = torch.as_tensor(self.obs)
            dist = Categorical(logits=self.net.actor(obs_t))
            actions = dist.sample()
            buf["obs"][t] = self.obs
            buf["actions"][t] = actions.numpy()
            buf["logprobs"][t] = dist.log_prob(actions).numpy()
            buf["values"][t] = self.net.value(obs_t).numpy()

            next_obs = np.empty_like(self.obs)
            for i, env in enumerate(self.envs):
                o, r, terminated, truncated, info = env.step(int(actions[i]))
                self.ep_rewards[i] += r
                if truncated and not terminated:  # time limit: bootstrap from the critic
                    r += self.cfg.gamma * self.net.value(torch.as_tensor(o)).item()
                if terminated or truncated:
                    self.finished_episodes.append((self.ep_rewards[i], info.get("result")))
                    self.ep_rewards[i] = 0.0
                    o, _ = env.reset()
                buf["rewards"][t, i] = r
                buf["dones"][t, i] = float(terminated or truncated)
                next_obs[i] = o
            self.obs = next_obs

        last_values = self.net.value(torch.as_tensor(self.obs)).numpy()
        return buf, last_values

    def compute_gae(self, buf, last_values):
        rewards, values, dones = buf["rewards"], buf["values"], buf["dones"]
        steps = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(steps)):
            next_value = last_values if t == steps - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * not_done - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * not_done * gae
            advantages[t] = gae
        return advantages, advantages + values

    def train_step(self):
        buf, last_values = self.collect_rollout()
        advantages, returns = self.compute_gae(buf, last_values)
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in buf.items()}
        return self.update(flat, advantages.reshape(-1), returns.reshape(-1))
