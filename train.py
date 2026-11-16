"""
Train a PPO agent on Treasure Grid.

    python train.py                 # default: 150 updates (~300k steps)
    python train.py --updates 300   # train longer
"""

import argparse
import time
from collections import Counter, deque

import torch

from ppo import PPO, PPOConfig
from treasure_env import TreasureGridEnv


@torch.no_grad()
def evaluate(net, episodes=500, seed=12345):
    """Play with the greedy policy (always pick the most likely action)."""
    env = TreasureGridEnv()
    results = Counter()
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            action = net.actor(torch.as_tensor(obs)).argmax().item()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results[info["result"]] += 1
    return {k: results[k] / episodes for k in ("win", "trap", "timeout")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="ppo_treasure.pt")
    args = parser.parse_args()

    env = TreasureGridEnv()
    agent = PPO(env, PPOConfig(), seed=args.seed)

    # Before training: how does a random (untrained) policy do?
    before = evaluate(agent.net)
    print(f"Untrained agent  -> win {before['win']:.0%}  trap {before['trap']:.0%}  timeout {before['timeout']:.0%}\n")

    recent = deque(maxlen=200)  # last 200 episodes for smoothed logging
    start = time.time()
    for update in range(1, args.updates + 1):
        stats = agent.train_step()

        recent.extend(agent.finished_episodes)
        agent.finished_episodes.clear()

        if update % 10 == 0 or update == 1:
            avg_reward = sum(r for r, _ in recent) / len(recent)
            win_rate = sum(res == "win" for _, res in recent) / len(recent)
            trap_rate = sum(res == "trap" for _, res in recent) / len(recent)
            print(
                f"Update {update:4d} | steps {update * agent.cfg.rollout_steps:7d} | "
                f"avg reward {avg_reward:+.2f} | win {win_rate:4.0%} | trap {trap_rate:4.0%} | "
                f"entropy {stats['entropy']:.2f} | kl {stats['approx_kl']:.4f} | "
                f"{time.time() - start:5.0f}s"
            )

    agent.save(args.out)
    after = evaluate(agent.net)
    print(f"\nTrained agent    -> win {after['win']:.0%}  trap {after['trap']:.0%}  timeout {after['timeout']:.0%}")
    print(f"Model saved to {args.out}.  Watch it play:  python play.py")


if __name__ == "__main__":
    main()
