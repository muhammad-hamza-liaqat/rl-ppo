"""
Watch the trained agent play Treasure Grid in the terminal.

    python play.py                    # 5 episodes with the trained model
    python play.py --episodes 10 --delay 0.2
    python play.py --random           # compare: an untrained, random agent
"""

import argparse
import os
import time

import torch

from ppo import load_policy
from treasure_env import ACTION_NAMES, TreasureGridEnv


def show(env, header, footer=""):
    os.system("cls" if os.name == "nt" else "clear")
    print(header)
    print()
    print(env.render())
    print()
    print(footer)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_treasure.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--random", action="store_true", help="ignore the model, act randomly")
    args = parser.parse_args()

    net = None if args.random else load_policy(args.model)
    env = TreasureGridEnv()
    who = "RANDOM agent" if args.random else "TRAINED agent"
    wins = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total, done, step = 0.0, False, 0
        header = f"{who} | episode {ep}/{args.episodes}   (A=agent  T=treasure  X=trap)"
        show(env, header, "start")
        time.sleep(args.delay)

        while not done:
            if net is None:
                action = env.action_space.sample()
            else:
                probs = torch.softmax(net.actor(torch.as_tensor(obs)), dim=-1)
                action = probs.argmax().item()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            step += 1
            done = terminated or truncated
            show(env, header, f"step {step:2d} | move: {ACTION_NAMES[action]:5s} | reward so far {total:+.2f}")
            time.sleep(args.delay)

        result = info["result"]
        wins += result == "win"
        label = {"win": "FOUND THE TREASURE!", "trap": "fell into a trap", "timeout": "ran out of time"}[result]
        print(f"\n>>> {label}  (total reward {total:+.2f})")
        time.sleep(1.2)

    print(f"\n{who} won {wins}/{args.episodes} episodes.")


if __name__ == "__main__":
    main()
