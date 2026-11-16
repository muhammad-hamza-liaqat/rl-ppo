"""
WebSocket session for the RLHF page (/rlhf in the web UI).

Client -> server
    {"type": "pair"}                         new pair of palettes to compare
    {"type": "choose", "pick": "a"|"b"|"skip"}
    {"type": "simulate", "n": 20}            a simulated rater answers n pairs
    {"type": "tune", "updates": 30, "beta": 0.1}
    {"type": "stop_tune"}
    {"type": "gallery"}                      fresh samples: original vs tuned generator
    {"type": "reset"}                        forget picks, reward model and tuning

Server -> client
    state, pair, reward_model, tune_start, tune_progress, tune_done, gallery, error
"""

import asyncio

import numpy as np

from palettes import (FEATURES, SIMULATED_DESCRIPTION, SIMULATED_TASTE, features, hexes,
                      random_palette, simulated_choice)
from rlhf import RewardModel, RLHFTrainer

MIN_PICKS = 8  # comparisons needed before tuning


class RLHFSession:
    def __init__(self, send):
        self.send = send
        self.rng = np.random.default_rng()
        self.task: asyncio.Task | None = None
        self.reset_all()

    def reset_all(self):
        self.pairs = []          # (winner features, loser features)
        self.n_skipped = 0
        self.simulated = 0       # how many picks came from the simulated rater
        self.rm = RewardModel()
        self.trainer = RLHFTrainer(self.rm, beta=0.1)  # holds both the tuned and the original generator
        self.tuned_updates = 0
        self.history = []        # tuning chart points
        self.current = None      # the pair on screen: (palette_a, palette_b)

    # ------------------------------------------------------------------ #
    def palette_msg(self, p):
        f = features(p)
        trained = bool(self.pairs)
        return {"colors": hexes(p), "score": round(self.rm.score(f), 3) if trained else None}

    def sample_pair(self):
        """Pairs come from the current generator: the original one until it has been tuned."""
        while True:
            a, b = self.trainer.sample(2)
            if a != b:
                return a, b

    def rm_msg(self):
        # the best and worst of 400 random palettes, according to the reward model
        extremes = None
        if self.pairs:
            pool = [random_palette(self.rng) for _ in range(400)]
            pool.sort(key=lambda p: self.rm.score(features(p)))
            extremes = {"best": self.palette_msg(pool[-1]), "worst": self.palette_msg(pool[0])}
        return {
            "type": "reward_model",
            "weights": [round(float(w), 3) for w in self.rm.w],
            "agreement": self.rm.agreement(self.pairs),
            "picks": len(self.pairs),
            "skipped": self.n_skipped,
            "simulated": self.simulated,
            "hidden_taste": SIMULATED_TASTE.tolist() if self.simulated else None,
            "extremes": extremes,
        }

    def state_msg(self):
        return {
            "type": "state",
            "features": [{"key": k, "label": l, "description": d} for k, l, d in FEATURES],
            "min_picks": MIN_PICKS,
            "simulated_description": SIMULATED_DESCRIPTION,
            "beta": self.trainer.beta,
            "tuned_updates": self.tuned_updates,
            "history": self.history,
            "tuning": bool(self.task and not self.task.done()),
        }

    async def send_pair(self):
        self.current = self.sample_pair()
        a, b = self.current
        await self.send({
            "type": "pair",
            "a": self.palette_msg(a),
            "b": self.palette_msg(b),
            "source": "tuned" if self.tuned_updates else "original",
        })

    def record(self, a, b, pick):
        fa, fb = features(a), features(b)
        self.pairs.append((fa, fb) if pick == "a" else (fb, fa))

    def refit(self):
        self.rm.fit(self.pairs)
        self.rm.calibrate(self.rng, n=600)

    # ------------------------------------------------------------------ #
    async def tune(self, updates, beta):
        t = self.trainer
        t.beta = beta
        await self.send({"type": "tune_start", "updates": updates, "beta": beta})
        stopped = False
        try:
            for i in range(1, updates + 1):
                await asyncio.to_thread(t.train_step)
                eps = t.finished_episodes[-256:]
                t.finished_episodes.clear()
                self.tuned_updates += 1
                ps = t.last_palettes
                point = {
                    "update": self.tuned_updates,
                    "score": float(np.mean([r for r, _ in eps])) if eps else 0.0,
                    "kl": t.last_kl,
                    "unique": len({tuple(p) for p in ps}) / max(len(ps), 1),
                }
                self.history.append(point)
                await self.send({
                    "type": "tune_progress",
                    "point": point,
                    "done_in_run": i,
                    "updates": updates,
                    "samples": [self.palette_msg(p) for p in ps[:6]],
                })
        except asyncio.CancelledError:
            stopped = True
        await self.send({"type": "tune_done", "stopped": stopped, "tuned_updates": self.tuned_updates})
        await self.send_gallery()
        await self.send_pair()

    async def send_gallery(self):
        t = self.trainer
        original = t.sample(8, net=t.reference)
        tuned = t.sample(8) if self.tuned_updates else []
        avg = lambda ps: float(np.mean([self.rm.reward(p) for p in ps])) if ps and self.pairs else None
        await self.send({
            "type": "gallery",
            "original": [self.palette_msg(p) for p in original],
            "tuned": [self.palette_msg(p) for p in tuned],
            "original_avg": avg(original),
            "tuned_avg": avg(tuned),
        })

    # ------------------------------------------------------------------ #
    async def start(self):
        await self.send(self.state_msg())
        await self.send(self.rm_msg())
        await self.send_pair()
        await self.send_gallery()

    def busy(self):
        return self.task is not None and not self.task.done()

    def stop(self):
        if self.busy():
            self.task.cancel()

    async def handle(self, msg):
        kind = msg.get("type")
        if kind in ("pair", "choose", "simulate", "gallery") and self.busy():
            await self.send({"type": "error", "code": "busy",
                             "message": "Picking is paused while PPO tunes the generator."})
            return
        if kind == "pair":
            await self.send_pair()
        elif kind == "choose":
            pick = msg.get("pick")
            if self.current and pick in ("a", "b"):
                self.record(*self.current, pick)
                self.refit()
                await self.send(self.rm_msg())
            elif pick == "skip":
                self.n_skipped += 1
            await self.send_pair()
        elif kind == "simulate":
            n = min(max(int(msg.get("n", 20)), 1), 200)
            for _ in range(n):
                a, b = self.sample_pair()
                self.record(a, b, simulated_choice(features(a), features(b), self.rng))
            self.simulated += n
            self.refit()
            await self.send(self.rm_msg())
            await self.send_pair()
        elif kind == "tune":
            if self.busy():
                await self.send({"type": "error", "message": "Tuning is already running."})
            elif len(self.pairs) < MIN_PICKS:
                await self.send({"type": "error",
                                 "message": f"Make at least {MIN_PICKS} picks first so the reward model has something to go on."})
            else:
                updates = min(max(int(msg.get("updates", 30)), 1), 200)
                beta = min(max(float(msg.get("beta", 0.1)), 0.0), 2.0)
                self.task = asyncio.create_task(self.tune(updates, beta))
        elif kind == "stop_tune":
            self.stop()
        elif kind == "gallery":
            await self.send_gallery()
        elif kind == "reset":
            self.stop()
            self.reset_all()
            await self.start()
