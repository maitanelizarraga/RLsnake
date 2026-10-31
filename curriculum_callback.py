"""Curriculum learning callback for SnakeEnv.

Raises the environment difficulty when the agent reaches a performance
threshold on the current level. Every env in the VecEnv must implement
`set_curriculum_level(level)`.

Thresholds example:
    thresholds={2: 1.0, 3: 2.0, 4: 3.0}
means:
    - start at level 1
    - advance to level 2 when mean apples/20 episodes >= 1.0
    - advance to level 3 when mean apples/20 episodes >= 2.0
    - advance to level 4 when mean apples/20 episodes >= 3.0
"""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, thresholds: dict[int, float], patience: int = 20,
                 verbose: int = 1):
        super().__init__(verbose=verbose)
        self.thresholds = dict(thresholds)
        self.patience = patience
        self.episode_apples: deque[int] = deque(maxlen=patience)
        self._levels_sorted = sorted(self.thresholds)
        self.current_level = self._levels_sorted[0]

    def _on_training_start(self) -> None:
        self.training_env.env_method(
            "set_curriculum_level", self.current_level, verbose=False
        )
        if self.verbose:
            print(f"[Curriculum] starting at level {self.current_level}")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                self.episode_apples.append(int(info.get("apples", 0)))
        self._maybe_upgrade()
        return True

    def _maybe_upgrade(self) -> None:
        if len(self.episode_apples) < self.patience:
            return
        mean_apples = float(np.mean(self.episode_apples))
        next_level = self._next_level()
        if next_level is None:
            return
        if mean_apples >= self.thresholds[next_level]:
            self.current_level = next_level
            self.episode_apples.clear()
            self.training_env.env_method(
                "set_curriculum_level", next_level, verbose=False
            )
            if self.verbose:
                print(f"[Curriculum] mean apples {mean_apples:.2f} >= "
                      f"{self.thresholds[next_level]:.2f} -> level {next_level}")

    def _next_level(self) -> int | None:
        idx = self._levels_sorted.index(self.current_level)
        if idx + 1 >= len(self._levels_sorted):
            return None
        return self._levels_sorted[idx + 1]