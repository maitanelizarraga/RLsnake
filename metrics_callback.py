import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """Guarda en TensorBoard las manzanas por episodio y el nivel del curriculum."""

    def __init__(self):
        super().__init__()
        self.apples = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.apples.append(info["apples"])
                self.logger.record("snake/apples_mean", np.mean(self.apples[-100:]))
                self.logger.record("snake/curriculum_level", info["curriculum_level"])
        return True
