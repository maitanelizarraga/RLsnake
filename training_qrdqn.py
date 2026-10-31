import argparse
import os

import gymnasium as gym
import snake_env  # noqa: F401
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from curriculum_callback import CurriculumCallback

ALGORITHM = "QRDQN"

DEFAULT_THRESHOLDS = {2: 1.0, 3: 2.0, 4: 3.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--models-dir", default=f"models/{ALGORITHM}")
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    start_level = 1 if args.curriculum else 4
    env = gym.make("Snake-v0", render_mode="rgb_array",
                   curriculum_level=start_level)
    eval_env = gym.make("Snake-v0", render_mode="rgb_array",
                        curriculum_level=4)

    callbacks = [EvalCallback(
        eval_env,
        best_model_save_path=f"{args.models_dir}/best_model",
        log_path=f"./logs/eval/{ALGORITHM}",
        eval_freq=args.eval_freq,
        deterministic=True,
    )]
    if args.curriculum:
        callbacks.append(CurriculumCallback(
            thresholds=DEFAULT_THRESHOLDS, patience=20,
        ))

    model = QRDQN("MlpPolicy", env, verbose=1, tensorboard_log="logs")
    model.learn(total_timesteps=args.timesteps,
                tb_log_name=ALGORITHM, callback=callbacks)
    model.save(f"{args.models_dir}/{args.timesteps}")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward (level 4): {mean_reward} ± {std_reward}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()