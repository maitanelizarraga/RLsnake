import argparse
import os

import gymnasium as gym
import snake_env  # noqa: F401 - registers Snake-v0
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from curriculum_callback import CurriculumCallback
from metrics_callback import MetricsCallback

ALGORITHM = "PPO"

DEFAULT_THRESHOLDS = {2: 1.0, 3: 2.0, 4: 3.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", action="store_true",
                    help="Enable curriculum learning (start at level 1, raise on performance).")
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    variant = "curriculum" if args.curriculum else "baseline"
    if args.models_dir is None:
        args.models_dir = f"models/{ALGORITHM}_curriculum_s{args.seed}" if args.curriculum else f"models/{ALGORITHM}_s{args.seed}"

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    start_level = 1 if args.curriculum else 4
    env = gym.make("Snake-v0", render_mode="rgb_array",
                   curriculum_level=start_level)
    eval_env = gym.make("Snake-v0", render_mode="rgb_array",
                        curriculum_level=4)  # always evaluate on hardest level

    callbacks = [EvalCallback(
        eval_env,
        best_model_save_path=f"{args.models_dir}/best_model",
        log_path=f"./logs/eval/{ALGORITHM}",
        eval_freq=args.eval_freq,
        n_eval_episodes=20,
        deterministic=True,
    ), MetricsCallback()]
    if args.curriculum:
        callbacks.append(CurriculumCallback(
            thresholds=DEFAULT_THRESHOLDS, patience=20,
        ))

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs" , seed=args.seed)
    model.learn(total_timesteps=args.timesteps,
                tb_log_name=f"{ALGORITHM}_{variant}_s{args.seed}", callback=callbacks)
    model.save(f"{args.models_dir}/{args.timesteps}")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Mean reward (level 4): {mean_reward} ± {std_reward}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()