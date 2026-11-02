import gymnasium as gym
import snake_env
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import QRDQN

# Mapa algoritmo -> clase SB3
ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "QRDQN": QRDQN,
}

# Number of episodes to watch
NUM_EPISODES = 5

# Create environment with rendering
env = gym.make("Snake-v0", render_mode="human", render_fps=10)

try:
    for algo, algo_cls in ALGORITHMS.items():
        model_path = f"models/{algo}_s0/best_model/best_model"  # igual que benchmark.py (EvalCallback)
        print(f"\n=== {algo} -> {model_path} ===")

        # Load model
        model = algo_cls.load(model_path)

        for ep in range(NUM_EPISODES):
            obs, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                env.render()
finally:
    env.close()