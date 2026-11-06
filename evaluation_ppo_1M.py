"""Ver jugar a PPO entrenado con 1.000.000 de pasos (baseline y curriculum).

Uso (desde la carpeta del proyecto):
    uv run python evaluation_ppo_1M.py
"""
import gymnasium as gym
import snake_env  # noqa: F401
from stable_baselines3 import PPO

CARPETA = "resultados_PPO_1M"   # carpeta que creó main_ppo.py
SEMILLA = 0                     # semilla del modelo que quieres ver (0, 1 o 2)
NUM_EPISODES = 5                # partidas que se ven de cada variante

MODELOS = {
    "baseline": f"{CARPETA}/models/PPO_s{SEMILLA}/best_model/best_model",
    "curriculum": f"{CARPETA}/models/PPO_curriculum_s{SEMILLA}/best_model/best_model",
}

# Entorno con ventana
env = gym.make("Snake-v0", render_mode="human", render_fps=10)

try:
    for variante, model_path in MODELOS.items():
        print(f"\n=== PPO {variante} (semilla {SEMILLA}) -> {model_path} ===")
        model = PPO.load(model_path)

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

            print(f"Episodio {ep + 1}: manzanas = {info['apples']}, retorno = {total_reward:.1f}")
finally:
    env.close()