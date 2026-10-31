from stable_baselines3.common.env_checker import check_env
from snake_env.env import SnakeEnv

env = SnakeEnv(curriculum_level=4)
check_env(env)
print("SnakeEnv OK")