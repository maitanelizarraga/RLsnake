from gymnasium.envs.registration import register

register(
    id="Snake-v0",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
)

# NOTE: original "Snake-v0-step5" used action_size=2 (step of 20 px, not 5).
# Kept as alias for backward compatibility; new name reflects the real step.
register(
    id="Snake-v0-step20",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
    kwargs={"action_size": 2},
)
register(
    id="Snake-v0-step5",  # deprecated alias (real step is 20 px)
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
    kwargs={"action_size": 2},
)

register(
    id="Snake-v1",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=100,
)

register(
    id="Snake-v2",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
    reward_threshold=30,
)

# Curriculum variants: Snake-v0-curriculum-L1 ... L4
for _lvl in (1, 2, 3, 4):
    register(
        id=f"Snake-v0-curriculum-L{_lvl}",
        entry_point="snake_env.env:SnakeEnv",
        max_episode_steps=500,
        kwargs={"curriculum_level": _lvl},
    )