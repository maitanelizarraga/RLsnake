# Import necessary libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
from collections import deque

# --- Constants ---
SNAKE_LEN_GOAL = 30
STACKED_ACTIONS = 5

tableSizeObs = 500
tableSize = 500
halfTable = int(tableSize / 2)

MAX_EPISODE_STEPS = 500          # upper bound (kept in sync with Snake-v0 registration)
MAX_STEPS_SINCE_APPLE = 100.0    # saturation of the "hunger" observation feature

# Curriculum: harder as level increases.
#   max_k     -> radius (in grid cells) where the apple can spawn.
#                None = full board (default behaviour).
#   max_steps -> internal truncation threshold for the episode.
CURRICULUM_LEVELS = {
    1: {"max_k": 1,    "max_steps": 100},
    2: {"max_k": 2,    "max_steps": 200},
    3: {"max_k": 5,    "max_steps": 350},
    4: {"max_k": None, "max_steps": 500},
}
DEFAULT_CURRICULUM_LEVEL = 4     # non-curriculum runs use the hardest level


# --- Helpers ---
def collision_with_apple(apple_position, score, action_size=1,
                         max_k_override=None, snake_position=None):
    apple_position = random_apple_position(
        action_size,
        max_k_override=max_k_override,
        snake_position=snake_position,
    )
    score += 1
    return apple_position, score


def random_apple_position(action_size=1, max_k_override=None, snake_position=None):
    step = 10 * action_size
    max_k = (halfTable - step) // step
    if max_k_override is not None:
        max_k = min(max_k, max_k_override)
    max_k = max(max_k, 0)

    # Retry to avoid spawning the apple on top of the snake body
    pos = None
    for _ in range(100):
        pos = [
            halfTable + step * random.randint(-max_k, max_k),
            halfTable + step * random.randint(-max_k, max_k),
        ]
        if snake_position is None or list(pos) not in snake_position:
            return pos
    return pos  # fallback


def collision_with_boundaries(snake_head):
    return (snake_head[0] >= tableSize or snake_head[0] < 0
            or snake_head[1] >= tableSize or snake_head[1] < 0)


def collision_with_self(snake_position):
    return snake_position[0] in snake_position[1:]


class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, render_mode=None, action_size=1, render_fps=10,
                 curriculum_level=DEFAULT_CURRICULUM_LEVEL):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-tableSizeObs,
            high=tableSizeObs,
            shape=(7 + STACKED_ACTIONS,),
            dtype=np.float64,
        )

        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        self.prev_reward = 0
        self.total_reward = 0
        self.score = 0
        self.max_score = 0

        self.render_mode = render_mode
        self.render_fps = render_fps
        self.action_size = action_size

        # Curriculum state
        self.curriculum_level = DEFAULT_CURRICULUM_LEVEL
        self.current_max_k = None
        self.current_max_steps = MAX_EPISODE_STEPS
        self.set_curriculum_level(curriculum_level, verbose=False)

        # Time counters
        self.step_count = 0
        self.steps_since_last_apple = 0

    # --- Curriculum control ---
    def set_curriculum_level(self, level, verbose=True):
        if level not in CURRICULUM_LEVELS:
            raise ValueError(
                f"Unknown curriculum level {level}. Valid: {sorted(CURRICULUM_LEVELS)}"
            )
        cfg = CURRICULUM_LEVELS[level]
        self.curriculum_level = level
        self.current_max_k = cfg["max_k"]
        self.current_max_steps = cfg["max_steps"]
        if verbose:
            print(f"[SnakeEnv] curriculum level -> {level} "
                  f"(max_k={self.current_max_k}, max_steps={self.current_max_steps})")

    # --- Gym API ---
    def step(self, action):
        self.step_count += 1
        self.prev_actions.append(action)

        prev_distance = np.linalg.norm(
            np.array(self.snake_head) - np.array(self.apple_position)
        )
        self._take_action(action)

        reward = 0.0

        # Eat apple?
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(
                self.apple_position, self.score, self.action_size,
                max_k_override=self.current_max_k,
                snake_position=self.snake_position,
            )
            self.snake_position.insert(0, list(self.snake_head))
            reward += 10
            self.steps_since_last_apple = 0
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()
            self.steps_since_last_apple += 1

        # Collisions
        terminated = (collision_with_boundaries(self.snake_head)
                      or collision_with_self(self.snake_position))
        # Internal truncation at curriculum level (outer TimeLimit wraps at 500)
        truncated = self.step_count >= self.current_max_steps

        if terminated:
            reward -= 10
        else:
            current_distance = np.linalg.norm(
                np.array(self.snake_head) - np.array(self.apple_position)
            )
            reward += 1 if current_distance < prev_distance else -1

        # --- Reward hardening: step cost + hunger penalty ---
        # If you have NOT applied the hardening yet, replace this line with:
        #     reward -= 0.1
        reward -= 0.5 + 0.01 * self.steps_since_last_apple

        info = {
            "apples": self.score,
            "steps": self.step_count,
            "curriculum_level": self.curriculum_level,
        }

        head_x, head_y = self.snake_head
        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y
        time_progress = self.step_count / self.current_max_steps
        hunger = min(self.steps_since_last_apple / MAX_STEPS_SINCE_APPLE, 1.0)

        observation = np.array(
            [head_x, head_y, apple_delta_x, apple_delta_y, snake_length,
             time_progress, hunger]
            + list(self.prev_actions),
            dtype=np.float64,
        )
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')

        step = 10 * self.action_size
        self.snake_position = [
            [halfTable, halfTable],
            [halfTable - step, halfTable],
            [halfTable - 2 * step, halfTable],
        ]
        self.apple_position = random_apple_position(
            self.action_size,
            max_k_override=self.current_max_k,
            snake_position=self.snake_position,
        )

        if self.score > self.max_score:
            self.max_score = self.score
            print(f"New maximum score registered: {self.max_score}")

        self.score = 0
        self.snake_head = [halfTable, halfTable]
        self.direction = 1
        self.prev_reward = 0
        self.total_reward = 0
        self.step_count = 0
        self.steps_since_last_apple = 0

        self.prev_actions = deque(maxlen=STACKED_ACTIONS)
        for _ in range(STACKED_ACTIONS):
            self.prev_actions.append(-1)

        head_x, head_y = self.snake_head
        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        observation = np.array(
            [head_x, head_y, apple_delta_x, apple_delta_y, snake_length,
             0.0, 0.0]
            + list(self.prev_actions),
            dtype=np.float64,
        )
        return observation, {}

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode
        self._update_ui()
        if mode == 'human':
            cv2.imshow('Snake Game', self.img)
            cv2.waitKey(int(1000 / self.render_fps))
        elif mode == 'rgb_array':
            return self.img.copy()

    def close(self):
        cv2.destroyAllWindows()

    def _update_ui(self):
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255), -1,
        )
        for i, pos in enumerate(self.snake_position):
            cv2.rectangle(
                self.img,
                (pos[0], pos[1]), (pos[0] + 10, pos[1] + 10),
                (0, 255, 255) if i == 0 else (0, 255, 0), -1,
            )

    def _take_action(self, action):
        if action == 0 and self.direction == 1:
            action = 1
        elif action == 1 and self.direction == 0:
            action = 0
        elif action == 2 and self.direction == 3:
            action = 3
        elif action == 3 and self.direction == 2:
            action = 2
        self.direction = action
        if action == 0:
            self.snake_head[0] -= 10 * self.action_size
        elif action == 1:
            self.snake_head[0] += 10 * self.action_size
        elif action == 2:
            self.snake_head[1] += 10 * self.action_size
        elif action == 3:
            self.snake_head[1] -= 10 * self.action_size