import numpy as np


class RandomCommands:
    def __init__(self, env,
                 min_x_velocity=-1.0, max_x_velocity=1.0,
                 min_y_velocity=-1.0, max_y_velocity=1.0,
                 min_yaw_velocity=-1.0, max_yaw_velocity=1.0,
                 zero_clip=False, zero_clip_threshold=0.1):
        self.env = env
        self.min_x_velocity = min_x_velocity
        self.max_x_velocity = max_x_velocity
        self.min_y_velocity = min_y_velocity
        self.max_y_velocity = max_y_velocity
        self.min_yaw_velocity = min_yaw_velocity
        self.max_yaw_velocity = max_yaw_velocity
        self.zero_clip = zero_clip
        self.zero_clip_threshold = zero_clip_threshold

    def get_next_command(self):
        goal_x_velocity = self.env.np_rng.uniform(self.min_x_velocity, self.max_x_velocity)
        goal_y_velocity = self.env.np_rng.uniform(self.min_y_velocity, self.max_y_velocity)
        goal_yaw_velocity = self.env.np_rng.uniform(self.min_yaw_velocity, self.max_yaw_velocity)

        if self.zero_clip:
            if np.abs(goal_x_velocity) < self.zero_clip_threshold:
                goal_x_velocity = 0.0
            if np.abs(goal_y_velocity) < self.zero_clip_threshold:
                goal_y_velocity = 0.0
            if np.abs(goal_yaw_velocity) < self.zero_clip_threshold:
                goal_yaw_velocity = 0.0

        return goal_x_velocity, goal_y_velocity, goal_yaw_velocity

    def setup(self):
        return

    def step(self, obs, reward, absorbing, info):
        return
