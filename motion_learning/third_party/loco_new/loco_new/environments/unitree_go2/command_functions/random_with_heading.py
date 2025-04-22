import numpy as np


class RandomWithHeadingCommands:
    def __init__(self, env,
                 min_x_velocity=-1.0, max_x_velocity=1.0,
                 min_y_velocity=-1.0, max_y_velocity=1.0,
                 min_yaw_velocity=-1.0, max_yaw_velocity=1.0,
                 zero_clip=False, zero_clip_threshold=0.1,
                 new_command_probability=0.002):
        self.env = env
        self.min_x_velocity = min_x_velocity
        self.max_x_velocity = max_x_velocity
        self.min_y_velocity = min_y_velocity
        self.max_y_velocity = max_y_velocity
        self.min_yaw_velocity = min_yaw_velocity
        self.max_yaw_velocity = max_yaw_velocity
        self.zero_clip = zero_clip
        self.zero_clip_threshold = zero_clip_threshold
        self.new_command_probability = new_command_probability

        self.goal_x_velocity = 0.0
        self.goal_y_velocity = 0.0
        self.goal_global_heading = 0.0

    def get_next_command(self):
        if self.env.np_rng.uniform() < self.new_command_probability:
            self.goal_global_heading = self.env.np_rng.uniform(-np.pi, np.pi)
            self.goal_x_velocity = self.env.np_rng.uniform(self.min_x_velocity, self.max_x_velocity)
            self.goal_y_velocity = self.env.np_rng.uniform(self.min_y_velocity, self.max_y_velocity)
            if self.zero_clip:
                if np.abs(self.goal_x_velocity) < self.zero_clip_threshold:
                    self.goal_x_velocity = 0.0
                if np.abs(self.goal_y_velocity) < self.zero_clip_threshold:
                    self.goal_y_velocity = 0.0

        projected_forward = self.env.orientation_quat.apply(np.array([1.0, 0.0, 0.0]))
        current_global_heading = np.arctan2(projected_forward[1], projected_forward[0])
        goal_yaw_velocity = np.clip(self.goal_global_heading - current_global_heading, self.min_yaw_velocity, self.max_yaw_velocity)

        if self.zero_clip and np.abs(goal_yaw_velocity) < self.zero_clip_threshold:
            goal_yaw_velocity = 0.0

        return self.goal_x_velocity, self.goal_y_velocity, goal_yaw_velocity

    def setup(self):
        return

    def step(self, obs, reward, absorbing, info):
        return
