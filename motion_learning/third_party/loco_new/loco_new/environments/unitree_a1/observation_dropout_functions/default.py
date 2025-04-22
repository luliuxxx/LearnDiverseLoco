import numpy as np

from loco_new.environments import observation_indices as obs_idx


class DefaultObservationDropout:
    def __init__(self, env, dynamic_dropout_chance=0.005):
        self.env = env
        self.dynamic_dropout_chance = dynamic_dropout_chance

    def modify_observation(self, obs):
        for joint_indices in self.env.joint_order:
            obs[joint_indices] = np.where(self.env.np_rng.uniform(0, 1) < self.dynamic_dropout_chance, 0.0, obs[joint_indices])
        for foot_indices in self.env.feet_order:
            obs[foot_indices] = np.where(self.env.np_rng.uniform(0, 1) < self.dynamic_dropout_chance, 0.0, obs[foot_indices])

        return obs
