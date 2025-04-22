import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco


class RandomInitialState:
    def __init__(self, env,
                 roll_angle_factor=0.0625,
                 pitch_angle_factor=0.0625,
                 yaw_angle_factor=1.0,
                 nominal_joint_position_factor=0.5,
                 joint_velocity_factor=0.5,
                 max_linear_velocity=0.5,
                 max_angular_velocity=0.5,
        ):
        self.env = env
        self.roll_angle_factor = roll_angle_factor
        self.pitch_angle_factor = pitch_angle_factor
        self.yaw_angle_factor = yaw_angle_factor
        self.nominal_joint_position_factor = nominal_joint_position_factor
        self.joint_velocity_factor = joint_velocity_factor
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

    def step(self, obs, reward, absorbing, info):
        return

    def setup(self):
        generate_new_init_position = True

        while generate_new_init_position:
            roll_angle = self.env.np_rng.uniform(-np.pi * self.roll_angle_factor, np.pi * self.roll_angle_factor)
            pitch_angle = self.env.np_rng.uniform(-np.pi * self.pitch_angle_factor, np.pi * self.pitch_angle_factor)
            yaw_angle = self.env.np_rng.uniform(-np.pi * self.yaw_angle_factor, np.pi * self.yaw_angle_factor)
            quaternion = R.from_euler('xyz', [roll_angle, pitch_angle, yaw_angle]).as_quat()
            joint_positions = self.env.nominal_joint_positions * self.env.np_rng.uniform(self.nominal_joint_position_factor, 1 + self.nominal_joint_position_factor, size=(self.env.nominal_joint_positions.shape[0],))
            joint_velocities = self.env.max_joint_velocities * self.env.np_rng.uniform(-self.joint_velocity_factor, self.joint_velocity_factor)
            linear_velocities = self.env.np_rng.uniform(-self.max_linear_velocity, self.max_linear_velocity, size=(3,))
            angular_velocities = self.env.np_rng.uniform(-self.max_angular_velocity, self.max_angular_velocity, size=(3,))

            qpos = [
                0.0, 0.0, self.env.initial_drop_height + self.env.terrain_function.center_height,
                # quaternion[3], quaternion[0], quaternion[1], quaternion[2],
                1.0, 0.0, 0.0, 0.0,
                *joint_positions
            ]

            qvel = [
                *linear_velocities,
                *angular_velocities,
                *joint_velocities
            ]

            self.env.data.qpos[:] = qpos
            self.env.data.qvel[:] = qvel
            mujoco.mj_forward(self.env.model, self.env.data)
            
            generate_new_init_position = np.any(self.env.data.geom_xpos[self.env.foot_geom_indices, 2] - self.env.terrain_function.center_height < 0)
        
        return qpos, qvel
