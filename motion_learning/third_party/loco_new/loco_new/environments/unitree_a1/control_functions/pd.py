import numpy as np


class PDControl:
    def __init__(self, env, control_frequency_hz=50, p_gain=20, d_gain=0.5, scaling_factor=0.25):
        self.env = env
        self.control_frequency_hz = control_frequency_hz
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.scaling_factor = scaling_factor
        self.extrinsic_p_gain_factor = np.ones(12)
        self.extrinsic_d_gain_factor = np.ones(12)
        self.extrinsic_motor_strength_factor = np.ones(12)
        self.extrinsic_position_offset = np.zeros(12)

    def process_action(self, action):
        scaled_action = action * self.scaling_factor
        target_joint_positions = self.env.nominal_joint_positions + scaled_action
        torques = self.p_gain * self.extrinsic_p_gain_factor * (target_joint_positions - self.env.data.qpos[7:] + self.extrinsic_position_offset) \
                  - self.d_gain * self.extrinsic_d_gain_factor * self.env.data.qvel[6:]
        
        return torques * self.extrinsic_motor_strength_factor
