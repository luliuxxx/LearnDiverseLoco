import numpy as np
from loco_new.environments import observation_indices as obs_idx
from scipy.spatial.transform import Rotation as R

class ImitationTask:
    def __init__(self, env,
                 curriculum_steps=800e6,
                 tracking_xy_velocity_command_coeff=2.0, tracking_yaw_velocity_command_coeff=1.0,
                 xy_tracking_temperature=0.25, yaw_tracking_temperature=0.25,
                 z_velocity_coeff=2e0, pitch_roll_vel_coeff=5e-2, pitch_roll_pos_coeff=2e-1, joint_nominal_diff_coeff=0.0,
                 joint_position_limit_coeff=1e1, soft_joint_position_limit=0.9,
                 joint_velocity_coeff=0.0, joint_acceleration_coeff=2.5e-7, joint_torque_coeff=2e-4, action_rate_coeff=1e-2,
                 collision_coeff=1e0, base_height_coeff=3e1, nominal_trunk_z=0.303, air_time_coeff=1e-1, air_time_max=0.5, symmetry_air_coeff=0.5):
        self.env = env
        self.tracking_xy_velocity_command_coeff = 0.1
        self.tracking_yaw_velocity_command_coeff = 0.1
        self.curriculum_steps = curriculum_steps
        self.xy_tracking_temperature = 1/2.0
        self.yaw_tracking_temperature = 1/0.2
        self.z_velocity_coeff = z_velocity_coeff 
        self.pitch_roll_vel_coeff = pitch_roll_vel_coeff 
        self.pitch_roll_pos_coeff = pitch_roll_pos_coeff 
        self.joint_nominal_diff_coeff = joint_nominal_diff_coeff 
        self.joint_position_limit_coeff = joint_position_limit_coeff 
        self.soft_joint_position_limit = soft_joint_position_limit
        self.joint_velocity_coeff = joint_velocity_coeff 
        self.joint_acceleration_coeff = joint_acceleration_coeff * self.env.dt
        self.joint_torque_coeff = joint_torque_coeff *self.env.dt
        self.action_rate_coeff = action_rate_coeff *self.env.dt
        self.collision_coeff = collision_coeff *self.env.dt
        self.base_height_coeff = base_height_coeff 
        self.nominal_trunk_z = nominal_trunk_z
        self.air_time_coeff = air_time_coeff 
        self.air_time_max = air_time_max
        self.symmetry_air_coeff = symmetry_air_coeff 

        self.time_since_last_touchdown_fr = 0
        self.time_since_last_touchdown_fl = 0
        self.time_since_last_touchdown_rr = 0
        self.time_since_last_touchdown_rl = 0
        self.prev_joint_vel = None

    def init(self):
        self.joint_limits = self.env.model.jnt_range[1:].copy()
        joint_limits_midpoint = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
        joint_limits_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        self.joint_limits[:, 0] = joint_limits_midpoint - joint_limits_range / 2 * self.soft_joint_position_limit
        self.joint_limits[:, 1] = joint_limits_midpoint + joint_limits_range / 2 * self.soft_joint_position_limit

    def setup(self):
        self.time_since_last_touchdown_fr = 0
        self.time_since_last_touchdown_fl = 0
        self.time_since_last_touchdown_rr = 0
        self.time_since_last_touchdown_rl = 0
        self.prev_joint_vel = np.zeros(self.env.model.nu)
        self.sum_tracking_performance_percentage = 0.0

    def step(self, action):
        self.time_since_last_touchdown_fr = 0 if self.env.check_collision("floor", "FR_foot") else self.time_since_last_touchdown_fr + self.env.dt
        self.time_since_last_touchdown_fl = 0 if self.env.check_collision("floor", "FL_foot") else self.time_since_last_touchdown_fl + self.env.dt
        self.time_since_last_touchdown_rr = 0 if self.env.check_collision("floor", "RR_foot") else self.time_since_last_touchdown_rr + self.env.dt
        self.time_since_last_touchdown_rl = 0 if self.env.check_collision("floor", "RL_foot") else self.time_since_last_touchdown_rl + self.env.dt
        self.prev_joint_vel = np.array(self.env.data.qvel[6:])

    def reward_and_info(self, info, done):
        total_timesteps = self.env.total_timesteps * self.env.total_nr_envs
        curriculum_coeff = min(total_timesteps / self.curriculum_steps, 1.0)
        
        if self.env.eval or self.env.mode == "test":
            curriculum_coeff = 1.0
        obs = self.env.get_observation()

        # Tracking velocity command reward
        current_global_linear_velocity = self.env.data.qvel[:3]
        current_local_linear_velocity = self.env.orientation_quat_inv.apply(current_global_linear_velocity)
        desired_global_linear_velocity, desired_local_ang_velocity = self.env.track_trunk_lin_vel.copy(), self.env.track_trunk_ang_vel.copy()
        desired_global_linear_velocity_xy = desired_global_linear_velocity[:2]
  
        current_local_angular_velocity = obs[obs_idx.TRUNK_ANGULAR_VELOCITIES]
        desired_local_yaw_velocity = desired_local_ang_velocity[2] 
        yaw_velocity_difference_norm = np.sum(np.square(current_local_angular_velocity - desired_local_ang_velocity))
        xy_velocity_difference_norm = np.sum(np.square(current_global_linear_velocity - desired_global_linear_velocity))
        tracking_xy_velocity_command_reward = self.tracking_xy_velocity_command_coeff * np.exp((-xy_velocity_difference_norm*2.0) -yaw_velocity_difference_norm*0.2)

        ref_root_pose_reward, root_pose_diff_norm = self._calc_reward_trunk_pose(obs)
        ref_joint_pos_reward, joint_pos_diff_norm = self._calc_reward_joint_position(obs)
        ref_joint_vel_reward, joint_vel_diff_norm = self._calc_reward_joint_velocity(obs)
        ref_toes_pos_reward, toes_pos_diff_norm = self._calc_reward_toes_position(obs)

        trunk_z = obs[obs_idx.HEIGHT]
        # Joint acceleration reward
        acceleration_norm = np.sum(np.square((self.prev_joint_vel - np.array(self.env.data.qvel[6:])) / self.env.dt))
        acceleration_reward = curriculum_coeff * self.joint_acceleration_coeff * -acceleration_norm

        # Joint torque reward
        torque_norm = np.sum(np.square(np.array(self.env.data.qfrc_actuator[6:])))
        torque_reward = curriculum_coeff * self.joint_torque_coeff * -torque_norm

        # Action rate reward
        action_rate_norm = np.sum(np.square(self.env.current_action - self.env.last_action))
        action_rate_reward = curriculum_coeff * self.action_rate_coeff * -action_rate_norm

        # Collision reward
        collisions = self.env.check_any_collision_for_all([
            "FR_calf0", "FL_calf0", "RR_calf0", "RL_calf0",
        ])
        trunk_collision = 1 if self.env.check_any_collision(["trunk_1", "trunk_2", "trunk_3"]) else 0
        nr_collisions = sum(collisions.values()) + trunk_collision
        collision_reward = curriculum_coeff * self.collision_coeff * -nr_collisions
        power = np.sum(abs(self.env.current_torques) * abs(self.env.data.qvel[6:]))
        # Total reward

        tracking_reward = ref_toes_pos_reward  + ref_joint_pos_reward + ref_root_pose_reward + tracking_xy_velocity_command_reward + ref_joint_vel_reward
        reward_penalty =  acceleration_reward + torque_reward + action_rate_reward + \
                                 collision_reward 
    
        reward = tracking_reward + reward_penalty
        reward = max(reward, 0.0)


        # More logging metrics
        mass_of_robot = np.sum(self.env.model.body_mass)
        gravity = -self.env.model.opt.gravity[2]
        velocity = np.linalg.norm(current_local_linear_velocity)
        cost_of_transport = power / (mass_of_robot * gravity * velocity)
        froude_number = velocity ** 2 / (gravity * trunk_z)
        current_global_velocities = np.array([current_global_linear_velocity[0], current_global_linear_velocity[1], current_local_angular_velocity[2]])
        desired_global_velocities = np.array([desired_global_linear_velocity_xy[0], desired_global_linear_velocity_xy[1], desired_local_yaw_velocity])
        tracking_performance_percentage = max(np.mean(1 - (np.abs(current_global_velocities - desired_global_velocities) / np.abs(desired_global_velocities))), 0.0)
        self.sum_tracking_performance_percentage += tracking_performance_percentage
        if done:
            episode_tracking_performance_percentage = self.sum_tracking_performance_percentage / self.env.horizon

        # Info
        info[f"reward/torque"] = torque_reward
        info[f"reward/acceleration"] = acceleration_reward
        info[f"reward/action_rate"] = action_rate_reward
        info[f"reward/collision"] = collision_reward
        info["env_info/target_x_vel"] = desired_global_linear_velocity_xy[0]
        info["env_info/target_y_vel"] = desired_global_linear_velocity_xy[1]
        info["env_info/target_yaw_vel"] = desired_local_yaw_velocity
        info["env_info/current_x_vel"] = current_local_linear_velocity[0]
        info["env_info/current_y_vel"] = current_local_linear_velocity[1]
        info["env_info/current_yaw_vel"] = current_local_angular_velocity[2]
        info[f"env_info/track_perf_perc"] = tracking_performance_percentage
        if done:
            info[f"env_info/eps_track_perf_perc"] = episode_tracking_performance_percentage
        info["env_info/walk_height"] = trunk_z
        info["env_info/xy_vel_diff_norm"] = xy_velocity_difference_norm
        info["env_info/yaw_vel_diff_norm"] = yaw_velocity_difference_norm
        info["env_info/torque_norm"] = torque_norm
        info["env_info/acceleration_norm"] = acceleration_norm
        info["env_info/action_rate_norm"] = action_rate_norm
        info["env_info/power"] = power
        info["env_info/cost_of_transport"] = cost_of_transport
        info["env_info/froude_number"] = froude_number
        info["env_info/curriculum_coeff"] = curriculum_coeff

        #############################TRACKING INFO##################################
        info[f"reward/track_trunk_vel"] = tracking_xy_velocity_command_reward
        info[f"reward/ref_root_pose_reward"] = ref_root_pose_reward
        info[f"reward/ref_joint_pos_reward"] = ref_joint_pos_reward
        info[f"reward/ref_joint_vel_reward"] = ref_joint_vel_reward
        info[f"reward/ref_toes_pos_reward"] = ref_toes_pos_reward
        info[f"reward/root_pose_diff_norm"] = root_pose_diff_norm
        info[f"reward/joint_pos_diff_norm"] = joint_pos_diff_norm
        info[f"reward/joint_vel_diff_norm"] = joint_vel_diff_norm
        info[f"reward/toes_pos_diff_norm"] = toes_pos_diff_norm
        info[f"env_info/target_gait"] = self.env.target_gait
        #############################################################################

        return reward, info



    def _calc_reward_trunk_pose(self, obs):
        target_trunk_pos, target_trunk_quat = self.env.track_trunk_pos.copy(), self.env.track_trunk_quat.copy() #todo
        current_trunk_pos = self.env.data.qpos[:3].copy()
        current_trunk_quat = self.env.data.qpos[3:7].copy()
 
        trunk_quat_diff = self._quat_diff(target_trunk_quat,current_trunk_quat) 
        trunk_pos_diff = np.sum(np.square(target_trunk_pos - current_trunk_pos))
   
        tmp_pose_reward = np.exp((-trunk_pos_diff * self.env.ref_trunk_pos_coeff) -trunk_quat_diff * self.env.ref_trunk_quat_coeff)

        tmp_reward = self.env.ref_tracking_base_pose_coeff*(tmp_pose_reward)
        return tmp_reward, trunk_quat_diff+trunk_pos_diff
    
    def _quat_diff(self, quat1, quat2):
        r1 = R.from_quat(quat1, scalar_first=True)
        r2 = R.from_quat(quat2, scalar_first=True)
        r_rel = r2 * r1.inv()
        return np.sum(np.square(r_rel.as_rotvec()))


    def _calc_reward_joint_position(self, obs):
        # joint position
        target_joint_pos = self.env.track_joint_pos.copy()
        current_joint_pos = obs[self.env.joint_pos_obs_idx].copy()
        joint_pos_diff = np.sum(np.square(target_joint_pos - current_joint_pos))
        tmp_reward = np.exp(-joint_pos_diff * self.env.ref_joint_pos_coeff)
        tmp_reward = self.env.ref_tracking_joint_pos_coeff * tmp_reward
        return tmp_reward, joint_pos_diff
        

    def _calc_reward_joint_velocity(self, obs):
        target_joint_vel = self.env.track_joint_vel.copy()
        current_joint_vel = obs[self.env.joint_vel_obs_idx].copy()
        joint_vel_diff = np.sum(np.square(target_joint_vel - current_joint_vel))
        tmp_reward = np.exp(-joint_vel_diff * self.env.ref_joint_vel_coeff)
        tmp_reward = self.env.ref_tracking_joint_vel_coeff * tmp_reward
        return tmp_reward, joint_vel_diff

    def _calc_reward_toes_position(self, obs):
        target_toes_pos = self.env.track_toes_pos.copy() #+ np.array(list(self.env.track_trunk_pos.copy()) *4)
        current_toes_pos = self.env.data.geom_xpos[self.env.foot_geom_indices].copy() - self.env.data.qpos[:3]
        toes_pos_diff = np.sum(np.square(target_toes_pos - current_toes_pos.flatten()))
        tmp_reward = np.exp(-toes_pos_diff * self.env.ref_toes_pos_coeff)
        tmp_reward = self.env.ref_tracking_toes_pos_coeff * tmp_reward
        return tmp_reward, toes_pos_diff
    