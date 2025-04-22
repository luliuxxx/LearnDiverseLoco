from pathlib import Path
import psutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from dm_control import mjcf
import pygame

from loco_new.environments import observation_indices as obs_idx

from loco_new.environments.unitree_go2.viewer import MujocoViewer
from loco_new.environments.unitree_go2.control_functions.handler import get_control_function
from loco_new.environments.unitree_go2.command_functions.handler import get_command_function
from loco_new.environments.unitree_go2.sampling_functions.handler import get_sampling_function
from loco_new.environments.unitree_go2.initial_state_functions.handler import get_initial_state_function
from loco_new.environments.unitree_go2.reward_functions.handler import get_reward_function
from loco_new.environments.unitree_go2.termination_functions.handler import get_termination_function
from loco_new.environments.unitree_go2.domain_randomization.action_delay_functions.handler import get_get_domain_randomization_action_delay_function
from loco_new.environments.unitree_go2.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from loco_new.environments.unitree_go2.domain_randomization.control_functions.handler import get_domain_randomization_control_function
from loco_new.environments.unitree_go2.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from loco_new.environments.unitree_go2.observation_noise_functions.handler import get_observation_noise_function
from loco_new.environments.unitree_go2.observation_dropout_functions.handler import get_observation_dropout_function
from loco_new.environments.unitree_go2.terrain_functions.handler import get_terrain_function

# third_party
import torch
import torch.nn.functional as F
import scipy.spatial.transform as transform
import random
import binascii
import os

class UnitreeGo2(gym.Env):
    def __init__(self, seed, render,
                 mode,
                 control_type, command_type, command_sampling_type, initial_state_type,
                 reward_type, termination_type,
                 domain_randomization_sampling_type,
                 domain_randomization_action_delay_type,
                 domain_randomization_mujoco_model_type,
                 domain_randomization_control_type,
                 domain_randomization_perturbation_type, domain_randomization_perturbation_sampling_type,
                 observation_noise_type, observation_dropout_type, terrain_type, mask_feet_for_policy,
                 add_goal_arrow, timestep, episode_length_in_seconds, total_nr_envs, target_gait_name=None,
                 cpu_id=None):
        
        if cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([cpu_id,])

        self.seed = seed
        self.mode = mode
        self.mask_feet_for_policy = mask_feet_for_policy
        self.add_goal_arrow = add_goal_arrow
        self.episode_length_in_seconds = episode_length_in_seconds
        self.total_nr_envs = total_nr_envs
        self.eval = False
        self.eval_at_last_setup = self.eval
        self.np_rng = np.random.default_rng(self.seed)
        self.nominal_joint_positions = np.array([
            0.1, 0.8, -1.5,
            -0.1, -0.8, 1.5,
            -0.1, 1.0, -1.5,
            0.1, -1.0, 1.5,
        ])
        self.max_joint_velocities = np.array([30.1, 30.1, 15.7] * 4)
        self.power_limit_watt = 3000
        self.initial_drop_height = 0.325

        self.total_timesteps = 0
        self.goal_x_velocity = 0
        self.goal_y_velocity = 0
        self.goal_yaw_velocity = 0

        if mode == "test":
            initial_state_type = "default"
            domain_randomization_sampling_type = "none"
            domain_randomization_perturbation_sampling_type = "none"
            observation_noise_type = "none"
            observation_dropout_type = "none"
            domain_randomization_action_delay_type = 'none'
            domain_randomization_mujoco_model_type = 'none'
            domain_randomization_control_type = 'none'
            domain_randomization_perturbation_type = 'none'



        self.control_function = get_control_function(control_type, self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / timestep))
        self.nr_intermediate_steps = 1
        self.dt = timestep * self.nr_substeps * self.nr_intermediate_steps
        self.horizon = int(round(episode_length_in_seconds * self.control_frequency_hz))
        self.command_function = get_command_function(command_type, self)
        self.command_sampling_function = get_sampling_function(command_sampling_type, self)
        self.initial_state_function = get_initial_state_function(initial_state_type, self)
        self.reward_function = get_reward_function(reward_type, self)
        self.termination_function = get_termination_function(termination_type, self)
        self.domain_randomization_sampling_function = get_sampling_function(domain_randomization_sampling_type, self)
        self.domain_randomization_action_delay_function = get_get_domain_randomization_action_delay_function(domain_randomization_action_delay_type, self)
        self.domain_randomization_mujoco_model_function = get_domain_randomization_mujoco_model_function(domain_randomization_mujoco_model_type, self)
        self.domain_randomization_control_function = get_domain_randomization_control_function(domain_randomization_control_type, self)
        self.domain_randomization_perturbation_function = get_domain_randomization_perturbation_function(domain_randomization_perturbation_type, self)
        self.domain_randomization_perturbation_sampling_function = get_sampling_function(domain_randomization_perturbation_sampling_type, self)
        self.observation_noise_function = get_observation_noise_function(observation_noise_type, self)
        self.observation_dropout_function = get_observation_dropout_function(observation_dropout_type, self)
        self.terrain_function = get_terrain_function(terrain_type, self)

        xml_file_name = self.terrain_function.xml_file_name
        xml_path = (Path(__file__).resolve().parent / "data" / xml_file_name).as_posix()
        if self.add_goal_arrow:
            # Add goal arrow
            xml_handle = mjcf.from_path(xml_path)
            trunk = xml_handle.find("body", "trunk")
            trunk.add("body", name="dir_arrow", pos="0 0 0.15")
            dir_vec = xml_handle.find("body", "dir_arrow")
            dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 0")
            dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")
            self.model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

        collision_groups = [("floor", ["floor"]),
                            ("feet", ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
                            ("FL_foot", ["FL_foot"]), ("FR_foot", ["FR_foot"]), ("RL_foot", ["RL_foot"]), ("RR_foot", ["RR_foot"]),
                            ("FR_calf0", ["FR_calf0"]), ("FL_calf0", ["FL_calf0"]), ("RR_calf0", ["RR_calf0"]), ("RL_calf0", ["RL_calf0"]),
                            ("trunk", ["trunk_1", "trunk_2", "trunk_3"]),
                            ("trunk_1", ["trunk_1"]), ("trunk_2", ["trunk_2"]), ("trunk_3", ["trunk_3"])]
        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                self.collision_groups[name] = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) for geom_name in geom_names}

        self.viewer = None if not render else MujocoViewer(self.model, self.dt)

        action_space_low = -np.ones(self.model.nu) * np.inf
        action_space_high = np.ones(self.model.nu) * np.inf
        self.action_space = gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)

        self.joint_order = [
            obs_idx.QUADRUPED_FRONT_LEFT_HIP, obs_idx.QUADRUPED_FRONT_LEFT_THIGH, obs_idx.QUADRUPED_FRONT_LEFT_CALF,
            obs_idx.QUADRUPED_FRONT_RIGHT_HIP, obs_idx.QUADRUPED_FRONT_RIGHT_THIGH, obs_idx.QUADRUPED_FRONT_RIGHT_CALF,
            obs_idx.QUADRUPED_BACK_LEFT_HIP, obs_idx.QUADRUPED_BACK_LEFT_THIGH, obs_idx.QUADRUPED_BACK_LEFT_CALF,
            obs_idx.QUADRUPED_BACK_RIGHT_HIP, obs_idx.QUADRUPED_BACK_RIGHT_THIGH, obs_idx.QUADRUPED_BACK_RIGHT_CALF,
        ]

        self.joint_vel_obs_idx = [self.joint_order[i][1] for i in range(0, len(self.joint_order))]
        self.joint_pos_obs_idx = [self.joint_order[i][0] for i in range(0, len(self.joint_order))]
        
        self.foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.foot_geom_indices = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.foot_names])

        qpos, qvel = get_initial_state_function("default", self).setup()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.update_orientation_attributes()
        
        self.observation_space = self.get_observation_space()

        self.current_action = np.zeros(self.model.nu)
        self.initial_observation = self.get_initial_observation()

        self.reward_function.init()
        self.domain_randomization_mujoco_model_function.init()
        self.observation_noise_function.init()

        ######################
        self._ref_init()
        self.target_gait = 0
        ######################

        if self.mode == "test":
            assert target_gait_name is not None
            self.target_gait = self.name2index[target_gait_name]

        self.target_traj = 0
        
        self.obs_tracking_data = self.name2obs_tracking_buf[self.index2name[self.target_gait]][self.target_traj]
        self.gait = self.index2name[self.target_gait].split('_')[-1]
        self.source = self.index2name[self.target_gait].split('_')[0]



    def reset(self, seed=None):
 
        self.episode_step = 0
        self.current_action = np.zeros(self.model.nu)
        self.last_action = np.zeros(self.model.nu)
        self.current_torques = np.zeros(self.model.nu)

        self.termination_function.setup()
        self.terrain_function.sample()
        self.command_function.setup()
        self.reward_function.setup()
        self.handle_domain_randomization(function="setup")

        qpos, qvel = self.initial_state_function.setup()

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        
        self.update_orientation_attributes()
        

        if self.viewer:
            self.viewer.render(self.data)


        random.seed(os.urandom(4))

        seed = int(binascii.hexlify(os.urandom(4)), 16)
     
        np.random.seed(seed)
        
        if self.mode == "train":
            self.target_gait = np.random.choice(self.total_ref_gaits)
            self.target_traj = np.random.choice(len(self.name2obs_tracking_buf[self.index2name[self.target_gait]]))


        self.obs_tracking_data = self.name2obs_tracking_buf[self.index2name[int(self.target_gait)]][self.target_traj]
        self.gait = self.index2name[int(self.target_gait)].split('_')[-1]
        self.source = self.index2name[int(self.target_gait)].split('_')[0]
        
        self.track_trunk_pos = self.default_pos_tracking[self.ref_idx_trunk_pos]
        self.track_trunk_quat = self.default_pos_tracking[self.ref_idx_trunk_quat]
        self.track_trunk_lin_vel = self.default_pos_tracking[self.ref_idx_trunk_lin_vel]
        self.track_trunk_ang_vel = self.default_pos_tracking[self.ref_idx_trunk_ang_vel]
        self.track_joint_pos = self.default_pos_tracking[self.ref_idx_joint_pos]
        self.track_joint_vel = self.default_pos_tracking[self.ref_idx_joint_vel]
        self.track_toes_pos = self.default_pos_tracking[self.ref_idx_toes_pos]
        return self.get_observation(), {}


    def step(self, action):
        if self.episode_step == 0:
            self.target_traj += 1 if self.target_traj < len(self.name2obs_tracking_buf[self.index2name[self.target_gait]]) - 1 else 0


        action = self.domain_randomization_action_delay_function.delay_action(action)

        torques = self.control_function.process_action(action)

        for _ in range(self.nr_intermediate_steps):
            self.data.ctrl = torques
            mujoco.mj_step(self.model, self.data, self.nr_substeps)
            self.data.qvel[6:] = np.clip(self.data.qvel[6:], -self.max_joint_velocities, self.max_joint_velocities)

        self.update_orientation_attributes()

        if self.add_goal_arrow:
            trunk_rotation = self.orientation_euler[2]
            desired_angle = trunk_rotation + np.arctan2(self.goal_y_velocity, self.goal_x_velocity)
            rot_mat = R.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            self.data.site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([self.goal_x_velocity, self.goal_y_velocity])))
            self.model.site_size[1, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            self.data.site("dir_arrow").xpos += [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            self.data.site("dir_arrow_ball").xpos = self.data.body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]

        if self.viewer:
            self.viewer.render(self.data)
        
        self.current_action = action.copy()
        self.current_torques = torques

        self.handle_domain_randomization(function="step")

        next_observation = self.get_observation()
        terminated = self.termination_function.should_terminate(next_observation)
        truncated = self.episode_step + 1 >= self.horizon
        done = terminated | truncated
        reward, r_info = self.get_reward_and_info(done)
        info = {**r_info}

        self.reward_function.step(action)
        self.command_function.step(next_observation, reward, done, info)
        self.initial_state_function.step(next_observation, reward, done, info)
        self.terrain_function.step(next_observation, reward, done, info)
        
        self.last_action = action.copy()
        self.episode_step += 1
        if not self.eval:
            self.total_timesteps += 1

        return next_observation, reward, terminated, truncated, info


    def update_orientation_attributes(self):
        self.orientation_quat = R.from_quat([self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3]])
        self.orientation_euler = self.orientation_quat.as_euler("xyz")
        self.orientation_quat_inv = self.orientation_quat.inv()


    def handle_domain_randomization(self, function="setup"):
        if function == "setup":
            if self.eval_at_last_setup != self.eval:
                self.should_randomize_domain = True
                self.should_randomize_domain_perturbation = True
                self.eval_at_last_setup = self.eval
            else:
                self.should_randomize_domain = self.domain_randomization_sampling_function.setup()
                self.should_randomize_domain_perturbation = self.domain_randomization_perturbation_sampling_function.setup()
            self.domain_randomization_action_delay_function.setup()
        elif function == "step":
            self.should_randomize_domain = self.domain_randomization_sampling_function.step()
            self.should_randomize_domain_perturbation = self.domain_randomization_perturbation_sampling_function.step()
        if self.should_randomize_domain:
            self.domain_randomization_control_function.sample()
            self.domain_randomization_mujoco_model_function.sample()
            self.domain_randomization_action_delay_function.sample()
            self.reward_function.init()
        if self.should_randomize_domain_perturbation:
            self.domain_randomization_perturbation_function.sample()


    def get_observation_space(self):
        
        space_low = np.array([-np.inf] * obs_idx.OBSERVATION_SIZE, dtype=np.float32)
        space_high = np.array([np.inf] * obs_idx.OBSERVATION_SIZE, dtype=np.float32)

        return gym.spaces.Box(low=space_low, high=space_high, shape=space_low.shape, dtype=np.float32)


    def get_initial_observation(self):
        return np.zeros(obs_idx.OBSERVATION_SIZE, dtype=np.float32)
    

    def get_observation(self):
        observation = self.initial_observation.copy()
        

        ####################TRACKING################################

        tracking_data = self.obs_tracking_data[self.episode_step]

        observation[obs_idx.TARGET_TRACKING] = tracking_data[3:]

        observation[obs_idx.TRACK_TRUNK_ANG_VEL] /= 10.0

        observation[obs_idx.TRACK_JOINT_POS] -= self.nominal_joint_positions.flatten()

        observation[obs_idx.TRACK_JOINT_POS] /= 3.14

        observation[obs_idx.TRACK_JOINT_VEL] /= self.max_joint_velocities

        gait_index = self.gait_name2index[self.gait]
        source_index = self.source_name2index[self.source]
        observation[obs_idx.TRACK_GAIT] = F.one_hot(torch.tensor(gait_index), num_classes=len(self.gait_name2index)).numpy()
        observation[obs_idx.TRACK_SOURCE] = F.one_hot(torch.tensor(source_index), num_classes=len(self.source_name2index)).numpy()

        for i, joint_range in enumerate(self.joint_order):
            observation[joint_range[0]] = self.data.qpos[i+7] - self.nominal_joint_positions[i]
            observation[joint_range[1]] = self.data.qvel[i+6]
            observation[joint_range[2]] = self.current_action[i]

        trunk_linear_velocity = self.data.qvel[:3]
        observation[obs_idx.TRUNK_LINEAR_VELOCITIES] = trunk_linear_velocity

        trunk_angular_velocity = self.data.qvel[3:6]
        observation[obs_idx.TRUNK_ANGULAR_VELOCITIES] = trunk_angular_velocity

        projected_gravity_vector = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))
        observation[obs_idx.PROJECTED_GRAVITY] = projected_gravity_vector

        observation[obs_idx.HEIGHT] = self.terrain_function.get_height_samples()


        observation = self.observation_noise_function.modify_observation(observation)

        observation = self.observation_dropout_function.modify_observation(observation)

        for i, joint_range in enumerate(self.joint_order):
            observation[joint_range[0]] /= 3.14
            observation[joint_range[1]] /= self.max_joint_velocities[i]
            observation[joint_range[2]] /= 3.14

        observation[obs_idx.TRUNK_ANGULAR_VELOCITIES] /= 10.0

        self.track_trunk_pos = tracking_data[self.ref_idx_trunk_pos]
        self.track_trunk_quat = tracking_data[self.ref_idx_trunk_quat]
        self.track_trunk_lin_vel = tracking_data[self.ref_idx_trunk_lin_vel] * 0.8
        self.track_trunk_ang_vel = tracking_data[self.ref_idx_trunk_ang_vel] / 10.0
        self.track_joint_pos = (tracking_data[self.ref_idx_joint_pos] - self.nominal_joint_positions.flatten()) / 3.14
        self.track_joint_vel = tracking_data[self.ref_idx_joint_vel] / self.max_joint_velocities.flatten()
        self.track_toes_pos = tracking_data[self.ref_idx_toes_pos]

        observation[obs_idx.TRACK_JOINT_POS] -= observation[self.joint_pos_obs_idx]
        observation[obs_idx.TRACK_JOINT_VEL] -= observation[self.joint_vel_obs_idx]
        observation[obs_idx.TRACK_TRUNK_ANG_VEL] -= observation[obs_idx.TRUNK_ANGULAR_VELOCITIES]

        return observation


    def get_reward_and_info(self, done):
        info = {"t": self.episode_step}
        reward, info = self.reward_function.reward_and_info(info, done)
        info = self.terrain_function.info(info)

        return reward, info


    def close(self):
        if self.viewer:
            self.viewer.close()
        if self.mode == "test":
            pygame.quit()


    def check_collision(self, groups1, groups2):
        if isinstance(groups1, list):
            ids1 = [self.collision_groups[group] for group in groups1]
            ids1 = set().union(*ids1)
        else:
            ids1 = self.collision_groups[groups1]
        
        if isinstance(groups2, list):
            ids2 = [self.collision_groups[group] for group in groups2]
            ids2 = set().union(*ids2)
        else:
            ids2 = self.collision_groups[groups2]

        for coni in range(0, self.data.ncon):
            con = self.data.contact[coni]

            collision = con.geom1 in ids1 and con.geom2 in ids2
            collision_trans = con.geom1 in ids2 and con.geom2 in ids1

            if collision or collision_trans:
                return True

        return False
    

    def check_any_collision(self, groups):
        if isinstance(groups, list):
            ids = [self.collision_groups[group] for group in groups]
            ids = set().union(*ids)
        else:
            ids = self.collision_groups[groups]

        for con_i in range(0, self.data.ncon):
            con = self.data.contact[con_i]
            if con.geom1 in ids or con.geom2 in ids:
                return True
        
        return False


    def check_any_collision_for_all(self, groups):
        ids = [self.collision_groups[group] for group in groups]
        ids = set().union(*ids)

        any_collision = {idx: False for idx in ids}

        for con_i in range(0, self.data.ncon):
            con = self.data.contact[con_i]
            if con.geom1 in ids:
                any_collision[con.geom1] = True
                ids.remove(con.geom1)
            if con.geom2 in ids:
                any_collision[con.geom2] = True
                ids.remove(con.geom2)
        
        return any_collision

    def _load_ref_data(self):
        import glob
        from pathlib import Path
        if self.mode == "train":
            file_path = Path(__file__).resolve().parent / 'dataset' / 'train'
        else:
            file_path = Path(__file__).resolve().parent / 'dataset' / 'train'
        file_list = glob.glob(str(file_path / '*.npz'))
        file_list.sort()
        assert len(file_list) > 0
        self.name2index = {}
        self.index2name = {}
        tmp_buf_src = []
        tmp_buf_gait = []
        tmp_buf_motion = []
        self.source_name2index = {}
        self.gait_name2index = {}
        self.name2ref = {}
        for file in file_list:
            motion_name = file.split('/')[-1].split('.')[0].split('_')[-3] + '_' + file.split('/')[-1].split('.')[0].split('_')[-2]
            source_name = motion_name.split('_')[0] # dog, horse, mpc, solo8
            gait_name = motion_name.split('_')[1] # trot, walk, pace, bound, gallop...
            if source_name not in self.source_name2index:
                tmp_buf_src.append(source_name)
                self.source_name2index[source_name] = len(tmp_buf_src) - 1
            if gait_name not in self.gait_name2index:
                tmp_buf_gait.append(gait_name)
                self.gait_name2index[gait_name] = len(tmp_buf_gait) - 1
            if motion_name not in self.name2index:
                tmp_buf_motion.append(motion_name)
                self.name2index[motion_name] = len(tmp_buf_motion) - 1
                self.name2ref[motion_name] = []
            data = np.asarray(np.load(file)['data'])
            self.name2ref[motion_name].append(data)
        self.index2name = {v: k for k, v in self.name2index.items()}
        # 100 frames, 19 dims
        # in order: trunk position 3, trunk quat 4, joint position 12
    
    def _ref_init(self):
        self._load_ref_data()
        
        self.ref_dt = 0.02 # 50Hz
        self.total_ref_gaits = len(self.name2ref)
        self.name2obs_tracking_buf = {}
        
        self.default_toes_abs_pos =  np.array([[ 0.17629184,  0.17273066,  0.02349743],
                                                [ 0.17629184, -0.17273066,  0.02349743],
                                                [-0.27227085,  0.17176926,  0.03307934],
                                                [-0.27227085, -0.17176926,  0.03307934]])
        for key, data_buf in self.name2ref.items():
            assert len(data_buf) != 0
            obs_tracking_buf = []
            for ref_data in data_buf:
                assert ref_data.shape[1] == 31
                ref_joint_pos = ref_data[:, 7:19]
                ref_trunk_pos = ref_data[:, :3]
                ref_trunk_quat = ref_data[:, 3:7] # wxyz
                ref_toes_pos = ref_data[:, 19:]
                rel_toes_pos = ref_toes_pos - np.hstack((ref_trunk_pos, ref_trunk_pos, ref_trunk_pos, ref_trunk_pos))
                ref_trunk_linear_vel = self._calc_ref_trunk_linear_velocity(ref_trunk_pos)
                ref_trunk_angular_vel = self.calc_ref_trunk_angular_velocity(ref_trunk_quat)
                ref_joint_vel = self._calc_ref_joint_velocity(ref_joint_pos)
                sig_obs_tracking_data = np.hstack((ref_trunk_pos, ref_trunk_linear_vel, ref_trunk_quat, ref_trunk_angular_vel, ref_joint_pos, ref_joint_vel, rel_toes_pos))
                obs_tracking_buf.append(sig_obs_tracking_data)
            self.name2obs_tracking_buf[key] = obs_tracking_buf
        self.ref_idx_joint_pos = np.arange(3+3+4+3, 3+3+4+3+12)
        self.ref_idx_trunk_lin_vel = np.arange(3, 3+3)
        self.ref_idx_trunk_ang_vel = np.arange(3+3+4, 3+3+4+3)
        self.ref_idx_joint_vel = np.arange(3+3+4+3+12, 3+3+4+3+12+12)
        self.ref_idx_toes_pos = np.arange(3+3+4+3+12+12, 3+3+4+3+12+12+12)
        self.ref_idx_trunk_pos = np.arange(0, 3)
        self.ref_idx_trunk_quat = np.arange(3+3, 3+3+4)
        self.default_pos_tracking = np.zeros((sig_obs_tracking_data.shape[-1]))

        self.default_pos_tracking[self.ref_idx_joint_pos] = self.nominal_joint_positions
        self.default_pos_tracking[self.ref_idx_trunk_pos] = np.array([0.0, 0.0, 0.325])
        self.default_pos_tracking[self.ref_idx_trunk_quat] = np.array([1.0, 0.0, 0.0, 0.0])
        self.default_pos_tracking[self.ref_idx_trunk_lin_vel] = np.array([0.0, 0.0, 0.0])
        self.default_pos_tracking[self.ref_idx_trunk_ang_vel] = np.array([0.0, 0.0, 0.0])
        self.default_pos_tracking[self.ref_idx_joint_vel] = np.zeros(12)
        self.default_pos_tracking[self.ref_idx_toes_pos] = (self.default_toes_abs_pos - self.default_pos_tracking[self.ref_idx_trunk_pos]).flatten()
        self.ref_joint_pos_coeff = 5.0
        self.ref_tracking_joint_pos_coeff = 0.5
        self.ref_joint_vel_coeff = 0.1
        self.ref_tracking_joint_vel_coeff = 0.05
        self.ref_toes_pos_coeff = 40.0
        self.ref_tracking_toes_pos_coeff = 0.2

        self.ref_trunk_pos_coeff = 20.0
        self.ref_trunk_quat_coeff = 10.0
        self.ref_tracking_base_pose_coeff = 0.15

        

    def calc_ref_trunk_angular_velocity(self,ref_trunk_quat):

        trunk_quat = ref_trunk_quat.copy()
        trunk_rot_mats = np.array([transform.Rotation.from_quat(quat, scalar_first=True).as_matrix() for quat in trunk_quat])
        
        num_data = trunk_rot_mats.shape[0]
        matrix_tmp = trunk_rot_mats.copy()
        angular_velocity = np.zeros((num_data, 3))  

        for i in range(1, num_data):
            R1 = matrix_tmp[i-1]
            R2 = matrix_tmp[i]
            R_rel = R2 @ np.linalg.inv(R1)
            r = R.from_matrix(R_rel)
            angular_velocity[i] = r.as_rotvec() / self.ref_dt
        return angular_velocity


    def _calc_ref_trunk_linear_velocity(self,ref_trunk_pos):

        trunk_position = ref_trunk_pos.copy()
        linear_velocity = np.diff(trunk_position, axis=0) / self.ref_dt
        linear_velocity = np.vstack((np.zeros((1, 3)), linear_velocity))
        assert len(linear_velocity) == len(ref_trunk_pos)
        return linear_velocity
            
    
    def _calc_ref_joint_velocity(self,ref_joint_pos):
        joint_position = ref_joint_pos.copy()
        joint_velocities = np.diff(joint_position, axis=0) / self.ref_dt 
        joint_vel = np.vstack((np.zeros((1, 12)), joint_velocities))
        assert len(joint_vel) == len(ref_joint_pos)
        return joint_vel
    
