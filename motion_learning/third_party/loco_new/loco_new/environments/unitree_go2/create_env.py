import logging
import gymnasium as gym

from loco_new.environments.unitree_go2.environment import UnitreeGo2
from loco_new.environments.unitree_go2.wrappers import RLXInfo, RecordEpisodeStatistics
from loco_new.environments.unitree_go2.general_properties import GeneralProperties
from loco_new.environments.unitree_go2.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from loco_new.environments.unitree_go2.cpu_gpu_testing import get_global_cpu_ids, get_fastest_cpu_for_gpu_connection

import os
import random
import binascii

rlx_logger = logging.getLogger("rl_x")


def create_env(config):
    def make_env(seed, env_cpu_id):
        def thunk():
            env = UnitreeGo2(
                seed=seed,
                render=config.environment.render,
                mode=config.environment.mode,
                control_type=config.environment.control_type,
                command_type=config.environment.command_type,
                command_sampling_type=config.environment.command_sampling_type,
                initial_state_type=config.environment.initial_state_type,
                reward_type=config.environment.reward_type,
                termination_type=config.environment.termination_type,
                domain_randomization_sampling_type=config.environment.domain_randomization_sampling_type,
                domain_randomization_action_delay_type=config.environment.domain_randomization_action_delay_type,
                domain_randomization_mujoco_model_type=config.environment.domain_randomization_mujoco_model_type,
                domain_randomization_control_type=config.environment.domain_randomization_control_type,
                domain_randomization_perturbation_type=config.environment.domain_randomization_perturbation_type,
                domain_randomization_perturbation_sampling_type=config.environment.domain_randomization_perturbation_sampling_type,
                observation_noise_type=config.environment.observation_noise_type,
                observation_dropout_type=config.environment.observation_dropout_type,
                terrain_type=config.environment.terrain_type,
                mask_feet_for_policy=config.environment.mask_feet_for_policy,
                add_goal_arrow=config.environment.add_goal_arrow,
                timestep=config.environment.timestep,
                episode_length_in_seconds=config.environment.episode_length_in_seconds,
                total_nr_envs=config.environment.nr_envs,
                target_gait_name=config.environment.target_gait_name,
                cpu_id=env_cpu_id
            )
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    

    global_cpu_ids = None
    if config.environment.cycle_cpu_affinity or config.algorithm.determine_fastest_cpu_for_gpu:
        global_cpu_ids = get_global_cpu_ids()
        rlx_logger.info(f"Global CPU IDs: {global_cpu_ids}")

    fastest_cpu_id = None
    if config.algorithm.determine_fastest_cpu_for_gpu:
        fastest_cpu_id = get_fastest_cpu_for_gpu_connection(global_cpu_ids)

    env_cpu_ids = None
    if config.environment.cycle_cpu_affinity:
        usable_cpu_ids_for_envs = global_cpu_ids.copy()
        if fastest_cpu_id is not None:
            usable_cpu_ids_for_envs.remove(fastest_cpu_id)
        env_cpu_ids = []
        for i in range(config.environment.nr_envs):
            env_cpu_ids.append(usable_cpu_ids_for_envs[i % len(usable_cpu_ids_for_envs)])

    env_list = []
    env_id = 0
    for i in range(config.environment.nr_envs):
        env_cpu_id = None if env_cpu_ids is None else env_cpu_ids[env_id]
        random.seed(os.urandom(4))
        seed = int(binascii.hexlify(os.urandom(4)), 16)
        env_list.append(make_env(
            seed=seed,
            env_cpu_id=env_cpu_id
        ))
        env_id += 1
    if config.environment.nr_envs == 1:
        env = gym.vector.SyncVectorEnv(env_list)
    else:
        env = AsyncVectorEnvWithSkipping(env_list, config.environment.async_skip_percentage)
    env = RLXInfo(env, fastest_cpu_id)

    env.general_properties = GeneralProperties


    random.seed(os.urandom(4))
    seed = int(binascii.hexlify(os.urandom(4)), 16)
    env.reset(seed=seed)

    return env
