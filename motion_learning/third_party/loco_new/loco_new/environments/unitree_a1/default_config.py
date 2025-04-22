from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.nr_envs = 1

    config.seed = 1
    config.async_skip_percentage = 0.0
    config.cycle_cpu_affinity = False
    config.render = False
    config.mode = "train"                                                           # "train", "test"
    config.control_type = "pd"                                                      # "pd", "torque"
    config.command_type = "random"                                                  # "random", "random_with_heading"
    config.command_sampling_type = "step_probability"                               # "step_probability", "every_step", "none"
    config.initial_state_type = "random"                                            # "default", "random"
    config.reward_type = "default"                                                  # "default"
    config.termination_type = "trunk_collision_and_power"                           # "trunk_collision_and_power"
    config.domain_randomization_sampling_type = "step_probability"                  # "step_probability", "every_step", "none"
    config.domain_randomization_action_delay_type = "default"                       # "default", "hard", "none"
    config.domain_randomization_mujoco_model_type = "default"                       # "default", "hard", "none"
    config.domain_randomization_control_type = "default"                            # "default", "hard", "none"
    config.domain_randomization_perturbation_sampling_type = "step_probability"     # "step_probability", "every_step", "none"
    config.domain_randomization_perturbation_type = "default"                       # "default", "hard", "none"
    config.observation_dropout_type = "default"                                     # "default", "hard", "none"
    config.observation_noise_type = "default"                                       # "default", "hard", "none"
    config.terrain_type = "plane"                                                   # "plane", "hfield_rough_plane", "hfield_inverse_pyramid", "hfield_curriculum"
    config.mask_feet_for_policy = False
    config.add_goal_arrow = False
    config.timestep = 0.005
    config.episode_length_in_seconds = 20

    return config
