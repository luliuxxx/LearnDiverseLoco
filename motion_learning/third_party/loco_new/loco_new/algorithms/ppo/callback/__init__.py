from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from loco_new.algorithms.ppo.callback.ppo import PPO
from loco_new.algorithms.ppo.callback.default_config import get_config
from loco_new.algorithms.ppo.callback.general_properties import GeneralProperties


CALLBACK_PPO = extract_algorithm_name_from_file(__file__)
register_algorithm(CALLBACK_PPO, get_config, PPO, GeneralProperties)
