from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from loco_new.algorithms.ppo.default.ppo import PPO
from loco_new.algorithms.ppo.default.default_config import get_config
from loco_new.algorithms.ppo.default.general_properties import GeneralProperties


DEFAULT_PPO = extract_algorithm_name_from_file(__file__)
register_algorithm(DEFAULT_PPO, get_config, PPO, GeneralProperties)
