from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from loco_new.algorithms.sac.default.sac import SAC
from loco_new.algorithms.sac.default.default_config import get_config
from loco_new.algorithms.sac.default.general_properties import GeneralProperties


SAC_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(SAC_FLAX, get_config, SAC, GeneralProperties)
