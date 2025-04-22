from loco_new.environments.unitree_go2.observation_noise_functions.default import DefaultObservationNoise
from loco_new.environments.unitree_go2.observation_noise_functions.hard import HardObservationNoise
from loco_new.environments.unitree_go2.observation_noise_functions.none import NoneObservationNoise


def get_observation_noise_function(name, env, **kwargs):
    if name == "default":
        return DefaultObservationNoise(env, **kwargs)
    elif name == "hard":
        return HardObservationNoise(env, **kwargs)
    elif name == "none":
        return NoneObservationNoise(env, **kwargs)
    else:
        raise NotImplementedError
