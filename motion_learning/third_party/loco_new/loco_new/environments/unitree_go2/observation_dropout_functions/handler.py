from loco_new.environments.unitree_go2.observation_dropout_functions.default import DefaultObservationDropout
from loco_new.environments.unitree_go2.observation_dropout_functions.none import NoneObservationDropout
from loco_new.environments.unitree_go2.observation_dropout_functions.hard import HardObservationDropout



def get_observation_dropout_function(name, env, **kwargs):
    if name == "default":
        return DefaultObservationDropout(env, **kwargs)
    elif name == "hard":
        return HardObservationDropout(env, **kwargs)
    elif name == "none":
        return NoneObservationDropout(env, **kwargs)
    else:
        raise NotImplementedError
