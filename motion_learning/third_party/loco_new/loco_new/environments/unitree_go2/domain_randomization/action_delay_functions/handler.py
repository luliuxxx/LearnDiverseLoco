from loco_new.environments.unitree_go2.domain_randomization.action_delay_functions.default import DefaultActionDelay
from loco_new.environments.unitree_go2.domain_randomization.action_delay_functions.hard import HardActionDelay
from loco_new.environments.unitree_go2.domain_randomization.action_delay_functions.none import NoneActionDelay


def get_get_domain_randomization_action_delay_function(name, env, **kwargs):
    if name == "none":
        return NoneActionDelay(env, **kwargs)
    elif name == "default":
        return DefaultActionDelay(env, **kwargs)
    elif name == "hard":
        return HardActionDelay(env, **kwargs)
    else:
        raise NotImplementedError
