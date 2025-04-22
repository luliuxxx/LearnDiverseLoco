from loco_new.environments.unitree_go2.domain_randomization.control_functions.default import DefaultDomainControl
from loco_new.environments.unitree_go2.domain_randomization.control_functions.hard import HardDomainControl
from loco_new.environments.unitree_go2.domain_randomization.control_functions.none import NoneDomainControl


def get_domain_randomization_control_function(name, env, **kwargs):
    if name == "default":
        return DefaultDomainControl(env, **kwargs)
    elif name == "hard":
        return HardDomainControl(env, **kwargs)
    elif name == "none":
        return NoneDomainControl(env, **kwargs)
    else:
        raise NotImplementedError
