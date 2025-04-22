from loco_new.environments.unitree_go2.control_functions.pd import PDControl
from loco_new.environments.unitree_go2.control_functions.torque import TorqueControl


def get_control_function(name, env, **kwargs):
    if name == "pd":
        return PDControl(env, **kwargs)
    elif name == "torque":
        return TorqueControl(env, **kwargs)
    else:
        raise NotImplementedError
