from loco_new.environments.unitree_go2.termination_functions.trunk_collision_and_power import TrunkCollisionAndPowerTermination


def get_termination_function(name, env, **kwargs):
    if name == "trunk_collision_and_power":
        return TrunkCollisionAndPowerTermination(env, **kwargs)
    else:
        raise NotImplementedError
