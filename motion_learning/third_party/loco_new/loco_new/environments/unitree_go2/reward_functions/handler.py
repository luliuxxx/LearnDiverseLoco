from loco_new.environments.unitree_go2.reward_functions.default import DefaultReward
from loco_new.environments.unitree_go2.reward_functions.imitation import ImitationTask


def get_reward_function(name, env, **kwargs):
    if name == "default":
        return DefaultReward(env, **kwargs)
    elif name == "imitation":
        return ImitationTask(env, **kwargs)
    else:
        raise NotImplementedError
