from loco_new.environments.unitree_a1.reward_functions.default import DefaultReward


def get_reward_function(name, env, **kwargs):
    if name == "default":
        return DefaultReward(env, **kwargs)
    else:
        raise NotImplementedError
