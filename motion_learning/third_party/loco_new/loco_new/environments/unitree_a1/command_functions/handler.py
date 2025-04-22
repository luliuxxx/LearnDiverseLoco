from loco_new.environments.unitree_a1.command_functions.random import RandomCommands
from loco_new.environments.unitree_a1.command_functions.random_with_heading import RandomWithHeadingCommands


def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    elif name == "random_with_heading":
        return RandomWithHeadingCommands(env, **kwargs)
    else:
        raise NotImplementedError
