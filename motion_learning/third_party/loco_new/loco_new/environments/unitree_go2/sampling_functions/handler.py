from loco_new.environments.unitree_go2.sampling_functions.step_probability import StepProbabilitySampling
from loco_new.environments.unitree_go2.sampling_functions.every_step import EveryStepSampling
from loco_new.environments.unitree_go2.sampling_functions.none import NoneSampling


def get_sampling_function(name, env, **kwargs):
    if name == "step_probability":
        return StepProbabilitySampling(env, **kwargs)
    elif name == "every_step":
        return EveryStepSampling(env, **kwargs)
    elif name == "none":
        return NoneSampling(env, **kwargs)
    else:
        raise NotImplementedError
