from loco_new.environments.unitree_a1.terrain_functions.plane import PlaneTerrainGeneration
from loco_new.environments.unitree_a1.terrain_functions.hfield_rough_plane import HfieldRoughPlaneTerrainGeneration
from loco_new.environments.unitree_a1.terrain_functions.hfield_inverse_pyramid import HfieldInversePyramidTerrainGeneration
from loco_new.environments.unitree_a1.terrain_functions.hfield_curriculum import HFieldCurriculumTerrainGeneration


def get_terrain_function(name, env, **kwargs):
    if name == "plane":
        return PlaneTerrainGeneration(env, **kwargs)
    elif name == "hfield_rough_plane":
        return HfieldRoughPlaneTerrainGeneration(env, **kwargs)
    elif name == "hfield_inverse_pyramid":
        return HfieldInversePyramidTerrainGeneration(env, **kwargs)
    elif name == "hfield_curriculum":
        return HFieldCurriculumTerrainGeneration(env, **kwargs)
    else:
        raise NotImplementedError
