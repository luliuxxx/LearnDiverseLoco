import os
import numpy as np
import mujoco
from loco_new.environments.unitree_go2.viewer import MujocoViewer
file_path = os.path.abspath(__file__)

dir_path = os.path.dirname(file_path)

folder = 'train'
# get parent directory of dir_path
from pathlib import Path

xml_path = os.path.join(Path(dir_path).parent, 'data/plane.xml')

# load the model

model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(model)
viewer = MujocoViewer(model, dt=1/50)

def view_data(ref_data):
    # trunk base x, y, z 3
    # trunk orientation w, x, y, z 4
    # joint angles FR FL RR RL 12
    # toes  FR FL RR RL 4*3 12
    # total 31

    mj_data = mujoco.MjData(model)
    frames = ref_data.shape[0]
    for frame in range(frames):
        data = ref_data[frame]
        trunk_base = data[:3]
        # print("height:", trunk_base[2])
        trunk_orientation = data[3:7]
        joint_angles = data[7:19]
        toes = data[19:]
        mj_data.qpos[:3] = trunk_base
        mj_data.qpos[3:7] = trunk_orientation
        mj_data.qpos[7:19] = joint_angles
        mujoco.mj_step(model, mj_data)
        viewer.render(mj_data)
        # sleep(0.01)

        # viewer.update(trunk_base, trunk_orientation, joint_angles, toes


# Load the dataset
for root, dirs, files in os.walk(os.path.join(dir_path, folder)):
    for file in sorted(files):
        if file.endswith('.npz'):
            if "mpc_trot" not in file:
                continue
            file_name = os.path.join(root, file)
            file_data = np.load(file_name)
            data = file_data['data']
            print("data shape:", data.shape)
            for iter_time in range(2):
                view_data(data) 
            
viewer.close()