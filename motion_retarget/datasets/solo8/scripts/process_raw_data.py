import pinocchio as pin
import numpy as np
from matplotlib import pyplot as plt
from transforms3d.quaternions import quat2mat
from transforms3d.affines import compose
import torch
import json
from pathlib import Path

motion_file = Path(__file__).resolve().parents[1].joinpath("raw_data/motion_data.pt")
ref_state_idx_dir = Path(__file__).resolve().parents[1].joinpath("raw_data/reference_state_idx_dict.json")
# import pdb; pdb.set_trace()
urdf_path = str(Path(__file__).resolve().parents[1].joinpath("urdf/solo8.urdf"))

legs = ["FL", "FR", "HL", "HR"]

loaded_data = torch.load(motion_file).to("cpu")
state_idx_dict = json.load(open(ref_state_idx_dir, "r"))

base_pos = loaded_data[:, :, state_idx_dict["base_pos"][0]:state_idx_dict["base_pos"][1]]
base_quat = loaded_data[:, :, state_idx_dict["base_quat"][0]:state_idx_dict["base_quat"][1]] # xyzw
base_height = loaded_data[:, :, state_idx_dict["base_height"][0]:state_idx_dict["base_height"][1]]
dof_pos = loaded_data[:, :, state_idx_dict["dof_pos"][0]:state_idx_dict["dof_pos"][1]]

# Create a robot model
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
q = np.zeros(model.nq)

pin.forwardKinematics(model,data,q)
pin.updateFramePlacements(model, data)

traj_nums = loaded_data.shape[0]
traj_len = loaded_data.shape[1]
traj_dims = loaded_data.shape[2]

data_txt_buffer = []
for traj in range(traj_nums):
    for t in range(traj_len):
        base_pos_t = base_pos[traj, t].numpy()
        base_height_t = base_height[traj, t].numpy()
        base_pos_t[2] = base_height_t[0]
        
        base_quat_t = base_quat[traj, t].numpy()
        base_quat_t = np.roll(base_quat_t, 1)
        base_mat_t = quat2mat(base_quat_t)
        base_T_mat = compose(T = base_pos_t, R = base_mat_t, Z = np.ones(3))
        dof_pos_t = dof_pos[traj, t].numpy()
        
        pin.forwardKinematics(model, data, dof_pos_t)
        pin.updateFramePlacements(model, data)
        # get foot positions
        foot_pos_list = []
        for i in range(4):
            foot_frame = model.getFrameId(f"{legs[i]}_FOOT")
            foot_pos = data.oMf[foot_frame].translation
            foot_pos_homo = base_T_mat @ np.append(foot_pos, 1)
            foot_pos_list.append(foot_pos_homo[:-1])
        # get shoulder positions
        shoulder_pos_list = []
        for i in range(4):
            shoulder_frame = model.getFrameId(f"{legs[i]}_UPPER_LEG")
            shoulder_pos = data.oMf[shoulder_frame].translation
            shoulder_pos_homo = base_T_mat @ np.append(shoulder_pos, 1)
            shoulder_pos_list.append(shoulder_pos_homo[:-1])
        foot_pos_list = np.array(foot_pos_list)
        foot_pos_list = foot_pos_list.reshape(-1)
        shoulder_pos_list = np.array(shoulder_pos_list)
        shoulder_pos_list = shoulder_pos_list.reshape(-1)
        base_height_t = base_height_t[0]
        data_txt = np.concatenate((base_pos_t, shoulder_pos_list, foot_pos_list), axis=None)
        # import pdb; pdb.set_trace()
        data_txt_buffer.append(data_txt)
    save_data_buffer = np.array(data_txt_buffer)
    # filename = "motion_data_{traj}.txt"
    # np.savetxt(filename, save_data_buffer, fmt='%5f', delimiter=',')
    data_txt_buffer = []
    

