#process raw mocap horse data
#including
# - delete incomplete rows
# - get the moving velocities
# - apply the velocity

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from pathlib import Path
#optimization library
from casadi import *
# constants
FL_idx = [9,10,11]
FR_idx = [12,13,14]
BL_idx = [15,16,17]
BR_idx = [18,19,20]
NECK_idx = [3,4,5]
PELVIS_idx = [6,7,8]
HEAD_idx = [0,1,2]
legs_idx = [FL_idx, FR_idx, BL_idx, BR_idx]
legs_name = ["FL", "FR", "BL", "BR"]

# load raw mocap data
def load_mocap_data(csv_path):
    return pd.read_csv(csv_path, delimiter=',')

# filter out incomplete rows
# have to make sure we keep the longest consectuive sequence of complete rows
def filter_incomplete_rows(df,csv_file):
    # start from beginning, count the number of consecutive complete rows
    if df.isnull().sum().sum() > 0:
        count = 0
        max_count = 0
        record_sequence = {}
        for i in range(len(df)):
            if df.iloc[i].isnull().sum() == 0:
                count += 1
            else:
                max_count = max(max_count, count)
                record_sequence[i] = count
                count = 0
        # get the row index of the longest sequence
        re = max(record_sequence, key=record_sequence.get)
        df = df.iloc[re-max_count:re]
        df.to_csv(csv_file, index=False)
        new_df = load_mocap_data(csv_file)
    else:
        new_df = df
    return new_df

def preprocess_mocap_data(csv_file):
    # out a txt file
    # load the data
    df = load_mocap_data(csv_file)
    # filter out incomplete rows
    new_df = filter_incomplete_rows(df,csv_file)
    # save the processed data
    #check if the data is complete
    if new_df.isnull().sum().sum() > 0:
        print(f"There is still incomplete rows, please check {csv_file}!")
    #apply scaling factor
    data_arr = new_df.iloc[:,2:].to_numpy() # get the data array
    data_arr = data_arr * 0.001 # apply the scaling factor
    return data_arr

def find_contact_phase(data, height_threshold=0.005):
    contact_phase = [[] for _ in range(len(legs_name))]
    for i in range(len(data)):
        for id, leg in enumerate(legs_idx):
            if data[i, leg[-1]] < height_threshold: # check the height of the leg
                contact_phase[id].append(1) # contact phase
            else:
                contact_phase[id].append(0) # swing phase
    return contact_phase

def visualize_velocity(base_vel_sol, timesteps):
    plt.figure()
    plt.plot(np.arange(timesteps), base_vel_sol[0,:], label='base_vel_x')
    plt.plot(np.arange(timesteps), base_vel_sol[1,:], label='base_vel_y')
    # plt.plot(np.arange(timesteps), base_vel_sol[2,:], label='base_vel_z')
    plt.legend()
    plt.show()

def get_moving_velocity(data,isVisualize=False):
    contact_buffer = find_contact_phase(data)
    opti = Opti()
    # symbolic variables
    timesteps = data.shape[0]
    smooth_cost_factor = 1.5
    base_vel_var = opti.variable(2, timesteps)
    objective = 0
    for t in range(timesteps):
        smooth_cost = 0
        if t > 0:
            smooth_cost = smooth_cost_factor * sum1((base_vel_var[:,t] - base_vel_var[:,t-1])**2)
        objective += smooth_cost
        next_frame =  t + 1
        curr_frame =  t
        if next_frame >= timesteps:
            next_frame = timesteps - 1
        
        # get trunk z velocity
        # curr_z_pos = (data[curr_frame,NECK_idx[-1]] + data[curr_frame,PELVIS_idx[-1]])*0.5
        # next_z_pos = (data[next_frame,NECK_idx[-1]]+ data[next_frame,PELVIS_idx[-1]])*0.5
        # trunk_z_vel = (next_z_pos - curr_z_pos) 

        for foot in range(len(legs_name)): 
            isContact = contact_buffer[foot][t]
            foot_vel_xyz = data[next_frame,legs_idx[foot]] - data[curr_frame,legs_idx[foot]]
            # replace the z velocity with the trunk z velocity
            foot_vel_xy = foot_vel_xyz[:2].reshape(2,-1)
            diff = foot_vel_xy + base_vel_var[:,t]
            objective += (diff.T@diff) * isContact
    opti.minimize(objective)
    opti.solver('ipopt')
    sol = opti.solve()
    base_vel_sol = sol.value(base_vel_var)
    if isVisualize:
        visualize_velocity(base_vel_sol, timesteps)
    return base_vel_sol

def apply_velocity(data, base_vel):
    len_frames = data.shape[0]
    copy_data = data.copy()
    idx_list = np.arange(0, copy_data.shape[-1], 1)
    for frame in range(1,len_frames):
        x_vel = base_vel[0,frame-1]
        y_vel = base_vel[1,frame-1]
        # z_vel = base_vel[2,frame-1]
        rel_pos_list = []
        for foot in range(len(legs_name)):
            rel_foot_pos = data[frame,legs_idx[foot]] - data[frame,NECK_idx]
            rel_pos_list.append(rel_foot_pos)
        
        # head rel pos
        rel_pos = data[frame,HEAD_idx] - data[frame,NECK_idx]
        rel_pos_list.append(rel_pos)

        # pelvis rel pos
        rel_pos = data[frame,PELVIS_idx] - data[frame,NECK_idx]
        rel_pos_list.append(rel_pos)

        # update the neck position
        data[frame,NECK_idx] = data[frame-1,NECK_idx] + np.array([x_vel, y_vel, 0.0])

        for foot in range(len(legs_name)):
            data[frame,legs_idx[foot]] = data[frame,NECK_idx] + rel_pos_list[foot]
        # head
        data[frame,HEAD_idx] = data[frame,NECK_idx] + rel_pos_list[-2]
        # pelvis
        data[frame,PELVIS_idx] = data[frame,NECK_idx] + rel_pos_list[-1]
    #keep z unchanged
    data[:,idx_list[2::3]] = copy_data[:,idx_list[2::3]].copy()
    return data

def save_as_txt(data, filename):
    # in ordering
    # head, neck, pelvis, FL, FR, BL, BR
    # x, y, z
    np.savetxt(filename, data, fmt='%5f', delimiter=',')




def main():
    # get all the csv files
    mocap_path = str(Path(os.path.abspath(__file__)).parent) + '/raw_mocap_data/*.csv'
    save_path = str(Path(os.path.abspath(__file__)).parent) + '/data'
    csv_files = glob.glob(mocap_path)
    need_cal_vel = True
    for csv_file in csv_files:
        print("Processing: ", csv_file)
        mocap_data = preprocess_mocap_data(csv_file)
        if mocap_data.shape[0] == 0:
            os.remove(csv_file)
            continue
        if need_cal_vel:
            base_vel = get_moving_velocity(mocap_data)
            new_mocap_data = apply_velocity(mocap_data, base_vel)
            save_as_txt(new_mocap_data, save_path + f"/{csv_file.split('/')[-1][:-4]}" + '.txt')


if __name__ == "__main__":
    main()