import numpy as np
from transforms3d.affines import compose, decompose44
import scipy
import glob
from pathlib import Path
# we only need positions
# Load the array from the MATLAB file
mat_file_path = str(Path(__file__).resolve().parents[1] / "mat_file")
mat_files = glob.glob(f"{mat_file_path}/*.mat")
N_joints = 9 # 4 toes 1 trunk 4 hips 
save_path = str(Path(__file__).resolve().parents[1] / "data")
for mat_file in mat_files:
    
    file_name = mat_file.split("/")[-1][:-4]
    # print(f"Processing {file_name}")
    mat_data = scipy.io.loadmat(mat_file)
    # SHAPE : 4 x 4 x N_LEGS x TIME
    Rs = mat_data['T_joints_hist']
    pos_hist = []
    for i in range(Rs.shape[3]):
        pos_list = []
        for j in range(N_joints):
            Tmat = Rs[:,:,j,i]
            pos, _, _, _ = decompose44(Tmat)
            # make sure pos is real

            pos = np.real(pos)
               
            pos_list.append(pos)
        pos_list = np.array(pos_list).flatten()
        pos_hist.append(pos_list)
    pos_hist = np.array(pos_hist)
    print(pos_hist.shape)
     # :5f is a format specifier for floating point numbers

    np.savetxt(f"{save_path}/{file_name}.txt", pos_hist, fmt='%5f', delimiter=',')
