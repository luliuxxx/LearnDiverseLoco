# downsample the dataset from 200Hz to 100Hz
# only for type crawl, gallop, pace, trot_run

# list all txt file in data folder
import os
from pathlib import Path

data_path = str(Path(__file__).resolve().parents[1] / "data")
mocap_files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
for mocap_file in mocap_files:
    motion_name = mocap_file.split(".")[0].split("_")[0]
    if motion_name == 'trot' and 'run' in mocap_file:
        motion_name = 'trot_run'
    if motion_name in ['crawl', 'gallop', 'pace', 'trot_run']:
        print(motion_name)
        with open(os.path.join(data_path, mocap_file), 'r') as f:
            data = f.readlines()
        # downsample from 200Hz to 100Hz
        downsampled_data = data[::2]
        print("Original data length: ", len(data))
        print("Downsampled data length: ", len(downsampled_data))
        with open(os.path.join(data_path, mocap_file), 'w') as f:
            f.writelines(downsampled_data)
        print("Downsampled data saved as ", mocap_file)
