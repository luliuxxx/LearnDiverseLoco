# Motion Retarget

## Datasets
- Dog Mocap Dataset: [AI4Animation](https://github.com/sebastianstarke/AI4Animation)
- Horse Mocap Dataset: [University of Bonn](http://horse.cs.uni-bonn.de/dataset1.html)
- MPC Dataset: [Representation -Free Model Predictive Control for Dynamic Quadruped ](https://github.com/YanranDing/RF-MPC.git)
- Solo8 Dataset: [Versatile Skill Control via Self-supervised Adversarial Imitation of Unlabeled Mixed Motions ](https://arxiv.org/pdf/2209.07899)


## Motion Retargeting Method Reference
For motion retargeting, refer to the implementation available [here](https://github.com/erwincoumans/motion_imitation/blob/master/retarget_motion/retarget_motion.py).

## Robots
- Unitree A1
- Unitree Go2

## Running Retargeting
Before running retargeting, ensure that the dataset parameters are correctly configured in the respective config files:
e.g dog motion capture dataset:
- `motion_retarget/robots/{robot}/dataset_config.py`
Before running the script, please specify the variable mocap_motions in the configuration file. Here's an example:
eg:
```bash
motions =   [
    #["file_name_save_as", "mocap_data.txt",start_frame,end_frame],
    ["pace00", "dog_walk00.txt", 90, 210],
]
```
The script  will retarget the file "dog_walk00.txt", and the resulting retargeted motion will be saved as "pace00.txt".

The motion starts from frame 90 and ends at frame 210 in the file "dog_walk00.txt".

To run retargeting or visulize the retargeted data, you can simply run run_script.sh.
In the script, you can choose either dog or horse. the script will take corresponding config.py
```bash
sh run_script.sh
```

The retargeted data includes the following information, ordered as:
- `root_pos`, `root_quat`, `fl_hip_thigh_calf`, `fr_hip_thigh_calf`, `rl_hip_thigh_calf`, `rr_hip_thigh_calf`

The total dimensions are calculated as follows: \(3 + 4 + 4 * 3 = 19\), where `pos` represents `x`, `y`, `z`, and `quat` represents `w`, `x`, `y`, `z`.

By setting the boolean in `motion_retarget/scripts/dataset_config.py` you can enable the visualization and contact phase checking during retargeting

The already retarget data are in the folder: `motion_retarget/{robot}_retarget`