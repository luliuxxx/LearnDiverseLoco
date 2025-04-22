import numpy as np


values_per_joint = 3
values_per_foot = 2

# Quadruped joints
QUADRUPED_FRONT_LEFT_HIP = np.arange(0, values_per_joint)
QUADRUPED_FRONT_LEFT_THIGH = np.arange(QUADRUPED_FRONT_LEFT_HIP[-1] + 1, QUADRUPED_FRONT_LEFT_HIP[-1] + 1 + values_per_joint)
QUADRUPED_FRONT_LEFT_CALF = np.arange(QUADRUPED_FRONT_LEFT_THIGH[-1] + 1, QUADRUPED_FRONT_LEFT_THIGH[-1] + 1 + values_per_joint)
QUADRUPED_FRONT_RIGHT_HIP = np.arange(QUADRUPED_FRONT_LEFT_CALF[-1] + 1, QUADRUPED_FRONT_LEFT_CALF[-1] + 1 + values_per_joint)
QUADRUPED_FRONT_RIGHT_THIGH = np.arange(QUADRUPED_FRONT_RIGHT_HIP[-1] + 1, QUADRUPED_FRONT_RIGHT_HIP[-1] + 1 + values_per_joint)
QUADRUPED_FRONT_RIGHT_CALF = np.arange(QUADRUPED_FRONT_RIGHT_THIGH[-1] + 1, QUADRUPED_FRONT_RIGHT_THIGH[-1] + 1 + values_per_joint)
QUADRUPED_BACK_LEFT_HIP = np.arange(QUADRUPED_FRONT_RIGHT_CALF[-1] + 1, QUADRUPED_FRONT_RIGHT_CALF[-1] + 1 + values_per_joint)
QUADRUPED_BACK_LEFT_THIGH = np.arange(QUADRUPED_BACK_LEFT_HIP[-1] + 1, QUADRUPED_BACK_LEFT_HIP[-1] + 1 + values_per_joint)
QUADRUPED_BACK_LEFT_CALF = np.arange(QUADRUPED_BACK_LEFT_THIGH[-1] + 1, QUADRUPED_BACK_LEFT_THIGH[-1] + 1 + values_per_joint)
QUADRUPED_BACK_RIGHT_HIP = np.arange(QUADRUPED_BACK_LEFT_CALF[-1] + 1, QUADRUPED_BACK_LEFT_CALF[-1] + 1 + values_per_joint)
QUADRUPED_BACK_RIGHT_THIGH = np.arange(QUADRUPED_BACK_RIGHT_HIP[-1] + 1, QUADRUPED_BACK_RIGHT_HIP[-1] + 1 + values_per_joint)
QUADRUPED_BACK_RIGHT_CALF = np.arange(QUADRUPED_BACK_RIGHT_THIGH[-1] + 1, QUADRUPED_BACK_RIGHT_THIGH[-1] + 1 + values_per_joint)
# # Quadruped feet
# QUADRUPED_FRONT_LEFT_FOOT = np.arange(QUADRUPED_SPINE[-1] + 1, QUADRUPED_SPINE[-1] + 1 + values_per_foot)
# QUADRUPED_FRONT_RIGHT_FOOT = np.arange(QUADRUPED_FRONT_LEFT_FOOT[-1] + 1, QUADRUPED_FRONT_LEFT_FOOT[-1] + 1 + values_per_foot)
# QUADRUPED_BACK_LEFT_FOOT = np.arange(QUADRUPED_FRONT_RIGHT_FOOT[-1] + 1, QUADRUPED_FRONT_RIGHT_FOOT[-1] + 1 + values_per_foot)
# QUADRUPED_BACK_RIGHT_FOOT = np.arange(QUADRUPED_BACK_LEFT_FOOT[-1] + 1, QUADRUPED_BACK_LEFT_FOOT[-1] + 1 + values_per_foot)
# General
TRUNK_LINEAR_VELOCITIES = np.arange(QUADRUPED_BACK_RIGHT_CALF[-1] + 1, QUADRUPED_BACK_RIGHT_CALF[-1] + 1 + 3)
TRUNK_ANGULAR_VELOCITIES = np.arange(TRUNK_LINEAR_VELOCITIES[-1] + 1, TRUNK_LINEAR_VELOCITIES[-1] + 1 + 3)
# GOAL_VELOCITIES = np.arange(TRUNK_ANGULAR_VELOCITIES[-1] + 1, TRUNK_ANGULAR_VELOCITIES[-1] + 1 + 3)
PROJECTED_GRAVITY = np.arange(TRUNK_ANGULAR_VELOCITIES[-1] + 1, TRUNK_ANGULAR_VELOCITIES[-1] + 1 + 3)
HEIGHT = np.arange(PROJECTED_GRAVITY[-1] + 1, PROJECTED_GRAVITY[-1] + 1 + 1)

###################################CUSTOM###################################
# TRACKING
TRACKING_DIM = 3 + 4 + 3+ 12 + 12 + 12   # 49 trunk pos,0-2 trunk quat, joint pos, joint vel, toe pos
TARGET_TRACKING = np.arange(HEIGHT[-1] + 1, HEIGHT[-1] + 1 + TRACKING_DIM) # 57 + 1 = 58
# TRACK_TRUNK_POS = np.arange(HEIGHT[-1] + 1, HEIGHT[-1] + 1 + 3) # 0-2

TRACK_TRUNK_LIN_VEL = np.arange(HEIGHT[-1] + 1, HEIGHT[-1] + 1 + 3)#3-5
TRACK_TRUNK_QUAT = np.arange(TRACK_TRUNK_LIN_VEL[-1] + 1, TRACK_TRUNK_LIN_VEL[-1] + 1 + 4)#6-9
TRACK_TRUNK_ANG_VEL = np.arange(TRACK_TRUNK_QUAT[-1] + 1, TRACK_TRUNK_QUAT[-1] + 1 + 3)#10-12
TRACK_JOINT_POS = np.arange(TRACK_TRUNK_ANG_VEL[-1] + 1, TRACK_TRUNK_ANG_VEL[-1] + 1 + 12)#13-24
TRACK_JOINT_VEL = np.arange(TRACK_JOINT_POS[-1] + 1, TRACK_JOINT_POS[-1] + 1 + 12)#25-36
TRACK_TOE_POS = np.arange(TRACK_JOINT_VEL[-1] + 1, TRACK_JOINT_VEL[-1] + 1 + 12)

SOURCE_DIM = 4
GAIT_DIM = 8
# # source encoder: dog, horse, mpc, solo8 4
# # gait encoder: trot, pace, gallop, bound, walk, crawl, stilt, wave 8
TRACK_SOURCE = np.arange(TRACK_TOE_POS[-1] + 1, TRACK_TOE_POS[-1] + 1 + SOURCE_DIM)  # 58 + 1 = 59
TRACK_GAIT = np.arange(TRACK_SOURCE[-1] + 1, TRACK_SOURCE[-1] + 1 + GAIT_DIM)  # 59 + 4 = 63


##############################################################################

# Total observation size
OBSERVATION_SIZE = HEIGHT[-1] + 1 + TRACKING_DIM +  SOURCE_DIM + GAIT_DIM


def update_nr_height_samples(nr_height_samples):
    global HEIGHT
    HEIGHT = np.arange(PROJECTED_GRAVITY[-1] + 1, PROJECTED_GRAVITY[-1] + 1 + nr_height_samples)
    global OBSERVATION_SIZE
    OBSERVATION_SIZE = HEIGHT[-1] + 1 + TRACKING_DIM  + SOURCE_DIM + GAIT_DIM
