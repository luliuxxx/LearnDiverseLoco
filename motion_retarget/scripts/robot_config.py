
from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass
class RobotConfig:
    name: str
    urdf_path: str
    xml_path: str
    stance_height: float
    default_pose: np.array
    # root offset for simulation
    mujoco_root_offset: np.array
    hip_joint_ids: list # FL, RL, FR, RR


unitree_a1 = RobotConfig(
    name="unitree_a1",
    urdf_path= Path(__file__).parent.parent / "robots" / "a1" / "urdf" / "a1.urdf",
    xml_path= Path(__file__).parent.parent / "robots" / "a1" / "xml" / "plane.xml",

    stance_height=0.32,
    default_pose = np.array(([0,0,0.32,0.,0.,0.,1.0,0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])), # for pinocchio, quat in xyzw
    
    mujoco_root_offset = np.array([0, 0, 0.43]),
    hip_joint_ids = [9, 12, 15, 18] #
    #mujoco_joint_order = ["FR", "FL", "RR", "RL"]
)

unitree_go2 = RobotConfig(
    name="unitree_go2",
    urdf_path= Path(__file__).parent.parent / "robots" / "go2" / "urdf" / "go2.urdf",
    xml_path= Path(__file__).parent.parent  / "robots" / "go2" / "xml" / "plane.xml",

    stance_height=0.325,
    default_pose = np.array(([0, 0, 0.325, 0, 0, 0, 1, 0.1, 0.8, -1.5, -0.1, -0.8, 1.5, -0.1, 1.0, -1.5, 0.1, -1.0, 1.5])), # for pinocchio, quat in xyzw
    
    mujoco_root_offset = np.array([0, 0, 0.0]),
    hip_joint_ids = [9, 12, 15, 18] #
    
)