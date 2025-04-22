from dataclasses import dataclass, field
import numpy as np
from transforms3d.euler import euler2quat
from pathlib import Path


@dataclass
class DatasetConfig:
    # Basic info
    name: str
    data_path: str
    frame_rate: int
    motions: list
    traj_format: list
    scale_factor: float
    skew_factor: float
    num_markers: int
    contact_threshold: float
    stance_height: float
    
    # IDs
    ref_height_id: int
    ref_pelvis_id: int
    ref_neck_id: int
    ref_hips_id_list: list
    ref_toes_id_list: list

    # Offsets
    ref_coord_rot: list
    ref_root_rot: list
    ref_init_rot: np.array
    ref_pos_offset: np.array
    forward_dir_offset: np.array
    ref2sim_toes_offset_local: list
    sim_root_offset: np.array

    # Default arguments
    legs_name: list = field(default_factory=lambda: ["FL", "FR", "RL", "RR"])
    save_dir: Path = Path(__file__).parent.parent.parent.parent / "motion_retarget" / "go2_retarget"

    # Booleans
    heading_calibration: bool = True  # whether to calibrate the heading direction
    visu_contact_phase: bool = True
    save_data: bool = False
    visualize: bool = False
    save_task_space: bool = False


############################################################################################################
# Horse dataset
############################################################################################################
horse_dataset = DatasetConfig(
    name = 'horse',

    data_path= Path(__file__).parent.parent.parent.parent / "motion_retarget" / "datasets" / "horse_mocap" / "data",
    
    frame_rate=120,

    scale_factor=None,
    num_markers = 7,
    contact_threshold=0.01,
    skew_factor=0.75,
    stance_height=None,

    ref_height_id=5, # use neck_z as the reference height
    ref_pelvis_id=2,
    ref_neck_id=1,
    ref_hips_id_list=[1, 1, 2, 2], # FL, FR, RL, RR
    ref_toes_id_list=[3, 4, 5, 6], # FL, FR, RL, RR


    ref_coord_rot=euler2quat(0,0,0),
    ref_root_rot=euler2quat(0,0,0),
    ref_init_rot=np.array([1,0,0,0]), # w,x,y,z
    ref_pos_offset=np.array([0, 0, 0]),
    forward_dir_offset=np.array([0, 0, 0]),
    ref2sim_toes_offset_local=[
            np.array([0, 0.05, 0.00]), #FL
            np.array([0, -0.05, 0.00]), # FR
            np.array([0, 0.05, 0.00]), #RL
            np.array([0, -0.05, 0.00]) # RR
    ],
    sim_root_offset = np.array([0, 0, -0.02]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=False,
    visu_contact_phase=True,


    traj_format = ["HeadX","HeadY","HeadZ",
                   "WithersX","WithersY","WithersZ",
                   "SacrumX","SacrumY","SacrumZ",
                   "HFLX", "HFLY", "HFLZ",
                   "HFRX", "HFRY", "HFRZ",
                   "HHLX", "HHLY", "HHLZ",
                   "HHRX", "HHRY", "HHRZ",
                     ],
    motions=[
        #["save_as", "mocap_data.txt",start_frame,end_frame],
        ####################### walk #######################
        # ["walk_horse_1", "Horse1_M3_walk1_kinematics.txt",None,None],
        # ["walk_horse_2", "Horse2_M2_walk2_kinematics.txt",None,None],
        # ["walk_horse_3", "Horse3_M1_walk3_kinematics.txt",None,None],
        # ["walk_horse_4", "Horse4_M3_walk3_kinematics.txt",None,None],
        # ["walk_horse_5", "Horse5_M1_walk2_kinematics.txt",None,None],
        # ["walk_horse_6", "Horse6_M1_walk3_kinematics.txt",None,None],#
        # ["walk_horse_7", "Horse7_M1_walk1_kinematics.txt",None,None],
        # ["walk_horse_8", "Horse11_M1_walk3_kinematics.txt",None,None],
        # ["walk_horse_9", "Horse9_M1_walk1_kinematics.txt",None,None],#  
        # ["walk_horse_10", "Horse10_M1_walk1_kinematics.txt",None,None],

        # ####################### trot #######################
        ["trot_horse_1", "Horse1_M1_trot1_kinematics.txt",None,None],
        ["trot_horse_2", "Horse2_M1_trot2_kinematics.txt",None,None],
        ["trot_horse_3", "Horse3_M1_trot3_kinematics.txt",None,None],#
        ["trot_horse_4", "Horse4_M1_trot1_kinematics.txt",None,None],
        ["trot_horse_5", "Horse5_M1_trot2_kinematics.txt",None,None],
        ["trot_horse_6", "Horse6_M1_trot3_kinematics.txt",None,None],#
        ["trot_horse_7", "Horse7_M1_trot1_kinematics.txt",None,None],
        ["trot_horse_8", "Horse11_M1_trot3_kinematics.txt",None,None],
        ["trot_horse_9", "Horse9_M1_trot1_kinematics.txt",None,None],
        ["trot_horse_10", "Horse10_M1_trot1_kinematics.txt",None,None],


    ]
)

############################################################################################################
# Dog dataset
############################################################################################################
dog_dataset = DatasetConfig(
    name = 'dog',
    data_path= Path(__file__).parent.parent.parent.parent /"motion_retarget" / "datasets" / "dog_mocap" / "data",
    frame_rate=60,

    scale_factor=0.825,
    num_markers = 27,
    contact_threshold=0.03, # threshold for contact detection, has to be reasoneable by checking the plot from visu_contact_phase
    skew_factor=None,
    stance_height = 0.45,

    ref_height_id=None,
    ref_pelvis_id=0,
    ref_neck_id=3,
    ref_hips_id_list=[6, 11, 16, 20], # FL, FR, RL, RR
    ref_toes_id_list=[10, 15, 19, 23], # FL, FR, RL, RR


    ref_coord_rot= euler2quat(0.5*np.pi,0,0),
    ref_root_rot= euler2quat(0,0,0.47*np.pi),
    ref_init_rot=np.array([1,0,0,0]), # w,x,y,z
    ref_pos_offset=np.array([0, 0, 0]),
    forward_dir_offset=np.array([0, 0, 0]),
    ref2sim_toes_offset_local=[
            np.array([0, 0.10, -0.00]), #FL
            np.array([0, -0.10, -0.00]), # FR
            np.array([0, 0.10, 0.0]), #RL
            np.array([0, -0.10, 0.0]) # RR
    ],
    sim_root_offset = np.array([0, 0, -0.06]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=False,
    visu_contact_phase=True,
  


    traj_format = None, #information about the mocap data
    motions=[
        #["save_as", "mocap_data.txt",start_frame,end_frame],
            # ["pace_dog_1", "dog_walk00.txt", 90, 230],
            # ["pace_dog_2", "dog_walk00.txt", 210, 330],
            # ["pace_dog_3", "dog_walk00.txt", 330, 450],
            # ["pace_dog_4", "dog_walk00.txt", 450, None],
            # ["pace_dog_5","dog_walk18.txt",5100,None],
            # ["pace_dog_6", "dog_walk03.txt", 200, 375],
            # ["pace_dog_7", "dog_walk14.txt", 300, 500],
            # ["pace_dog_8", "dog_walk05.txt", 200, 320],
            # ["pace_dog_9","dog_walk05.txt", 320, 440],
            # ["pace_dog_10", "dog_sit02.txt", None, 120],

            # ["trot_dog_1", "dog_walk03.txt", 390, 510],
            # ["trot_dog_2", "dog_walk04.txt", 232, 352],
            # ["trot_dog_3", "dog_run04.txt", 487, 607],
            # ["trot_dog_4", "dog_walk06.txt", 170, 290],
            # ["trot_dog_5", "dog_walk16.txt", None, 150], #
            # ["trot_dog_6", "dog_stepup_jump3.txt", 628, None],
            # ["trot_dog_7", "dog_walk06.txt", 280, 400],
            # ["trot_dog_8", "dog_run04.txt", 594, 716],
            # ["trot_dog_9", "dog_walk04.txt", 352, 472],
            # ["trot_dog_10", "dog_walk04.txt", 422, None],

            # ["gallop_dog_1", "dog_run00.txt", 397, None],
            # ["gallop_dog_2", "dog_run01.txt", None, None],
            # ["gallop_dog_3", "dog_run02.txt", None, None],
            # ["gallop_dog_4", "dog_run03.txt", None, 80],
            # ["gallop_dog_5", "dog_run03.txt", 200, 300],
            # ["gallop_dog_6", "dog_walk_run00.txt", 750, None],
            # ["gallop_dog_7", "dog_walk10.txt", None, 120],
            # ["gallop_dog_8", "dog_walk10.txt", 220, 320],
            # ["gallop_dog_9", "dog_walk11.txt", 3000, None],
            # ["gallop_dog_10", "dog_stepup_jump2.txt", 362, None],
            # ['dog_beg00', 'dog_beg00.txt', 0, None],
            # ['dog_beg01', 'dog_beg01.txt', 0, None],
            # ['dog_hop00', 'dog_hop00.txt', 0, None],
            # ['dog_hop01', 'dog_hop01.txt', 0, None],
            # ['dog_jump_down0', 'dog_jump_down0.txt', 0, None],
            # ['dog_jump00', 'dog_jump00.txt', 0, None],
            # ['dog_jump01', 'dog_jump01.txt', 0, None],
            # ['dog_run00', 'dog_run00.txt', 0, None],
            # ['dog_run01', 'dog_run01.txt', 0, None],
            # ['dog_run02', 'dog_run02.txt', 0, None],
            # ['dog_run03', 'dog_run03.txt', 0, None],
            # ['dog_run04', 'dog_run04.txt', 0, None],
            # ['dog_shake0', 'dog_shake0.txt', 0, None],
            # ['dog_sit_stand00', 'dog_sit_stand00.txt', 0, None],
            # ['dog_sit_stand01', 'dog_sit_stand01.txt', 0, None],
            # ['dog_sit_stand02', 'dog_sit_stand02.txt', 0, None],
            # ['dog_sit_stand03', 'dog_sit_stand03.txt', 0, None],
            # ['dog_sit_stand04', 'dog_sit_stand04.txt', 0, None],
            # ['dog_sit_walk00', 'dog_sit_walk00.txt', 0, None],
            # ['dog_sit_walk01', 'dog_sit_walk01.txt', 0, None],
            # ['dog_sit_walk02', 'dog_sit_walk02.txt', 0, None],
            # ['dog_sit00', 'dog_sit00.txt', 0, None],
            # ['dog_sit01', 'dog_sit01.txt', 0, None],
            # ['dog_sit02', 'dog_sit02.txt', 0, None],
            # ['dog_stepup_jump0', 'dog_stepup_jump0.txt', 0, None],
            # ['dog_stepup_jump1', 'dog_stepup_jump1.txt', 0, None],
            # ['dog_stepup_jump2', 'dog_stepup_jump2.txt', 0, None],
            # ['dog_stepup_jump3', 'dog_stepup_jump3.txt', 0, None],
            # ['dog_turn00', 'dog_turn00.txt', 0, None],
            # ['dog_walk_run00', 'dog_walk_run00.txt', 0, None],
            # ['dog_walk_run01', 'dog_walk_run01.txt', 0, None],
            # ['dog_walk_run02', 'dog_walk_run02.txt', 0, None],
            # ['dog_walk00', 'dog_walk00.txt', 0, None],
            # ['dog_walk01', 'dog_walk01.txt', 0, None],
            # ['dog_walk02', 'dog_walk02.txt', 0, None],
            # ['dog_walk03', 'dog_walk03.txt', 0, None],
            # ['dog_walk04', 'dog_walk04.txt', 0, None],
            # ['dog_walk05', 'dog_walk05.txt', 0, None],
            # ['dog_walk06', 'dog_walk06.txt', 0, None],
            # ['dog_walk07', 'dog_walk07.txt', 0, None],
            # ['dog_walk08', 'dog_walk08.txt', 0, None],
            # ['dog_walk09', 'dog_walk09.txt', 0, None],
            # ['dog_walk10', 'dog_walk10.txt', 0, None],
            # ['dog_walk11', 'dog_walk11.txt', 0, None],
            # ['dog_walk12', 'dog_walk12.txt', 0, None],
            # ['dog_walk13', 'dog_walk13.txt', 0, None],
            # ['dog_walk14', 'dog_walk14.txt', 0, None],
            # ['dog_walk15', 'dog_walk15.txt', 0, None],
            # ['dog_walk16', 'dog_walk16.txt', 0, None],
            # ['dog_walk17', 'dog_walk17.txt', 0, None],
            # ['dog_walk18', 'dog_walk18.txt', 0, None],



    ]
)
############################################################################################################
# MPC dataset
############################################################################################################
mpc_dataset = DatasetConfig(
    name = 'mpc',
    data_path= Path(__file__).parent.parent.parent.parent / "motion_retarget" /"datasets" / "rf_mpc" / "data",
    frame_rate=100,

    scale_factor=1.0,
    num_markers = 9,
    contact_threshold=0.03,
    skew_factor=None,
    stance_height = 0.29,


    ref_height_id=None,
    ref_pelvis_id=None,
    ref_neck_id=None,
    ref_hips_id_list=[5, 6, 7, 8], # FL, FR, RL, RR
    ref_toes_id_list=[0, 1, 2, 3], # FL, FR, RL, RR


    ref_coord_rot=euler2quat(0,0,0),
    ref_root_rot=euler2quat(0,0,0),
    ref_init_rot=np.array([1,0,0,0]), # w,x,y,z
    ref_pos_offset=np.array([0, 0, 0]),
    forward_dir_offset=np.array([0, 0, 0]),
    ref2sim_toes_offset_local=[
            np.array([0, 0.03, 0.00]), #FL
            np.array([0, -0.03, 0.00]), # FR
            np.array([0, 0.03, 0.00]), #RL
            np.array([0, -0.03, 0.00]) # RR
    ],
    sim_root_offset = np.array([0, 0, 0.07]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=True,
    visu_contact_phase=True,


    traj_format = None,
    motions=[
        #["save_as", "mocap_data.txt",start_frame,end_frame],

        ######bound######
        # ["bound_mpc_1" ,"bound_0.6_0.1_50.txt" ,None, None],
        # ["bound_mpc_2" ,"bound_0.6_0.3_50.txt" ,None, None],
        # ["bound_mpc_3" ,"bound_0.6_0.4_50.txt" ,None, None],
        # ["bound_mpc_4" ,"bound_0.6_0.6_50.txt" ,None, None],
        # ["bound_mpc_5" ,"bound_0.7_0.4_50.txt" ,None, None],
        # ["bound_mpc_6" ,"bound_0.7_0.7_50.txt" ,None, None],
        # ["bound_mpc_7" ,"bound_0.8_0.2_50.txt" ,None, None],
        # ["bound_mpc_8" ,"bound_0.8_0.4_50.txt" ,None, None],
        # ["bound_mpc_9" ,"bound_0.8_0.6_50.txt" ,None, None],
        # ["bound_mpc_10" ,"bound_0.8_0.8_50.txt" ,None, None],


        ######trot######
        # ["trot_mpc_1" ,"trot_0.8_0.4_50.txt" ,None, None],
        # ["trot_mpc_2" ,"trot_0.8_0.7_50.txt" ,None, None],
        # ["trot_mpc_3" ,"trot_0.8_0.8_50.txt" ,None, None],
        # ["trot_mpc_4" ,"trot_1.2_0.8_50.txt" ,None, None],
        # ["trot_mpc_5" ,"trot_1.2_1.1_50.txt" ,None, None],
        # ["trot_mpc_6" ,"trot_1.2_1.2_50.txt" ,None, None],
        # ["trot_mpc_7" ,"trot_1.2_1_50.txt" ,None, None],
        # ["trot_mpc_8" ,"trot_1_0.5_50.txt" ,None, None],
        # ["trot_mpc_9" ,"trot_1_0.7_50.txt" ,None, None],
        # ["trot_mpc_10" ,"trot_1_1_50.txt" ,None, None],

        # ["crawl_mpc_1" ,"crawl_0.2_0.2_50.txt" ,None, None],
        # ["crawl_mpc_2" ,"crawl_0.3_0.3_50.txt" ,None, None],
        # ["crawl_mpc_3" ,"crawl_0.4_0.3_50.txt" ,None, None],
        # ["crawl_mpc_4" ,"crawl_0.4_0.4_50.txt" ,None, None],
        # ["crawl_mpc_5" ,"crawl_0.5_0.4_50.txt" ,None, None],
        # ["crawl_mpc_6" ,"crawl_0.5_0.5_50.txt" ,None, None],
        # ["crawl_mpc_7" ,"crawl_0.5_0.6_50.txt" ,None, None],
        # ["crawl_mpc_8" ,"crawl_0.6_0.3_50.txt" ,None, None],
        # ["crawl_mpc_9" ,"crawl_0.6_0.6_50.txt" ,None, None],
        # ["crawl_mpc_10" ,"crawl_0.6_0.8_50.txt" ,None, None],

        # ["gallop_mpc_1" ,"gallop_1.5_1.2_50.txt" ,None, None],
        # ["gallop_mpc_2" ,"gallop_1.5_1.5_50.txt" ,None, None],
        # ["gallop_mpc_3" ,"gallop_1.5_1.8_50.txt" ,None, None],
        # ["gallop_mpc_4" ,"gallop_1.5_1_50.txt" ,None, None],
        # ["gallop_mpc_5" ,"gallop_1.8_1.8_50.txt" ,None, None],
        # ["gallop_mpc_6" ,"gallop_1_1_50.txt" ,None, None],
        # ["gallop_mpc_7" ,"gallop_2.1_2.1_50.txt" ,None, None],
        # ["gallop_mpc_8" ,"gallop_2_1.8_50.txt" ,None, None],
        # ["gallop_mpc_9" ,"gallop_2_1.9_50.txt" ,None, None],
        # ["gallop_mpc_10" ,"gallop_2_2.1_50.txt" ,None, None],
        # ["gallop_mpc_11" ,"gallop_2_2_50.txt" ,None, None],

        ["pace_mpc_1" ,"pace_0.6_0.3_50.txt" ,None, None],
        ["pace_mpc_2" ,"pace_0.6_0.6_50.txt" ,None, None],
        # ["pace_mpc_3" ,"pace_0.6_1_50.txt" ,None, None],
        # ["pace_mpc_4" ,"pace_0.75_0.4_50.txt" ,None, None],
        # ["pace_mpc_5" ,"pace_0.75_1_50.txt" ,None, None],
        # ["pace_mpc_6" ,"pace_0.7_0.4_50.txt" ,None, None],
        # ["pace_mpc_7" ,"pace_0.7_0.8_50.txt" ,None, None],
        # ["pace_mpc_8" ,"pace_0.8_0.5_50.txt" ,None, None],
        # ["pace_mpc_9" ,"pace_0.8_0.8_50.txt" ,None, None],
        # ["pace_mpc_10" ,"pace_0.8_1_50.txt" ,None, None],



    ]
)
############################################################################################################
# Solo8 dataset
############################################################################################################

solo8_dataset = DatasetConfig(
    name = "solo8",
    data_path= Path(__file__).parent.parent.parent.parent / "motion_retarget" /"datasets" / "solo8" / "data",
    frame_rate=50,

    scale_factor=1.0,
    num_markers = 9,
    contact_threshold=0.03,
    skew_factor=None,
    stance_height = 0.24, #https://is.mpg.de/news/four-legged-robot-makes-research-comparable-worldwide

    ref_height_id=None,
    ref_pelvis_id=None,
    ref_neck_id=None,
    ref_hips_id_list=[1, 2, 3, 4], # FL, FR, RL, RR
    ref_toes_id_list=[5, 6, 7, 8], # FL, FR, RL, RR


    ref_coord_rot=euler2quat(0,0,0),
    ref_root_rot=euler2quat(0,0,0),
    ref_init_rot=np.array([1,0,0,0]), # w,x,y,z
    ref_pos_offset=np.array([0, 0, 0]),
    forward_dir_offset=np.array([0, 0, 0]),
    ref2sim_toes_offset_local=
    [
            np.array([0.08, 0.07, 0.00]), #FL
            np.array([0.08, -0.07, -0.00]), # FR
            np.array([-0.05, 0.07, -0.00]), #RL
            np.array([-0.05, -0.07, -0.00]) # RR
    ]
    ,
    sim_root_offset = np.array([0, 0, 0.07]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=False,
    visu_contact_phase=True,


    traj_format = None,
    motions=[
 
            ################STILT#####################
            
          ['stilt_solo8_1', 'motion_data_1004.txt', 10, None],
          ['stilt_solo8_2', 'motion_data_1009.txt', 10, None],
          ['stilt_solo8_3', 'motion_data_1011.txt', 10, None],
          ['stilt_solo8_4', 'motion_data_1029.txt', 10, None],
          ['stilt_solo8_5', 'motion_data_103.txt', 10, None],
          ['stilt_solo8_6', 'motion_data_1040.txt', 10, None],
          ['stilt_solo8_7', 'motion_data_1045.txt', 10, None],
          ['stilt_solo8_8', 'motion_data_1049.txt', 10, None],
          ['stilt_solo8_9', 'motion_data_1105.txt', 10, None],
          ['stilt_solo8_10', 'motion_data_1120.txt', 10, None],

          
            # #################LEAP######################
           
            ['bound_solo8_1', 'motion_data_1136.txt', 10, None],
            ['bound_solo8_2', 'motion_data_1144.txt', 10, None],
            ['bound_solo8_3', 'motion_data_1001.txt', 10, None],
            ['bound_solo8_4', 'motion_data_1089.txt', 10, None],
            ['bound_solo8_5', 'motion_data_1218.txt', 10, None],
            ['bound_solo8_6', 'motion_data_1272.txt', 10, None],
            ['bound_solo8_7', 'motion_data_1280.txt', 10, None],
            ['bound_solo8_8', 'motion_data_1311.txt', 10, None],
            ['bound_solo8_9', 'motion_data_1352.txt', 10, None],
            ['bound_solo8_10', 'motion_data_1482.txt', 10, None],


 
    ]
)



solo8_dataset_crawl = DatasetConfig(
    name = "solo8",
    data_path= Path(__file__).parent.parent.parent.parent / "motion_retarget" /"datasets" / "solo8" / "data",
    frame_rate=50,

    scale_factor=1.0,
    num_markers = 9,
    contact_threshold=0.03,
    skew_factor=None,
    stance_height=0.24, #https://is.mpg.de/news/four-legged-robot-makes-research-comparable-worldwide

    ref_height_id=None,
    ref_pelvis_id=None,
    ref_neck_id=None,
    ref_hips_id_list=[1, 2, 3, 4], # FL, FR, RL, RR
    ref_toes_id_list=[5, 6, 7, 8], # FL, FR, RL, RR


    ref_coord_rot=euler2quat(0,0,0),
    ref_root_rot=euler2quat(0,0,0),
    ref_init_rot=np.array([1,0,0,0]), # w,x,y,z
    ref_pos_offset=np.array([0, 0, 0]),
    forward_dir_offset=np.array([0, 0, 0]),
    ref2sim_toes_offset_local=
    [
            np.array([0.08, 0.07, 0.00]), #FL
            np.array([0.08, -0.07, -0.00]), # FR
            np.array([-0.05, 0.07, -0.00]), #RL
            np.array([-0.05, -0.07, -0.00]) # RR
    ]
    ,
    sim_root_offset = np.array([0, 0, 0.1]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=False,
    visu_contact_phase=True,


    traj_format = None,
    motions=[
        #["save_as", "mocap_data.txt",start_frame,end_frame],
        ##################CRAWL##################### 

        ['crawl_solo8_1', 'motion_data_1002.txt', 10, None],
        ['crawl_solo8_2', 'motion_data_1003.txt', 10, None],
        ['crawl_solo8_3', 'motion_data_1015.txt', 10, None],
        ['crawl_solo8_4', 'motion_data_1017.txt', 10, None],
        ['crawl_solo8_5', 'motion_data_1027.txt', 10, None],
        ['crawl_solo8_6', 'motion_data_1079.txt', 10, None],
        ['crawl_solo8_7', 'motion_data_1081.txt', 10, None],
        ['crawl_solo8_8', 'motion_data_1143.txt', 10, None],
        ['crawl_solo8_9', 'motion_data_1145.txt', 10, None],
        ['crawl_solo8_10', 'motion_data_1146.txt', 10, None],

    ]


)

solo8_dataset_trot = DatasetConfig(
    name = "solo8",
    data_path= Path(__file__).parent.parent.parent.parent / "motion_retarget" /"datasets" / "solo8" / "data",
    frame_rate=50,

    scale_factor=1.0,
    num_markers = 9,
    contact_threshold=0.03,
    skew_factor=None,
    stance_height= 0.24, #https://is.mpg.de/news/four-legged-robot-makes-research-comparable-worldwide

    ref_height_id=None,
    ref_pelvis_id=None,
    ref_neck_id=None,
    ref_hips_id_list=[1, 2, 3, 4], # FL, FR, RL, RR
    ref_toes_id_list=[5, 6, 7, 8], # FL, FR, RL, RR


    ref_coord_rot=euler2quat(0,0,0),
    ref_root_rot=euler2quat(0,0,0),
    ref_init_rot=np.array([1,0,0,0]), # w,x,y,z
    ref_pos_offset=np.array([0, 0, 0]),
    forward_dir_offset=np.array([0, 0, 0]),
    ref2sim_toes_offset_local=
    [
            np.array([0.08, 0.03, 0.00]), #FL
            np.array([0.08, -0.03, -0.00]), # FR
            np.array([-0.05, 0.03, -0.00]), #RL
            np.array([-0.05, -0.03, -0.00]) # RR
    ]
    ,
    sim_root_offset = np.array([0, 0, 0.1]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=False,
    visu_contact_phase=True,


    traj_format = None,
    motions=[
        #["save_as", "mocap_data.txt",start_frame,end_frame],

        #####################Walk##########################
        ['walk_solo8_1', 'motion_data_1026.txt', 10, None],
        ['walk_solo8_2', 'motion_data_1030.txt', 10, None],
        ['walk_solo8_3', 'motion_data_1032.txt', 10, None],
        ['walk_solo8_4', 'motion_data_1038.txt', 10, None],
        ['walk_solo8_5', 'motion_data_1043.txt', 10, None],
        ['walk_solo8_6', 'motion_data_1048.txt', 10, None],
        ['walk_solo8_7', 'motion_data_105.txt', 10, None],
        ['walk_solo8_8', 'motion_data_1057.txt', 10, None],
        ['walk_solo8_9', 'motion_data_1074.txt', 10, None],
        ['walk_solo8_10', 'motion_data_108.txt', 10, None],


        #####################TROT######################
        ['trot_solo8_1', 'motion_data_325.txt', 10, None],
        ['trot_solo8_2', 'motion_data_1005.txt', 10, None],
        ['trot_solo8_3', 'motion_data_1008.txt', 10, None],
        ['trot_solo8_4', 'motion_data_1014.txt', 10, None],
        ['trot_solo8_5', 'motion_data_1018.txt', 10, None],
        ['trot_solo8_6', 'motion_data_1019.txt', 10, None],
        ['trot_solo8_7', 'motion_data_1022.txt', 10, None],
        ['trot_solo8_8', 'motion_data_1034.txt', 10, None],
        ['trot_solo8_9', 'motion_data_1037.txt', 10, None],
        ['trot_solo8_10', 'motion_data_1039.txt', 10, None],



                
    ]

    
)


solo8_dataset_wave = DatasetConfig(
    name = "solo8",
    data_path= Path(__file__).parent.parent.parent.parent / "motion_retarget" /"datasets" / "solo8" / "data",
    frame_rate=50,

    scale_factor=1.0,
    num_markers = 9,
    contact_threshold=0.03,
    skew_factor=None,
    stance_height=0.24, #https://is.mpg.de/news/four-legged-robot-makes-research-comparable-worldwide

    ref_height_id=None,
    ref_pelvis_id=None,
    ref_neck_id=None,
    ref_hips_id_list=[1, 2, 3, 4], # FL, FR, RL, RR
    ref_toes_id_list=[5, 6, 7, 8], # FL, FR, RL, RR


    ref_coord_rot=euler2quat(0,0,0),
    ref_root_rot=euler2quat(0,0,0),
    ref_init_rot=np.array([1,0,0,0]), # w,x,y,z
    ref_pos_offset=np.array([0, 0, 0]),
    forward_dir_offset=np.array([0, 0, 0]),
    ref2sim_toes_offset_local=
    [
            np.array([0.08, 0.07, -0.0]), #FL
            np.array([0.08, -0.07, -0.0]), # FR
            np.array([-0.05, 0.07, -0.00]), #RL
            np.array([-0.05, -0.07, -0.00]) # RR
    ]
    ,
    sim_root_offset = np.array([0, 0, 0.07]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=False,
    visu_contact_phase=True,


    traj_format = None,
    motions=[
        #["save_as", "mocap_data.txt",start_frame,end_frame],
        ######################WAVE#########################
        ['wave_solo8_1', 'motion_data_1023.txt', 10, None],
        ['wave_solo8_2', 'motion_data_1053.txt', 10, None],
        ['wave_solo8_3', 'motion_data_1063.txt', 10, None],
        ['wave_solo8_4', 'motion_data_1087.txt', 10, None],
        ['wave_solo8_5', 'motion_data_1116.txt', 10, None],
        ['wave_solo8_6', 'motion_data_1130.txt', 10, None],
        ['wave_solo8_7', 'motion_data_1180.txt', 10, None],
        ['wave_solo8_8', 'motion_data_1182.txt', 10, None],
        ['wave_solo8_9', 'motion_data_1209.txt', 10, None],
        ['wave_solo8_10', 'motion_data_121.txt', 10, None],
   
    ]

    
)