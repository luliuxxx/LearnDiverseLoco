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
    save_dir: Path = Path(__file__).parent.parent.parent.parent / "motion_retarget" / "a1_retarget"

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
    sim_root_offset = np.array([0, 0, -0.04]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=True,
    # visu_contact_phase=True,


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
        ["walk_horse_1", "Horse1_M3_walk1_kinematics.txt",None,None],
        ["walk_horse_2", "Horse2_M2_walk2_kinematics.txt",None,None],
        ["walk_horse_3", "Horse3_M1_walk3_kinematics.txt",None,None],
        ["walk_horse_4", "Horse4_M3_walk3_kinematics.txt",None,None],
        ["walk_horse_5", "Horse5_M1_walk2_kinematics.txt",None,None],
        ["walk_horse_6", "Horse6_M1_walk3_kinematics.txt",None,None],
        ["walk_horse_7", "Horse7_M1_walk1_kinematics.txt",None,None],
        ["walk_horse_8", "Horse11_M1_walk3_kinematics.txt",None,None],
        ["walk_horse_9", "Horse9_M1_walk1_kinematics.txt",None,None],
        ["walk_horse_10", "Horse10_M1_walk1_kinematics.txt",None,None],

        ####################### trot #######################
        ["trot_horse_1", "Horse1_M1_trot1_kinematics.txt",None,None],
        ["trot_horse_2", "Horse2_M1_trot2_kinematics.txt",None,None],
        ["trot_horse_3", "Horse3_M1_trot3_kinematics.txt",None,None],
        ["trot_horse_4", "Horse4_M1_trot1_kinematics.txt",None,None],
        ["trot_horse_5", "Horse5_M1_trot2_kinematics.txt",None,None],
        ["trot_horse_6", "Horse6_M1_trot3_kinematics.txt",None,None],
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
    visualize=True,
    visu_contact_phase=True,
  


    traj_format = None, #information about the mocap data
    motions=[
        #["save_as", "mocap_data.txt",start_frame,end_frame],
        ####################PACE###################
        ["pace_dog_1", "dog_walk00.txt", 90, None],
        ["pace_dog_2", "dog_walk00.txt", 210, 330],
        ["pace_dog_3", "dog_walk00.txt", 330, 450],
        ["pace_dog_4", "dog_walk00.txt", 450, None],
        ["pace_dog_5","dog_walk18.txt",5100,None],
        ["pace_dog_6", "dog_walk03.txt", 200, 375],
        ["pace_dog_7", "dog_walk14.txt", 300, 500],
        ["pace_dog_8", "dog_walk05.txt", 200, 320],
        ["pace_dog_9","dog_walk05.txt", 320, 440],
        ["pace_dog_10", "dog_sit02.txt", None, 120],

# pace
        # ["dog_walk00", "dog_walk00.txt", 90, None],    
        # ["dog_walk03", "dog_walk03.txt", 160, 375],
        # ["dog_walk05", "dog_walk05.txt", 200, 545],
        # ["dog_jump01", "dog_jump01.txt", 512, 700],
        # ["dog_walk_run02", "dog_walk_run02.txt", 2200, 2350],
        # ["dog_sit02", "dog_sit02.txt", None, 250],
        # ["dog_sit02", "dog_sit02.txt", 2550, 2700],#
        # ["dog_walk14", "dog_walk14.txt", 100, 300],# 

        # ["dog_sit_walk01", "dog_sit_walk01.txt", 4700, 4850],#         
        # ["dog_walk16", "dog_walk16.txt", 2150, 2400],
        # ["dog_walk16", "dog_walk16.txt", 2600, 2900],
        # ["dog_walk16", "dog_walk16.txt", 3150, None], 
        # ["dog_shake0", "dog_shake0.txt", 50, 450],
        # ["dog_stepup_jump1", "dog_stepup_jump1.txt", 180, 280],
        # ["dog_sit_walk02", "dog_sit_walk02.txt", None, 120],
        # ["dog_sit_walk02", "dog_sit_walk02.txt", 220, 340],
        # ["dog_walk14", "dog_walk14.txt", 300, 500],
     
     #trot
            ["dog_walk03", "dog_walk03.txt", 390, 510],
            # ["dog_walk04", "dog_walk04.txt", 232, 352],
            # ["dog_run04", "dog_run04.txt", 487, 607],
            # ["dog_walk06", "dog_walk06.txt", 170, 290],

            # ["dog_walk16", "dog_walk16.txt", None, 150], #
            # ["dog_stepup_jump3", "dog_stepup_jump3.txt", 628, None],
            # ["dog_walk06", "dog_walk06.txt", 280, 400],
            # ["dog_run04", "dog_run04.txt", 594, 716],
            # ["dog_walk04", "dog_walk04.txt", 352, 472],
            # ["dog_walk04", "dog_walk04.txt", 422, None],
     # gallop
            # ["dog_run00", "dog_run00.txt", 397, None],
            # ["dog_run01", "dog_run01.txt", None, None],
            # ["dog_run02", "dog_run02.txt", None, None],
            # ["dog_run03", "dog_run03.txt", None, 80],
            # ["dog_run03", "dog_run03.txt", 200, 300],
            # ["dog_walk_run00", "dog_walk_run00.txt", 750, None],
            # ["dog_walk10", "dog_walk10.txt", None, 120],
            # ["dog_walk10", "dog_walk10.txt", 220, 320],
            # ["dog_walk11", "dog_walk11.txt", 3000, None],
            # ["dog_stepup_jump2", "dog_stepup_jump2.txt", 362, None],
     
     # canter
            # ["dog_run03", "dog_run03.txt", 300, None],
            # ["dog_walk_run01", "dog_walk_run01.txt", None, 180],

            # left turn
            # ["dog_walk01", "dog_walk01.txt", 325, 400],
            # ["dog_walk01", "dog_walk01.txt", 430, 520],
            # ["dog_walk01", "dog_walk01.txt", 990, None],
            # ["dog_walk02", "dog_walk02.txt", 520, 620],
            # ["dog_walk02", "dog_walk02.txt", 730, 820],
            # ["dog_sit_walk01", "dog_sit_walk01.txt", 4550, 4650],
            # ["dog_walk16", "dog_walk16.txt", 1550, 1650],
            # ["dog_sit_walk02", "dog_sit_walk02.txt", 110, 230],
            # ["dog_walk18","dog_walk18.txt",4515,4740],
            # ["dog_walk18","dog_walk18.txt",545,765],
     
     #right turn
            # ["dog_walk01", "dog_walk01.txt", 587, 720],
            # ["dog_walk01", "dog_walk01.txt", 900, 990],
            # ["dog_walk02", "dog_walk02.txt", 430, 520],
            # ["dog_walk02", "dog_walk02.txt", 620, 730],
            # ["dog_walk02", "dog_walk02.txt", 815, None],
            # ["dog_walk09", "dog_walk09.txt", 1085,1124],
            # ["dog_walk_run02", "dog_walk_run02.txt", 2350, 2450],
            # ["dog_walk16", "dog_walk16.txt", 2450, 2600],
            # ["dog_walk16", "dog_walk16.txt", 3050, 3150],
            # ["dog_sit_walk02", "dog_sit_walk02.txt", 2840, 3000],
     
     #spin
            # ["dog_turn00", "dog_turn00.txt", 326, 747],
            # ["dog_walk08", "dog_walk08.txt", 280, 620],
            # ["dog_sit_walk01", "dog_sit_walk01.txt", 1700, 1850],
        
            # ["dog_hop00", "dog_hop00.txt", 2100, 2190],
            # ["dog_jump_down0", "dog_jump_down0.txt", 2700, 2790],
            # ["dog_jump_down0", "dog_jump_down0.txt", 5350, 5440],
            # ["dog_shake0", "dog_shake0.txt", 4250, 4340],
        
        
            # ["dog_hop00", "dog_hop00.txt", 3150, 3300],
            # ["dog_hop00", "dog_hop00.txt", 3980, 4100],
        
            # ["dog_jump00", "dog_jump00.txt", 950, 1100],
            # ["dog_jump01", "dog_jump01.txt", 1550, 1650],
            # ["dog_stepup_jump0", "dog_stepup_jump0.txt", 575, 671],
            # ["dog_stepup_jump1", "dog_stepup_jump1.txt", 390, None],
            # ["dog_stepup_jump2", "dog_stepup_jump2.txt", None, 100],
            # ["dog_stepup_jump3", "dog_stepup_jump3.txt", 477, 557],

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

    scale_factor=1.6,
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
            np.array([0, 0.00, 0.00]), #FL
            np.array([0, -0.00, 0.00]), # FR
            np.array([0, 0.00, 0.00]), #RL
            np.array([0, -0.00, 0.00]) # RR
    ],
    sim_root_offset = np.array([0, 0, -0.03]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=False,
    # visu_contact_phase=True,


    traj_format = None,
    motions=[
        #["save_as", "mocap_data.txt",start_frame,end_frame],

        ######bound######
        ["bound_mpc_1" ,"bound_0.6_0.1_50.txt" ,None, None],
        ["bound_mpc_2" ,"bound_0.6_0.3_50.txt" ,None, None],
        ["bound_mpc_3" ,"bound_0.6_0.4_50.txt" ,None, None],
        ["bound_mpc_4" ,"bound_0.6_0.6_50.txt" ,None, None],
        ["bound_mpc_5" ,"bound_0.7_0.4_50.txt" ,None, None],
        ["bound_mpc_6" ,"bound_0.7_0.7_50.txt" ,None, None],
        ["bound_mpc_7" ,"bound_0.8_0.2_50.txt" ,None, None],
        ["bound_mpc_8" ,"bound_0.8_0.4_50.txt" ,None, None],
        ["bound_mpc_9" ,"bound_0.8_0.6_50.txt" ,None, None],
        ["bound_mpc_10" ,"bound_0.8_0.8_50.txt" ,None, None],

        
        ######trot######
        ["trot_mpc_1" ,"trot_0.8_0.4_50.txt" ,None, None],
        ["trot_mpc_2" ,"trot_0.8_0.7_50.txt" ,None, None],
        ["trot_mpc_3" ,"trot_0.8_0.8_50.txt" ,None, None],
        ["trot_mpc_4" ,"trot_1.2_0.8_50.txt" ,None, None],
        ["trot_mpc_5" ,"trot_1.2_1.1_50.txt" ,None, None],
        ["trot_mpc_6" ,"trot_1.2_1.2_50.txt" ,None, None],
        ["trot_mpc_7" ,"trot_1.2_1_50.txt" ,None, None],
        ["trot_mpc_8" ,"trot_1_0.5_50.txt" ,None, None],
        ["trot_mpc_9" ,"trot_1_0.7_50.txt" ,None, None],
        ["trot_mpc_10" ,"trot_1_1_50.txt" ,None, None],

        ["crawl_mpc_1" ,"crawl_0.2_0.2_50.txt" ,None, None],
        ["crawl_mpc_2" ,"crawl_0.3_0.3_50.txt" ,None, None],
        ["crawl_mpc_3" ,"crawl_0.4_0.3_50.txt" ,None, None],
        ["crawl_mpc_4" ,"crawl_0.4_0.4_50.txt" ,None, None],
        ["crawl_mpc_5" ,"crawl_0.5_0.4_50.txt" ,None, None],
        ["crawl_mpc_6" ,"crawl_0.5_0.5_50.txt" ,None, None],
        ["crawl_mpc_7" ,"crawl_0.5_0.6_50.txt" ,None, None],
        ["crawl_mpc_8" ,"crawl_0.6_0.3_50.txt" ,None, None],
        ["crawl_mpc_9" ,"crawl_0.6_0.6_50.txt" ,None, None],
        ["crawl_mpc_10" ,"crawl_0.6_0.8_50.txt" ,None, None],

        ["gallop_mpc_1" ,"gallop_1.5_1.2_50.txt" ,None, None],
        ["gallop_mpc_2" ,"gallop_1.5_1.5_50.txt" ,None, None],
        ["gallop_mpc_3" ,"gallop_1.5_1.8_50.txt" ,None, None],
        ["gallop_mpc_4" ,"gallop_1.5_1_50.txt" ,None, None],
        ["gallop_mpc_5" ,"gallop_1.8_1.8_50.txt" ,None, None],
        ["gallop_mpc_6" ,"gallop_1_1_50.txt" ,None, None],
        ["gallop_mpc_7" ,"gallop_2.1_2.1_50.txt" ,None, None],
        ["gallop_mpc_8" ,"gallop_2_1.8_50.txt" ,None, None],
        ["gallop_mpc_9" ,"gallop_2_1.9_50.txt" ,None, None],
        ["gallop_mpc_10" ,"gallop_2_2.1_50.txt" ,None, None],
        # ["gallop_mpc_11" ,"gallop_2_2_50.txt" ,None, None],

        ["pace_mpc_1" ,"pace_0.6_0.3_50.txt" ,None, None],
        ["pace_mpc_2" ,"pace_0.6_0.6_50.txt" ,None, None],
        ["pace_mpc_3" ,"pace_0.6_1_50.txt" ,None, None],
        ["pace_mpc_4" ,"pace_0.75_0.4_50.txt" ,None, None],
        ["pace_mpc_5" ,"pace_0.75_1_50.txt" ,None, None],
        ["pace_mpc_6" ,"pace_0.7_0.4_50.txt" ,None, None],
        ["pace_mpc_7" ,"pace_0.7_0.8_50.txt" ,None, None],
        ["pace_mpc_8" ,"pace_0.8_0.5_50.txt" ,None, None],
        ["pace_mpc_9" ,"pace_0.8_0.8_50.txt" ,None, None],
        ["pace_mpc_10" ,"pace_0.8_1_50.txt" ,None, None],

   

    ]
)
############################################################################################################
# Solo8 dataset
############################################################################################################

solo8_dataset = DatasetConfig(
    name = "solo8",
    data_path= Path(__file__).parent.parent.parent.parent / "motion_retarget" /"datasets" / "solo8" / "data",
    frame_rate=50,

    scale_factor=1.1,
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
    ref2sim_toes_offset_local=[
            np.array([0.02, 0.03, 0.00]), #FL
            np.array([0.02, -0.03, -0.00]), # FR
            np.array([-0.1, 0.03, -0.00]), #RL
            np.array([-0.1, -0.03, -0.00]) # RR
    ],
    sim_root_offset = np.array([0, 0, 0.05]), #decide how high the root is from the ground

    #bool
    save_data=True,
    visualize=False,
    # visu_contact_phase=True,


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
        #   ['crawl_solo8_11', 'motion_data_1152.txt', 10, None],
        #   ['crawl_solo8_12', 'motion_data_1175.txt', 10, None],
        #   ['crawl_solo8_13', 'motion_data_1194.txt', 10, None],
        #   ['crawl_solo8_14', 'motion_data_1205.txt', 10, None],
        #   ['crawl_solo8_15', 'motion_data_1212.txt', 10, None],
        #   ['crawl_solo8_16', 'motion_data_1213.txt', 10, None],
        #   ['crawl_solo8_17', 'motion_data_1217.txt', 10, None],
        #   ['crawl_solo8_18', 'motion_data_1223.txt', 10, None],
        #   ['crawl_solo8_19', 'motion_data_1229.txt', 10, None],
        #   ['crawl_solo8_20', 'motion_data_1244.txt', 10, None],
        #   ['crawl_solo8_21', 'motion_data_1251.txt', 10, None],
        #   ['crawl_solo8_22', 'motion_data_1261.txt', 10, None],
        #   ['crawl_solo8_23', 'motion_data_1292.txt', 10, None],
        #   ['crawl_solo8_24', 'motion_data_1306.txt', 10, None],
        #   ['crawl_solo8_25', 'motion_data_1310.txt', 10, None],
        #   ['crawl_solo8_26', 'motion_data_1317.txt', 10, None],
        #   ['crawl_solo8_27', 'motion_data_1339.txt', 10, None],
        #   ['crawl_solo8_28', 'motion_data_1355.txt', 10, None],
        #   ['crawl_solo8_29', 'motion_data_1373.txt', 10, None],
        #   ['crawl_solo8_30', 'motion_data_1390.txt', 10, None],
        #   ['crawl_solo8_31', 'motion_data_1396.txt', 10, None],
        #   ['crawl_solo8_32', 'motion_data_1400.txt', 10, None],
        #   ['crawl_solo8_33', 'motion_data_1407.txt', 10, None],
        #   ['crawl_solo8_34', 'motion_data_1411.txt', 10, None],
        #   ['crawl_solo8_35', 'motion_data_0.txt', 10, None],

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
        #   ['stilt_solo8_11', 'motion_data_1121.txt', 10, None],
        #   ['stilt_solo8_12', 'motion_data_1122.txt', 10, None],
        #   ['stilt_solo8_13', 'motion_data_1123.txt', 10, None],
        #   ['stilt_solo8_14', 'motion_data_1124.txt', 10, None],
        #   ['stilt_solo8_15', 'motion_data_1125.txt', 10, None],
        #   ['stilt_solo8_16', 'motion_data_1128.txt', 10, None],
        #   ['stilt_solo8_17', 'motion_data_113.txt', 10, None],
        #   ['stilt_solo8_18', 'motion_data_1131.txt', 10, None],
        #   ['stilt_solo8_19', 'motion_data_1137.txt', 10, None],
        #   ['stilt_solo8_20', 'motion_data_1138.txt', 10, None],
        #   ['stilt_solo8_21', 'motion_data_1140.txt', 10, None],
        #   ['stilt_solo8_22', 'motion_data_1142.txt', 10, None],
        #   ['stilt_solo8_23', 'motion_data_1161.txt', 10, None],
        #   ['stilt_solo8_24', 'motion_data_1167.txt', 10, None],
        #   ['stilt_solo8_25', 'motion_data_1173.txt', 10, None],
        #   ['stilt_solo8_26', 'motion_data_1174.txt', 10, None],
        #   ['stilt_solo8_27', 'motion_data_118.txt', 10, None],
        #   ['stilt_solo8_28', 'motion_data_1186.txt', 10, None],
        #   ['stilt_solo8_29', 'motion_data_1191.txt', 10, None],
        #   ['stilt_solo8_30', 'motion_data_1195.txt', 10, None],
        #   ['stilt_solo8_31', 'motion_data_120.txt', 10, None],
        #   ['stilt_solo8_32', 'motion_data_122.txt', 10, None],
        #   ['stilt_solo8_33', 'motion_data_1226.txt', 10, None],
        #   ['stilt_solo8_34', 'motion_data_1228.txt', 10, None],
        #   ['stilt_solo8_35', 'motion_data_1000.txt', 10, None],
          
            #################LEAP######################
           
            ['leap_solo8_1', 'motion_data_1136.txt', 10, None],
            ['leap_solo8_2', 'motion_data_1144.txt', 10, None],
            ['leap_solo8_3', 'motion_data_1001.txt', 10, None],
            ['leap_solo8_4', 'motion_data_1089.txt', 10, None],
            ['leap_solo8_5', 'motion_data_1218.txt', 10, None],
            ['leap_solo8_6', 'motion_data_1272.txt', 10, None],
            ['leap_solo8_7', 'motion_data_1280.txt', 10, None],
            ['leap_solo8_8', 'motion_data_1311.txt', 10, None],
            ['leap_solo8_9', 'motion_data_1352.txt', 10, None],
            ['leap_solo8_10', 'motion_data_1482.txt', 10, None],
            # ['leap_solo8_11', 'motion_data_1444.txt', 10, None],
            # ['leap_solo8_12', 'motion_data_1496.txt', 10, None],
            # ['leap_solo8_13', 'motion_data_1519.txt', 10, None],
            # ['leap_solo8_14', 'motion_data_1529.txt', 10, None],
            # ['leap_solo8_15', 'motion_data_1533.txt', 10, None],
            # ['leap_solo8_16', 'motion_data_1548.txt', 10, None],
            # ['leap_solo8_17', 'motion_data_1559.txt', 10, None],
            # ['leap_solo8_18', 'motion_data_1601.txt', 10, None],
            # ['leap_solo8_19', 'motion_data_1655.txt', 10, None],
            # ['leap_solo8_20', 'motion_data_1721.txt', 10, None],
            # ['leap_solo8_21', 'motion_data_1824.txt', 10, None],
            # ['leap_solo8_22', 'motion_data_1858.txt', 10, None],
            # ['leap_solo8_23', 'motion_data_1907.txt', 10, None],
            # ['leap_solo8_24', 'motion_data_2042.txt', 10, None],
            # ['leap_solo8_25', 'motion_data_2044.txt', 10, None],
            # ['leap_solo8_26', 'motion_data_2084.txt', 10, None],
            # ['leap_solo8_27', 'motion_data_2085.txt', 10, None],
            # ['leap_solo8_28', 'motion_data_2094.txt', 10, None],
            # ['leap_solo8_29', 'motion_data_2192.txt', 10, None],
            # ['leap_solo8_30', 'motion_data_2259.txt', 10, None],
            # ['leap_solo8_31', 'motion_data_2272.txt', 10, None],
            # ['leap_solo8_32', 'motion_data_23.txt', 10, None],
            # ['leap_solo8_33', 'motion_data_2331.txt', 10, None],
            # ['leap_solo8_34', 'motion_data_2432.txt', 10, None],
            # ['leap_solo8_35', 'motion_data_1129.txt', 10, None],

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
            # ['trot_solo8_11', 'motion_data_1042.txt', 10, None],
            # ['trot_solo8_12', 'motion_data_1046.txt', 10, None],
            # ['trot_solo8_13', 'motion_data_1052.txt', 10, None],
            # ['trot_solo8_14', 'motion_data_107.txt', 10, None],
            # ['trot_solo8_15', 'motion_data_1075.txt', 10, None],
            # ['trot_solo8_16', 'motion_data_1077.txt', 10, None],
            # ['trot_solo8_17', 'motion_data_1096.txt', 10, None],
            # ['trot_solo8_18', 'motion_data_1115.txt', 10, None],
            # ['trot_solo8_19', 'motion_data_1118.txt', 10, None],
            # ['trot_solo8_20', 'motion_data_1132.txt', 10, None],
            # ['trot_solo8_21', 'motion_data_1139.txt', 10, None],
            # ['trot_solo8_22', 'motion_data_116.txt', 10, None],
            # ['trot_solo8_23', 'motion_data_1163.txt', 10, None],
            # ['trot_solo8_24', 'motion_data_1165.txt', 10, None],
            # ['trot_solo8_25', 'motion_data_1172.txt', 10, None],
            # ['trot_solo8_26', 'motion_data_1176.txt', 10, None],
            # ['trot_solo8_27', 'motion_data_1178.txt', 10, None],
            # ['trot_solo8_28', 'motion_data_1184.txt', 10, None],
            # ['trot_solo8_29', 'motion_data_1187.txt', 10, None],
            # ['trot_solo8_30', 'motion_data_119.txt', 10, None],
            # ['trot_solo8_31', 'motion_data_1190.txt', 10, None],
            # ['trot_solo8_32', 'motion_data_1192.txt', 10, None],
            # ['trot_solo8_33', 'motion_data_1196.txt', 10, None],
            # ['trot_solo8_34', 'motion_data_1203.txt', 10, None],
            # ['trot_solo8_35', 'motion_data_1732.txt', 10, None],
          
          
          
            #####################Walk################################
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
            # ['walk_solo8_11', 'motion_data_1085.txt', 10, None],
            # ['walk_solo8_12', 'motion_data_1099.txt', 10, None],
            # ['walk_solo8_13', 'motion_data_111.txt', 10, None],
            # ['walk_solo8_14', 'motion_data_1111.txt', 10, None],
            # ['walk_solo8_15', 'motion_data_1114.txt', 10, None],
            # ['walk_solo8_16', 'motion_data_1127.txt', 10, None],
            # ['walk_solo8_17', 'motion_data_1185.txt', 10, None],
            # ['walk_solo8_18', 'motion_data_1199.txt', 10, None],
            # ['walk_solo8_19', 'motion_data_1200.txt', 10, None],
            # ['walk_solo8_20', 'motion_data_1201.txt', 10, None],
            # ['walk_solo8_21', 'motion_data_1231.txt', 10, None],
            # ['walk_solo8_22', 'motion_data_1266.txt', 10, None],
            # ['walk_solo8_23', 'motion_data_128.txt', 10, None],
            # ['walk_solo8_24', 'motion_data_1281.txt', 10, None],
            # ['walk_solo8_25', 'motion_data_1295.txt', 10, None],
            # ['walk_solo8_26', 'motion_data_1305.txt', 10, None],
            # ['walk_solo8_27', 'motion_data_1307.txt', 10, None],
            # ['walk_solo8_28', 'motion_data_1315.txt', 10, None],
            # ['walk_solo8_29', 'motion_data_1347.txt', 10, None],
            # ['walk_solo8_30', 'motion_data_1351.txt', 10, None],
            # ['walk_solo8_31', 'motion_data_1365.txt', 10, None],
            # ['walk_solo8_32', 'motion_data_1366.txt', 10, None],
            # ['walk_solo8_33', 'motion_data_1387.txt', 10, None],
            # ['walk_solo8_34', 'motion_data_1388.txt', 10, None],
            # ['walk_solo8_35', 'motion_data_1025.txt', 10, None],
          
          
            ######################WAVE###############################
            
            ['wave_solo8_1', 'motion_data_1007.txt', 10, None],
            ['wave_solo8_2', 'motion_data_1023.txt', 10, None],
            ['wave_solo8_3', 'motion_data_1031.txt', 10, None],
            ['wave_solo8_4', 'motion_data_1035.txt', 10, None],
            ['wave_solo8_5', 'motion_data_1041.txt', 10, None],
            ['wave_solo8_6', 'motion_data_1053.txt', 10, None],
            ['wave_solo8_7', 'motion_data_1063.txt', 10, None],
            ['wave_solo8_8', 'motion_data_1064.txt', 10, None],
            ['wave_solo8_9', 'motion_data_1065.txt', 10, None],
            ['wave_solo8_10', 'motion_data_1086.txt', 10, None],
            # ['wave_solo8_11', 'motion_data_1087.txt', 10, None],
            # ['wave_solo8_12', 'motion_data_1095.txt', 10, None],
            # ['wave_solo8_13', 'motion_data_1104.txt', 10, None],
            # ['wave_solo8_14', 'motion_data_1116.txt', 10, None],
            # ['wave_solo8_15', 'motion_data_1130.txt', 10, None],
            # ['wave_solo8_16', 'motion_data_115.txt', 10, None],
            # ['wave_solo8_17', 'motion_data_1160.txt', 10, None],
            # ['wave_solo8_18', 'motion_data_1162.txt', 10, None],
            # ['wave_solo8_19', 'motion_data_1164.txt', 10, None],
            # ['wave_solo8_20', 'motion_data_1180.txt', 10, None],
            # ['wave_solo8_21', 'motion_data_1182.txt', 10, None],
            # ['wave_solo8_22', 'motion_data_1197.txt', 10, None],
            # ['wave_solo8_23', 'motion_data_1209.txt', 10, None],
            # ['wave_solo8_24', 'motion_data_121.txt', 10, None],
            # ['wave_solo8_25', 'motion_data_1211.txt', 10, None],
            # ['wave_solo8_26', 'motion_data_1215.txt', 10, None],
            # ['wave_solo8_27', 'motion_data_1219.txt', 10, None],
            # ['wave_solo8_28', 'motion_data_1227.txt', 10, None],
            # ['wave_solo8_29', 'motion_data_1232.txt', 10, None],
            # ['wave_solo8_30', 'motion_data_1235.txt', 10, None],
            # ['wave_solo8_31', 'motion_data_1239.txt', 10, None],
            # ['wave_solo8_32', 'motion_data_1241.txt', 10, None],
            # ['wave_solo8_33', 'motion_data_1249.txt', 10, None],
            # ['wave_solo8_34', 'motion_data_1252.txt', 10, None],
            # ['wave_solo8_35', 'motion_data_1.txt', 10, None],


    ]
)
