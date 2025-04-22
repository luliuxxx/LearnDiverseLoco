import pinocchio as pin
import numpy as np
import json
from transforms3d.quaternions import rotate_vector, qmult, qinverse, axangle2quat, mat2quat, quat2mat
from transforms3d.euler import euler2quat, quat2euler, mat2euler
import mujoco
import mujoco.viewer
from dm_control import mjcf
import time
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from utils.ik_solver import IKSolver
from utils.viewer import MujocoViewer
from tqdm import tqdm


#  define constants
POS_SIZE = 3
QUAT_SIZE = 4
NUMS_TOES = 4
color_map = ['#d7191c', '#fdae61' , '#abdda4', '#2b83ba']


class MotionRetarget():

    def __init__(self, robot_config, motion_config):
        # make sure the robot is in free flyer mode, floating base
        self.robot_config = robot_config
        self.motion_config = motion_config
        self.robot = pin.buildModelFromUrdf(robot_config.urdf_path, pin.JointModelFreeFlyer())
        self.raw_ref_trajs = [self.load_one_ref_data(motion[0], str(motion_config.data_path) + '/' + motion[1], motion[2], motion[3]) for motion in motion_config.motions]
        self.stars = [motion[2] for motion in motion_config.motions]
        self.ends = [motion[3] for motion in motion_config.motions]
        self.d = self.robot.createData()
        self.q = robot_config.default_pose
        
        self.ik_solver = IKSolver(self.robot, self.d, robot_config.default_pose)
        # variables
        self.retargeted_traj = []
        self.scale_factor = None
        self.mujoco_repeat_times = 1 # repeat the data for visualization

        self.join2id = dict({
            'base_pos':[0,1,2], #xyz
            'base_rot':[3,4,5,6], #xyzw
            'base':[0,1,2,3,4,5,6], #xyz-xyzw
            'FL_joint':[7,8,9], # hip, thigh, calf
            'FR_joint':[10,11,12], # hip, thigh, calf
            'RL_joint':[13,14,15], # hip, thigh, calf
            'RR_joint':[16,17,18], # hip, thigh, calf
        })
        self.start = None
        self.end = None
        xml_handle = self.add_marker_to_xml_handle()
        self.mujoco_model = mujoco.MjModel.from_xml_string(xml_handle.to_xml_string(),xml_handle.get_assets())
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        # here dt should be the control frequency
        if self.motion_config.visualize:
            self.mujoco_viewer = MujocoViewer(self.mujoco_model, dt = 1/self.motion_config.frame_rate)
            self.mujoco_viewer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
    
    def load_one_ref_data(self, motion_name, data, start=None, end=None):
        ref_pos_data = np.loadtxt(data,delimiter=',')
        start = 0 if start is None else start
        end = ref_pos_data.shape[0] if end is None else end
        ref_pos_data = ref_pos_data[start:end]
        ref_pos_data = {motion_name: ref_pos_data}
        return ref_pos_data
    
    def pin_get_configuration(self):
        q_pos = []
        for joint_idx in range(1,len(self.d.joints)):
            q = self.d.joints[joint_idx].joint_q
            q_pos.append(q)
        self.q = np.concatenate(q_pos)

    def pin_set_configuration(self, q_pos):
        self.q = q_pos.flatten().copy()
        pin.forwardKinematics(self.robot, self.d, self.q)
        pin.updateFramePlacements(self.robot, self.d)

    def set_joint_qpos(self, joint_name, pos):
        # set the joint position
        idx = self.joint2id(joint_name)
        assert idx is None, "joint name is not correct"
        assert len(idx) == len(pos), "joint position is not correct"
        self.q[idx] = pos.flatten().copy()
        self.pin_set_configuration(self.q)
    
    def get_joint_qpos(self, joint_name):
        # get current joint position
        q = self.pin_get_configuration()
        self.q = q.copy()
        return self.q[self.joint2id(joint_name)]

    def get_hip_wpos(self, leg_name):
        # get the hip world position
        frame_id = self.robot.getFrameId(f"{leg_name}_hip")
        return self.d.oMf[frame_id].translation
    
    def get_toes_wpos(self):
        # get the toes world position
        toes_pos = []
        for leg in self.motion_config.legs_name:
            frame_id = self.robot.getFrameId(f"{leg}_foot")
            toes_pos.append(self.d.oMf[frame_id].translation)
        return np.array(toes_pos)
    
    def set_base_pos(self, pos):
        # set the base position
        self.pin_get_configuration()
        self.q[:3] = pos.copy()
        self.pin_set_configuration(self.q)
    
    def get_base_pos(self):
        self.pin_get_configuration()
        return self.q[:3]

    def set_base_rot(self, quat):
        # set the base rotation
        # the quat should be in order of xyzw
        # during the calculation we use wxyz
        quat = np.roll(quat,-1) # wxyz -> xyzw
        self.pin_get_configuration()
        self.q[3:7] = quat.copy() # xyzw
        self.pin_set_configuration(self.q)
    
    def get_base_rot(self):
        # get the base rotation
        self.pin_get_configuration()
        quat = self.q[3:7]
        quat = np.roll(quat,1)
        return quat

    def retarget_root_pos_rot(self, pos):
        # retarget the root position
        # import pdb; pdb.set_trace()
        if self.motion_config.ref_pelvis_id is not None or self.motion_config.ref_neck_id is not None:
            pelvis_pos = pos[self.motion_config.ref_pelvis_id]
            neck_pos = pos[self.motion_config.ref_neck_id]
            root_pos = (pelvis_pos + neck_pos) / 2
        else:
            pelvis_pos = pos[self.motion_config.ref_hips_id_list[2]]
            neck_pos = pos[self.motion_config.ref_hips_id_list[0]]
            root_pos = 0.25 * (pos[self.motion_config.ref_hips_id_list[0]] 
                               + pos[self.motion_config.ref_hips_id_list[1]]
                               + pos[self.motion_config.ref_hips_id_list[2]] 
                               + pos[self.motion_config.ref_hips_id_list[3]])
            
        forward_dir = neck_pos - pelvis_pos
        forward_dir += self.motion_config.forward_dir_offset
        forward_dir /= np.linalg.norm(forward_dir)

        # no shoulder joint
        # take the neck and pelvis joint to calculate the root direction
        if self.motion_config.ref_hips_id_list[0] == self.motion_config.ref_hips_id_list[1] and self.motion_config.ref_hips_id_list[2] == self.motion_config.ref_hips_id_list[3]:
            arb_dir = np.array([0, 0, 1])
            left_dir = np.cross(forward_dir, arb_dir)
            left_dir = left_dir / np.linalg.norm(left_dir)
            left_dir = -left_dir
            up_dir = np.cross(forward_dir, left_dir)
            up_dir /= np.linalg.norm(up_dir)

        else:
            # take the shoulder joint to calculate the root direction
            left_shoulder_pos = pos[self.motion_config.ref_hips_id_list[0]]
            right_shoulder_pos = pos[self.motion_config.ref_hips_id_list[1]]

            left_hip_pos = pos[self.motion_config.ref_hips_id_list[2]]
            right_hip_pos = pos[self.motion_config.ref_hips_id_list[3]]

            delta_shoulder = left_shoulder_pos - right_shoulder_pos
            delta_hip = left_hip_pos - right_hip_pos
            dir_shoulder = delta_shoulder / np.linalg.norm(delta_shoulder)
            dir_hip = delta_hip / np.linalg.norm(delta_hip)

            left_dir = 0.5 * (dir_shoulder + dir_hip)
            up_dir = np.cross(forward_dir, left_dir)
            up_dir /= np.linalg.norm(up_dir)

            left_dir = np.cross(up_dir, forward_dir)
            # make the base more stable
            left_dir[2] = 0.0
            left_dir /= np.linalg.norm(left_dir)
        
        # rot_mat is a 3*3 matrix
        root_mat = np.array([[forward_dir[0], left_dir[0], up_dir[0]],
                            [forward_dir[1], left_dir[1], up_dir[1]],
                            [forward_dir[2], left_dir[2], up_dir[2]]])

        # calculate the root orientation
        root_mat /= np.linalg.norm(root_mat)
        root_quat = mat2quat(root_mat) #wxyz during calculation we use wxyz # only when we output the quat we use xyzw
        root_pos += self.motion_config.ref_pos_offset
        return root_pos, root_quat
    

    def compute_contact_phase(self, data):
        # import pdb; pdb.set_trace()
        contact_phase = np.zeros((data.shape[0],NUMS_TOES))
        for i in range(data.shape[0]):
            for j,k in enumerate(self.motion_config.ref_toes_id_list):
                if data[i, 3*k + 2] > self.motion_config.contact_threshold:
                    contact_phase[i, j] = 0
                else:
                    contact_phase[i, j] = 1
        return contact_phase

    def skew(self,x, skew_factor):
        # only apply on z
        if skew_factor == None:
            skew_factor = self.scale_factor
        skewed_x = skew_factor* (np.maximum(x,0))
        return skewed_x

    def cal_heading_rot(self, root_quat):
        # calculate the heading rotation
        ref_dir = np.array([1, 0, 0])
        rot_dir = rotate_vector(ref_dir, root_quat)
        heading = np.arctan2(rot_dir[1], rot_dir[0])
        h_quat = axangle2quat([0, 0, 1], heading)
        return h_quat
    
    def adjust_height(self, og_height):
        new_height = og_height.copy()
        idx = np.where(new_height <= self.motion_config.contact_threshold)
        if idx[0].shape[0] == 0:
            new_height = self.skew(new_height, self.motion_config.skew_factor)
        else:
            # keep the part under the threshold the same 
            new_height -= self.motion_config.contact_threshold
            new_height[idx] = 0
            new_height = self.skew(new_height, self.motion_config.skew_factor)
            new_height += self.motion_config.contact_threshold
            new_height[idx] = og_height[idx] # keep 
            # new_height[idx] = 0.0
        return new_height

    def convert_to_phases(self, binary_list):
        """
        Converts a binary list of ground contact indications to a list of (start_time, duration) tuples.
        
        Parameters:
        - binary_list: List of 0s and 1s indicating ground contact.
        - time_step: The time interval between each binary value. Default is 1.
        
        Returns:
        - List of tuples where each tuple represents (start_time, duration) of contact.
        """
        phases = []
        start_time = None
        time_step= 1 / self.motion_config.frame_rate
        for i, value in enumerate(binary_list):
            if value == 1 and start_time is None:
                # Start of a new contact phase
                start_time = i * time_step
            elif value == 0 and start_time is not None:
                # End of the current contact phase
                end_time = i * time_step
                duration = end_time - start_time
                phases.append((start_time, duration))
                start_time = None

        # If the list ends with an ongoing contact phase
        if start_time is not None:
            end_time = len(binary_list) * time_step
            duration = end_time - start_time
            phases.append((start_time, duration))

        return phases

    def display_contact_phase(self, og_contact_phase, new_contact_phase, og_z, new_z, ik_z, motion_name):
        # display the contact phase
        #convert the contact phase to a pair of (start_time, duration)
        og_phases = [self.convert_to_phases(og_contact_phase[:,i]) for i in range(og_contact_phase.shape[1])]
        new_phases = [self.convert_to_phases(new_contact_phase[:,i]) for i in range(new_contact_phase.shape[1])]
        fig, ax = plt.subplots(2,1)
        ax[0].broken_barh(og_phases[0], (0, 0.5), facecolors= color_map[0], label='FL', alpha=0.5)
        ax[0].broken_barh(og_phases[1], (1, 0.5), facecolors= color_map[1], label='FR', alpha=0.5)
        ax[0].broken_barh(og_phases[2], (2, 0.5), facecolors= color_map[2], label='RL', alpha=0.5)
        ax[0].broken_barh(og_phases[3], (3, 0.5), facecolors= color_map[3], label='RR', alpha=0.5)

        ax[0].broken_barh(new_phases[0], (0.25, 0.5), facecolors= color_map[0], label='FL')
        ax[0].broken_barh(new_phases[1], (1.25, 0.5), facecolors= color_map[1], label='FR')
        ax[0].broken_barh(new_phases[2], (2.25, 0.5), facecolors= color_map[2], label='RL')
        ax[0].broken_barh(new_phases[3], (3.25, 0.5), facecolors= color_map[3], label='RR')
        
        ax[0].set_title('Contact Phase')
        ax[0].set_yticks([0.25, 1.25, 2.25, 3.25])
        ax[0].set_yticklabels(['FL', 'FR', 'RL', 'RR'])
        ax[1].plot(og_z[:,0],  color=color_map[0],linestyle='--')
        ax[1].plot(og_z[:,1] + 0.1, color=color_map[1], linestyle='--')
        ax[1].plot(og_z[:,2] + 0.2, color=color_map[2], linestyle='--')
        ax[1].plot(og_z[:,3] + 0.3, color=color_map[3], linestyle='--')

        ax[1].plot(new_z[:,0],  color=color_map[0],linestyle='-.')
        ax[1].plot(new_z[:,1] +  0.1, color=color_map[1],linestyle='-.')
        ax[1].plot(new_z[:,2] +  0.2, color=color_map[2],linestyle='-.')
        ax[1].plot(new_z[:,3] +  0.3, color=color_map[3],linestyle='-.')

        ax[1].plot(ik_z[:,0],  color=color_map[0])
        ax[1].plot(ik_z[:,1] +  0.1, color=color_map[1])
        ax[1].plot(ik_z[:,2] +  0.2, color=color_map[2])
        ax[1].plot(ik_z[:,3] +  0.3, color=color_map[3])
        ax[1].yaxis.set_visible(False)
        ax[1].set_title('Toes Height')

        # check if the directory exists
        directory = self.motion_config.save_dir / 'contact_phase'
        if not directory.exists():
            directory.mkdir(parents=True)
        fig.savefig( str(directory) + '/' + motion_name +'_contact_phase.png')
        plt.close()
 

    def retarget_pose(self, traj_list):
        # traj list is in shape [T,[num_markers, 3]]
        root_pos, root_quat = self.retarget_root_pos_rot(traj_list)
        # import pdb; pdb.set_trace()
        root_pos += self.motion_config.sim_root_offset
        # set the root pos and quat
        self.set_base_rot(root_quat)
        self.set_base_pos(root_pos)
        heading_quat = self.cal_heading_rot(root_quat) # wxyz
        sim_toes_pos = []
        # get the ref hip pos
        for id in range(NUMS_TOES):
            toe_id = self.motion_config.ref_toes_id_list[id]
            hip_id = self.motion_config.ref_hips_id_list[id]
            ref_toe_pos = traj_list[toe_id].reshape(1,3)
            ref_hip_pos = traj_list[hip_id].reshape(1,3)
            toe_offset_local = self.motion_config.ref2sim_toes_offset_local[id]
            toe_offset_world = rotate_vector(toe_offset_local, heading_quat)

            sim_hip_pos = self.get_hip_wpos(self.motion_config.legs_name[id])
            delta_ref_toe_pos = ref_toe_pos - ref_hip_pos
            sim_toe_pos = sim_hip_pos + delta_ref_toe_pos
            sim_toe_pos += toe_offset_world
            sim_toe_pos[:,2] = ref_toe_pos[:,2] 
            sim_toes_pos.append(sim_toe_pos)
        return root_pos, root_quat, sim_toes_pos

    def retarget_one_traj(self, traj):
        # input the one single traj
        # output the retargeted traj
        # difference between traj_list and traj_hist, list contains all the joints, hist contains the joints in all the frames
        # handle the raw dataset
        print("Retargeting the pose")
        num_frames = traj.shape[0]
        ikq_hist = []
        toes_hist = []
        valid_hist = []
        limit_error = []
        for fr_idx in tqdm(range(num_frames)):
            traj_list = traj[fr_idx]
            traj_list = traj_list.reshape(self.motion_config.num_markers, POS_SIZE)
            root_pos, root_quat, toes_pos = self.retarget_pose(traj_list)
            if fr_idx == 0:
                pose_size = POS_SIZE + QUAT_SIZE +  NUMS_TOES * POS_SIZE
                new_frames = np.zeros((num_frames, pose_size))
                # get init heading quat
                heading_quat = self.cal_heading_rot(root_quat)
                init_quat_offset = qinverse(heading_quat)
                init_root_pos = root_pos.copy()
                init_root_pos[2] = 0.0
            if self.motion_config.heading_calibration:
                root_pos = rotate_vector(root_pos, init_quat_offset)
                root_quat = qmult(init_quat_offset, root_quat)
                for toe in range(NUMS_TOES):
                    toes_pos[toe] = rotate_vector(toes_pos[toe], init_quat_offset)
            root_pos -= init_root_pos
            toes_pos = [toe - init_root_pos for toe in toes_pos]
            new_frames[fr_idx] = np.concatenate((root_pos, root_quat, toes_pos[0], toes_pos[1], toes_pos[2], toes_pos[3]), axis=None)
 
            ikq, tmp_toes, limit_err = self.apply_ik(new_frames[fr_idx])
            if limit_err>0.0:
                valid_hist.append(-1)
            else:
                assert limit_err == 0.0, "Joint limit error is not zero"
                valid_hist.append(0)
            ikq_hist.append(ikq)
            toes_hist.append(tmp_toes)
            limit_error.append(limit_err)
        print("Retargeting done!")

        assert len(ikq_hist) == len(toes_hist), "The length of ikq_hist and toes_hist is not the same"
        assert len(ikq_hist) == len(valid_hist), "The length of ikq_hist and valid_hist is not the same"
        return np.asarray(ikq_hist), np.asarray(toes_hist), np.mean(np.asarray(limit_error)) # qpos: 19

    def apply_ik(self, frames):
        # apply the inverse kinematics
        return self.ik_solver.compute_ik(frames)

    def process_raw_traj(self, traj, traj_name):
        # process the raw data
        # apply the rotation and translation offset
        # apply the scale factor
        if self.motion_config.scale_factor is not None:
            self.scale_factor = self.motion_config.scale_factor 
        else:
            self.scale_factor = self.robot_config.stance_height / np.median(traj[:,self.motion_config.ref_height_id])
            self.traj_height = np.median(traj[:,self.motion_config.ref_height_id])

        og_traj = traj.copy()
        
        for idx,pose in enumerate(traj):
            pose = pose.reshape(self.motion_config.num_markers,POS_SIZE)
            pose_og = pose.copy()
            for joint in range(self.motion_config.num_markers):
                # apply rotation offset
                pose[joint] = rotate_vector(pose[joint],self.motion_config.ref_coord_rot,False) 
                pose[joint] = rotate_vector(pose[joint],self.motion_config.ref_root_rot,False)
                pose_og[joint] = pose[joint].copy()
                # apply translation offset
                pose[joint] *= self.scale_factor

                # adjust height:
                pose[joint] += self.motion_config.ref_pos_offset
                pose_og[joint] += self.motion_config.ref_pos_offset
                # update self.ref_data
            traj[idx] = pose.flatten()
            og_traj[idx] = pose_og.flatten()
        for maker in self.motion_config.ref_toes_id_list:
            traj[:, maker*3 + 2] = self.adjust_height(og_traj[:, maker*3 + 2])
        return og_traj, traj
    
    def load_json_data(self, file_name):
        with open( str(self.motion_config.save_dir) + '/' +  file_name +'.txt') as f:
            data = json.load(f)
        return np.asarray(data['Frames'])
    
        
    
    def forward(self, retarget=False):
        # forward the data
        # get the reference data
        for id, raw_traj_dict in enumerate(self.raw_ref_trajs):

            traj_name, raw_traj = raw_traj_dict.popitem()
            og_traj, cooked_traj = self.process_raw_traj(raw_traj, traj_name)

            print(f"Processing the {traj_name} with scale factor {self.scale_factor}")
    
            if retarget:
                print(f"Processing the {traj_name} ")
                cooked_qpos, toes_pos, error = self.retarget_one_traj(cooked_traj)
    
                if self.motion_config.visu_contact_phase:
                    contacted_phase = self.compute_contact_phase(og_traj)
                    scaled_phase = self.compute_contact_phase(cooked_traj)
                    z_idx_list = np.array(self.motion_config.ref_toes_id_list)*3 + 2
                    # import pdb; pdb.set_trace()
                    self.display_contact_phase(contacted_phase, scaled_phase, og_traj[:,z_idx_list], cooked_traj[:,z_idx_list],toes_pos[:,2::3],traj_name)
            
                self.retargeted_traj.append(cooked_qpos)
                if self.motion_config.save_data:
                    self.save_ik_joint_data(cooked_qpos, traj_name)
                if self.motion_config.visualize:
                    print(f"Visualizing the {traj_name} ")
                    
                    self.visualize(cooked_qpos,cooked_traj)
            else:
                #load the retargeted data
                print(f"Loading the {traj_name} ")
                ret_qpos = self.load_json_data(traj_name)
                if ret_qpos.shape[0] != og_traj.shape[0]:
                    start = self.stars[id]
                    end = self.ends[id]
                    chopped_qpos = ret_qpos[start:end]
                else:
                    chopped_qpos = ret_qpos
                self.visualize(chopped_qpos, cooked_traj)
                   
    def save_ik_joint_data(self, frames,motion_name,valid=None):
        # output the json file
        
        if self.motion_config.save_dir is None:
            save_dir = './'
        else:
            save_dir = str(self.motion_config.save_dir) + '/'
        
        file_name = save_dir + motion_name
        print(f"Saving the ik joint data to {file_name}.txt")
        motion_type = motion_name.split('_')[0]
        motion_source = motion_name.split('_')[1]
        
        with open(file_name+'.txt', 'w') as f:
            f.write("{\n")
            f.write(f"\"motion_type\": \"{motion_type}\",\n")
            f.write(f"\"motion_source\": \"{motion_source}\",\n")
            f.write("\"Order\": \" base: xyz,wxyz, FL,FR,RL,RR\",\n")
            f.write("\"Dimensions\": "+ str(frames.shape[1]) + ",\n")
            f.write("\"NumFrames\": "+ str(frames.shape[0]) + ",\n")
            f.write("\"FrameRate\": "+ str(self.motion_config.frame_rate) + ",\n")
            f.write(f"\"Space\": \"Joint Space\",\n")
            f.write("\"Frames\": \n")
            f.write("[")
            for i in range(frames.shape[0]):
                curr_frame = frames[i]
                if i != 0:
                    f.write(",")
                f.write("\n  [")
                for j in range(frames.shape[1]):
                    curr_val = curr_frame[j]
                    if j != 0:
                        f.write(", ")
                    f.write("%.5f" % curr_val)
                f.write("]")
            f.write("\n]")
            f.write("\n}")

    
    def add_marker_to_xml_handle(self):
        xml_handle = mjcf.from_path(self.robot_config.xml_path)
        trunk = xml_handle.find("body", "trunk")
        if trunk is None:
            xml_handle.worldbody.add("body", name="trunk", pos="0 0 0.5")
            trunk = xml_handle.find("body", "trunk")
        trunk.add("body", name="dir_arrow", pos="0 0 0.15")
        dir_vec = xml_handle.find("body", "dir_arrow")
        dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".03", pos="-.1 0 0")
        dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="-.1 0 0 .1 0 0")
        xml_handle.worldbody.add("body", name="fixed_marker", pos="0 0 0",mocap="true")
        fixed_marker = xml_handle.find("body", "fixed_marker")
        str_pos = "0 0 0.1"
        for marker in range(self.motion_config.num_markers):
            # add marker to the xml
            if (marker == self.motion_config.ref_neck_id) or (marker == self.motion_config.ref_pelvis_id):
                # green
                col = "0 1 0 1"
            elif (marker in self.motion_config.ref_hips_id_list[:2]) or (marker in self.motion_config.ref_toes_id_list[:2]):
                col = "0 0 1 1" # blue # FL FR
            elif (marker in self.motion_config.ref_hips_id_list[2:]) or (marker in self.motion_config.ref_toes_id_list[2:]) :
                col = "1 0 0 1" # red # RL RR
            else:
                #gray
                col = "0.5 0.5 0.5 0.5"
            fixed_marker.add("site", name=f"ref_{marker}", pos=str_pos, size=".03", rgba=col)
        fixed_marker.add("site", name="ref_base", type = 'box', pos="0 0 0.5", size=".3", rgba="1 0 0 1")
        return xml_handle

    def visualize(self, qpos, og_traj, toes_pos=None,base_pos=None, base_quat=None):
        # visualize the data
        # mujoco ordering: FR, FL, RR, RL
        # go2 ordering: FL, FR, RL, RR in urdf and xml
        # go2 mask: FR thigh -1 FR calf -1 RL hip -1 RR hip, thigh, calf -1

        pin_fl = [7, 8, 9]
        pin_fr = [10, 11, 12]
        pin_rl = [13, 14, 15]
        pin_rr = [16, 17, 18]
        if self.robot_config.name == 'unitree_go2':
            fl_mask = np.array([1, 1, 1])
            fr_mask = np.array([1, -1, -1])
            rl_mask = np.array([-1, 1, 1])
            rr_mask = np.array([-1, -1, -1])
        elif self.robot_config.name == 'unitree_a1':
            fl_mask = np.array([1,1,1])
            fr_mask = np.array([1,1,1])
            rl_mask = np.array([1,1,1])
            rr_mask = np.array([1,1,1])
        else:
            raise ValueError("Robot is not correct")
     
        for repeat in range(self.mujoco_repeat_times):
            for i in range(len(qpos)):
                if i ==0:
                    traj = og_traj[i].reshape(self.motion_config.num_markers, POS_SIZE)
                    og_traj_root_offset = np.sum(traj[self.motion_config.ref_hips_id_list], axis=0) / 4
                    og_traj_root_offset[2] = 0.0
                    q_pos_offset = qpos[0][:3]
                    q_pos_offset[2] = 0.0
       
                self.mujoco_data.qpos[3:7] = qpos[i][3:7]
                self.mujoco_data.qpos[pin_fr] = qpos[i][pin_fr]*fr_mask
                self.mujoco_data.qpos[pin_fl] = qpos[i][pin_fl]*fl_mask
                self.mujoco_data.qpos[pin_rr] = qpos[i][pin_rr]*rr_mask
                self.mujoco_data.qpos[pin_rl] = qpos[i][pin_rl]*rl_mask
                self.mujoco_data.qpos[:3] = qpos[i][:3]  - q_pos_offset
                
                mujoco.mj_fwdPosition(self.mujoco_model, self.mujoco_data)
                for j in range(self.motion_config.num_markers):
                    
                    self.mujoco_data.site(f'ref_{j}').xpos = og_traj[i][j*3:j*3+3] - og_traj_root_offset
                if toes_pos is not None:
                    for k in range(4):
                        
                        self.mujoco_data.site(f'ref_{k+self.motion_config.num_markers}').xpos = toes_pos[i][k*3:k*3+3] - og_traj_root_offset
                self.mujoco_data.site('ref_base').xpos = qpos[i][:3] - q_pos_offset
                mat = quat2mat(qpos[i][3:7])
                self.mujoco_data.site('ref_base').xmat = mat.flatten()
                self.mujoco_viewer.render(self.mujoco_data)

    

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--retarget',type=int,default=1,choices=[0,1],help="Retarget the data")
    parser.add_argument('--motion_source', type=str, default = 'horse', choices=['horse','solo8','mpc','dog','solo8_trot','solo8_crawl','solo8_wave'], help="Motion source")
    parser.add_argument('--robot', type=str, default = 'a1', choices=['a1','go2'], help="Robot type")

    # 1 means retarget the data
    # 0 means only visualize the retargeted data, please make sure the data is already retargeted!
    args = parser.parse_args()


    
    if args.robot == 'a1':
        from robot_config import unitree_a1 as robot_config
        from motion_retarget.robots.a1.dataset_config import horse_dataset, dog_dataset, mpc_dataset, solo8_dataset
    elif args.robot == 'go2':
        from robot_config import unitree_go2 as robot_config
        from motion_retarget.robots.go2.dataset_config import horse_dataset, dog_dataset, mpc_dataset, solo8_dataset, solo8_dataset_trot, solo8_dataset_wave, solo8_dataset_crawl
    else:
        raise ValueError("Robot is not correct")
    
        
    
    if args.motion_source == 'horse':
        motion_config = horse_dataset
    elif args.motion_source == 'dog':
        motion_config = dog_dataset
    elif args.motion_source == 'mpc':
        motion_config = mpc_dataset
    elif args.motion_source == 'solo8':
        motion_config = solo8_dataset
    elif args.motion_source == 'solo8_trot': # only for go2
        motion_config = solo8_dataset_trot
    elif args.motion_source == 'solo8_wave': # only for go2
        motion_config = solo8_dataset_wave
    elif args.motion_source == 'solo8_crawl': # only for go2
        motion_config = solo8_dataset_crawl
    else:
        raise ValueError("Motion source is not correct")
    retarget_obj = MotionRetarget(robot_config, motion_config)
    

    if args.retarget:
        print("Start retargeting the data")
        retarget_obj.forward(retarget=True)
    else:
        print("only visualize the retargeted data, please make sure the data is already retargeted!")
        retarget_obj.forward(retarget=False)


