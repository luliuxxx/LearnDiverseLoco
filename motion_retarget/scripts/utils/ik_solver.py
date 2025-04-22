import numpy as np
import pinocchio as pin
##
from transforms3d.euler import euler2quat,quat2euler, quat2mat
from numpy.linalg import solve
class IKSolver:
    def __init__(self, pin_model, pin_data, default_pose, eps=1e-5, it_max=2000, dt=0.1, damp=3e-2, limit_damp=1):
        """
        Initialize the IK solver with robot model and parameters.

        Parameters:
        urdf_file (str): Path to the URDF file of the robot.
        default_pose (ndarray): The default joint positions of the robot.
        eps (float): Convergence threshold for the IK solver.
        it_max (int): Maximum number of iterations for the IK solver.
        dt (float): Time step for integration.
        damp (float): Damping factor for the Jacobian.
        limit_damp (float): Damping factor for joint limits.
        """
        self.model = pin_model
        self.data = pin_data
        self.default_pose = default_pose
        self.eps = eps
        self.it_max = it_max
        self.dt = dt
        self.damp = damp
        self.limit_damp = limit_damp
        self.legs = ['FL', 'FR', 'RL', 'RR']
        self.frames = [self.get_frame_id(f"{leg}_foot") for leg in self.legs]
        self.damping = np.eye(3) * damp
        self.base_damping = np.eye(6) * 3e-2
        self.K_p = np.eye(3) * 1e-12
        self.success = False
        self.q_offset = 7
        self.dq_offset = 6
        self.max_update_norm = 2.0

    def get_joint_limits(self):
        """
        Get the joint limits of the robot model.

        Returns:
        ndarray: Array of joint limits with lower and upper bounds.
        """
        joint_limits = np.zeros((self.model.nq, 2))
        for i in range(self.model.nq):
            joint_limits[i] = self.model.lowerPositionLimit[i], self.model.upperPositionLimit[i]
        return joint_limits

    def get_joint_names(self):
        """
        Get the names of all joints in the robot model.

        Returns:
        list: List of joint names.
        """
        joint_names = []
        for i in range(self.model.njoints):
            joint_names.append(self.model.names[i])
        return joint_names

    def get_frame_names(self):
        """
        Get the names of all frames in the robot model.

        Returns:
        list: List of frame names.
        """
        frame_names = []
        for i in range(self.model.nframes):
            frame_names.append(self.model.frames[i].name)
        return frame_names

    def get_frame_id(self, frame_name):
        """
        Get the frame ID from the frame name.

        Parameters:
        frame_name (str): The name of the frame.

        Returns:
        int: The ID of the frame.
        """
        return self.model.getFrameId(frame_name)
    
    def get_joint_id(self, joint_name):
        """
        Get the joint ID from the joint name.

        Parameters:
        joint_name (str): The name of the joint.

        Returns:
        int: The ID of the joint.
        """
        return self.model.getJointId(joint_name)

    def compute_ik(self,targets_pos):
        """
        Compute the inverse kinematics to achieve the target foot positions.
        targets (list): history of target feet positions. T,4

        Returns:
        tuple: Final joint positions and the final error.
        """
        i = 0
        
        root_quat = targets_pos[3:self.q_offset].copy()
        # to mat
        base_mat = quat2mat(root_quat)
        base_pos = targets_pos[:3].copy()
        root_quat = np.roll(root_quat,-1) # wxyz -> xyzw

        q = self.default_pose.copy()
        q[:3] = base_pos.copy()
        legs_target_pose = targets_pos[self.q_offset:].reshape(-1,3).copy()
        targets_pos_SE3_legs = [pin.SE3(np.eye(3), legs_target_pose[i]) for i in range(4)]
        base_oMd = pin.SE3(base_mat, base_pos)
        while i < self.it_max:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q)
            
            # compute dq for base
            base_frame_id = self.get_frame_id("base")
            base_dMo = base_oMd.actInv(self.data.oMf[base_frame_id])
            base_err = pin.log6(base_dMo).vector # 6 dims
            # base_J = pin.getFrameJacobian(self.model, self.data, base_frame_id, pin.LOCAL_WORLD_ALIGNED)[:,:self.dq_offset]
            # base_pseudo_jacobian = base_J.T @ np.linalg.inv(base_J @ base_J.T + self.base_damping)
            # base_dq = -base_pseudo_jacobian @ base_err

            legs_err = []
            legs_dq = []
            legs_pos = []
            for id, frame_id in enumerate(self.frames):
                J = pin.getFrameJacobian(self.model, self.data, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, self.dq_offset:]

                leg_jac = J[:, id * 3:id * 3 + 3]
                legs_cur_pos = self.data.oMf[frame_id].translation
                legs_tar_pos = targets_pos_SE3_legs[id].translation.flatten()
                legs_pos.append(legs_cur_pos)
                err = legs_tar_pos - legs_cur_pos
                legs_err.append(err)
                residual = self.K_p @ (self.default_pose[self.q_offset+id * 3: self.q_offset+ id * 3 + 3] - q[self.q_offset+id * 3:self.q_offset+id * 3 + 3])
                pseudo_jacobian = leg_jac.T @ np.linalg.inv(leg_jac @ leg_jac.T + self.damping)
                null_space = (np.eye(3) - pseudo_jacobian @ leg_jac) @ residual
                leg_dq = pseudo_jacobian @ err + null_space
                legs_dq.append(leg_dq)

            dq = np.zeros(self.model.nv)
            dq[:self.dq_offset] = 0.0
            for id, leg_dq in enumerate(legs_dq):
                dq[self.dq_offset+id * 3:self.dq_offset+id * 3 + 3] = leg_dq
            
            
            new_q = pin.integrate(self.model, q, dq * self.dt)
            updated_q_norm = np.linalg.norm(new_q[self.q_offset:] - q[self.q_offset:])
            if updated_q_norm > self.max_update_norm:
                new_q[self.q_offset:] = q[self.q_offset:] + self.max_update_norm * (new_q[self.q_offset:] - q[self.q_offset:]) / updated_q_norm
            q[self.q_offset:] = new_q[self.q_offset:]
            q[:3] = base_pos.copy()
            q[3:self.q_offset] = root_quat.copy()

            total_err = np.linalg.norm(np.concatenate(legs_err))

            limit_err = self.limit_damp * (np.linalg.norm(np.maximum(self.model.lowerPositionLimit - q, 0)) + np.linalg.norm(np.maximum(q - self.model.upperPositionLimit, 0)))
            total_err += limit_err
            # clip the joint limits
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
            if total_err <=  self.eps:
                self.success = True
                break
            i += 1
        if not self.success:
            print("IK solver did not converge. The target may be out of reach. Please check the taget root position! check: scale_factor and skew_factor!")
            # pass
        # here q[3:7] is in xyzw format
        # we need to convert it back to wxyz
        root_quat = q[3:self.q_offset].copy()
        root_quat = np.roll(root_quat,1)
        q[3:self.q_offset] = root_quat #wxyz
     
        return q, np.asarray(legs_pos).flatten(), limit_err
