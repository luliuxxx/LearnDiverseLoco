import numpy as np
def cal_angular_velocities(q1,q2,dt):
    # https://mariogc.com/post/angular-velocity-quaternions/
    # q1 = q[t-1] wxyz
    # q2 = q[t]
    # dt = 1/fps
    # return w_x, w_y, w_z
    return (2 / dt) * np.array([
            q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
            q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
            q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])


def get_linear_vel( x):
    vel = x.copy()
    vel[1:,:] = vel[1:,:] - vel[:-1,:]
    vel[0,:] = 0
    return vel

