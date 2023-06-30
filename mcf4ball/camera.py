import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import json
import gtsam 

def read_cameraParams(json_file):
    with open(json_file,'r') as f:
        camera_param = json.load(f)
    return camera_param

def load_params(path):
    camera_param1 = read_cameraParams(path)
    K1 = np.array(camera_param1['K']).reshape(3,3); R_1w = np.array(camera_param1['R']).reshape(3,3); t_1w_1 = np.array(camera_param1['t'])
    R_w1 = R_1w.T; t_w1_w = -R_w1 @ t_1w_1
    return CameraParam(K1,R_1w,t_w1_w)




# def load_params(dir_path):

#     camera_param1 = read_cameraParams(dir_path+'/camera_calibration_data/camera_param1.json')
#     camera_param2 = read_cameraParams(dir_path+'/camera_calibration_data/camera_param2.json')
#     camera_param3 = read_cameraParams(dir_path+'/camera_calibration_data/camera_param3.json')

#     K1 = np.array(camera_param1['K']).reshape(3,3); R_1w = np.array(camera_param1['R']).reshape(3,3); t_1w_1 = np.array(camera_param1['t'])
#     K2 = np.array(camera_param2['K']).reshape(3,3); R_2w = np.array(camera_param2['R']).reshape(3,3); t_2w_2 = np.array(camera_param2['t'])
#     K3 = np.array(camera_param3['K']).reshape(3,3); R_3w = np.array(camera_param3['R']).reshape(3,3); t_3w_3 = np.array(camera_param3['t'])

#     R_w1 = R_1w.T; t_w1_w = -R_w1 @ t_1w_1
#     R_w2 = R_2w.T; t_w2_w = -R_w2 @ t_2w_2
#     R_w3 = R_3w.T; t_w3_w = -R_w3 @ t_3w_3

#     R_offset = np.eye(3)
#     offset = np.zeros(3)
#     scale = camera_param1['scale']
#     return CameraParam(K1,R_1w,t_w1_w),CameraParam(K2,R_2w,t_w2_w),CameraParam(K3,R_3w,t_w3_w), R_offset,offset,scale

def draw_camera(R, t, color='blue', scale=1, ax=None):

    '''
     Draw camera pose on matplotlib axis, for debugging or visualization
    '''
    points = np.array([[0,0,0],[1,1,2],[0,0,0],[-1,1,2],[0,0,0],[1,-1,2],[0,0,0],[-1,-1,2],
                        [-1,1,2],[1,1,2],[1,-1,2],[-1,-1,2]]) * scale
    x,y,z = R .T @ points.T + t[:,np.newaxis]

    if ax is not None:
        ax.plot(x,y,z,color=color)
        return ax
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(x,y,z,color=color)
        return ax

def axis_equal(ax,X,Y,Z):
   # Set the limits of the axes to be equal

    x = np.array(X).flatten()
    y = np.array(Y).flatten()
    z = np.array(Z).flatten()

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

class CameraParam:
    '''
    A camera paramter structure
    self.K  - intrinsics
    self.R  - camera orientation wrt world frame, w_R_cam
    self.t  - camera position wrt world frame, w_t_cam
    '''
    def __init__(self,K,R,t):
        self.K = K # intrinsics
        self.R = R # camera w.r.t world
        self.t =t # camera's position w.r.t world
        
    def parser(self):
        return self.K, self.R, self.t

    def to_gtsam(self):
        K1 = self.K
        R1 = self.R
        t1 = self.t
        K1_gtsam = gtsam.Cal3_S2(K1[0,0], K1[1,1], K1[2,2], K1[0,2], K1[1,2])
        R1_gtsam = gtsam.Rot3(R1.T) 
        t1_gtsam = gtsam.Point3(t1[0],t1[1],t1[2])
        pose1 = gtsam.Pose3(R1_gtsam, t1_gtsam) # pose should be camera pose in the world frame
        return K1_gtsam, pose1        

def world2camera(p,camera_params:CameraParam):
    K,R,t = camera_params.parser()
    if len(p.shape) ==1:
        uv_ = K@np.block([R,-R@t[:,np.newaxis]])@np.append(p,1)
        uv1 = uv_/uv_[2]
        return uv1[:2]
    else:
        N = p.shape[0]
        uv_ = K@np.block([R,-R@t[:,np.newaxis]])@np.append(p.T,np.ones((1,N)),axis=0)
        uv1 = uv_/uv_[2,:]
    return uv1.T


def projection_img2world_line(uv,camera_params,z=-2.0):

    '''
        Solve for a projection line from image to realworld.

        input:
            uv - point on image
            camera_params - camera paramters
            z - end point of the projection line 
    '''

    M = camParam2proj(camera_params)
    
    u = uv[0];v = uv[1]
    A = np.array([
        [M[0,0] - u*M[2,0], M[0,1] - u*M[2,1]],
        [M[1,0] - v*M[2,0], M[1,1] - v*M[2,1]]
        ])
    b1 = np.array([
        [M[0,2] - u*M[2,2]],
        [M[1,2] - v*M[2,2]]
        ])
    b2 = np.array([
        [M[0,3] - u*M[2,3]],
        [M[1,3] - v*M[2,3]]
        ])
    xy = np.linalg.solve(A,-b1*z-b2)

    start = camera_params.t
    end = np.append(xy,z)

    return start, end

def plot_line_3d(ax,start, end, **kwargs):
    '''
        plot a line using starting point and end point 
    '''
    ax.plot([start[0],end[0]], [start[1],end[1]],[start[2],end[2]],**kwargs)


def camParam2proj(cam_param):
    '''
    computing for projection matrix
    '''
    t = np.array(cam_param.t)
    t = np.expand_dims(t,axis = 1)
    R = np.array(cam_param.R)
    K = np.array(cam_param.K)
    M = K @ np.block([R,-R@t])
    return M

def closest_points_twoLine(p0,p1,q0,q1): # https://blog.csdn.net/Hunter_pcx/article/details/78577202

    '''
    solving for closest point of two line in 3d space 
    '''
    A = np.c_[p0-p1,q0-q1]
    b = q1-p1

    # solve normal equation
    x = np.linalg.solve(A.T@A, A.T@b)
    
    a = x[0]
    n = -x[1]

    pc = a*p0 + (1-a)*p1
    qc = n*q0 + (1-n)*q1

    return pc, qc

def triangulation(uv1,uv2,uv3,cam_param1, cam_param2, cam_param3):
    THRESH = 1.0
    closest_pts = []

    false_detection = np.array([uv1[0],uv2[0],uv3[0]]) == -1
    available_idx = np.arange(3)
    available_idx = available_idx[~false_detection]
    N_available = len(available_idx)

    uv_list = [uv1,uv2,uv3]
    cam_param_list = [cam_param1, cam_param2, cam_param3]

    if N_available <2:
        return np.array([-111,-111,-111]),-111

    elif N_available == 2:
        p0,p1 =  projection_img2world_line(uv_list[available_idx[0]],cam_param_list[available_idx[0]],z=-4.0)
        q0,q1 =  projection_img2world_line(uv_list[available_idx[1]],cam_param_list[available_idx[1]],z=-4.0)
        pc,qc = closest_points_twoLine(p0,p1,q0,q1)
        if np.linalg.norm(pc-qc) > THRESH:
            return np.array([-222,-222,-222]),-222
        else:
            return (pc+qc)/2.,  np.linalg.norm(pc-qc)/2.

    else:

        p0,p1 = projection_img2world_line(uv1,cam_param1,z=-4.0)
        q0,q1 = projection_img2world_line(uv2,cam_param2,z=-4.0)
        r0,r1 = projection_img2world_line(uv3,cam_param3,z=-4.0)

        
        pc,qc = closest_points_twoLine(p0,p1,q0,q1)
        if np.linalg.norm(pc-qc)<THRESH:
            closest_pts.append([pc]); closest_pts.append([qc])
        pc,rc = closest_points_twoLine(p0,p1,r0,r1)
        if np.linalg.norm(pc-rc)<THRESH:
            closest_pts.append([pc]); closest_pts.append([rc])
        qc,rc = closest_points_twoLine(q0,q1,r0,r1)
        if np.linalg.norm(rc-qc)<THRESH:
            closest_pts.append([qc]); closest_pts.append([rc])

        closest_pts = np.array(closest_pts).reshape((-1,3))

        if len(closest_pts) ==0:
            return np.array([-333,-333,-333]),-333
        else:
            # print(closest_pts)
            return closest_pts.mean(0), closest_pts.std(0)
       


def pairwise_dist(x,y):
    """
    x: N,D
    y: M,D

    return N,M
    """
    d_sq = np.sum(x**2,axis=1)[:,np.newaxis] + np.sum(y**2,axis=1)[np.newaxis,:] - 2*x @ y.T

    d_sq[d_sq<0] = 0

    return np.sqrt(d_sq)

def rotm(th,axis):
    if axis == 'z':
        return np.array([[np.cos(th),-np.sin(th),0],
                        [np.sin(th),np.cos(th),0],
                        [0,0,1]]) 
    elif axis=='y':
        return np.array([[np.cos(th),0,np.sin(th)],
                        [0,1,0],
                        [-np.sin(th),0,np.cos(th)]])
    elif axis == 'x':
        return np.array([[1,0,0],
                            [0, np.cos(th),-np.sin(th)],
                            [0,np.sin(th),np.cos(th)]])
    else:
        ValueError("axis not identified")