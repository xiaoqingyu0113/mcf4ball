import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from mcf4ball.camera import  load_params,draw_camera,world2camera
from mcf4ball.draw_util import set_axes_equal
from mcf4ball.predictor import predict_trajectory

CURRENT_DIR = os.path.dirname(__file__)


def add_noise_to_uv(obs,std=3):
    for uv in obs:
        uv += np.random.normal(0, std, uv.shape)

def init_camera_params():
    camera_param_list = []
    for i in range(6):
        camera_param = load_params(CURRENT_DIR + f"/camera_calibration_data/camera_param{i+1}_mar13.json")
        if i >= 3:
            camera_param.R=  camera_param.R @ np.array([[-1,0,0],[0,-1,0],[0,0,1]]).T
            camera_param.t = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) @ camera_param.t + np.array([0,-12.8,0])
        camera_param_list.append(camera_param)

    return camera_param_list



def make_data(total_time=2.5,like_bag=False):
    camera_param_list = init_camera_params()
    mag_vel = 15
    ang = 50
    p0 = np.array([0,-15,1])
    v0  =np.array([0,mag_vel*np.sin(np.radians(ang)),mag_vel*np.cos(np.radians(ang))])
    w0 = np.array([0,0, 20])*2*np.pi

    time_ticks, xN1 = predict_trajectory(p0,v0,w0,z0=0,total_time=total_time)

    obs = []
    for i in range(3):
        uv = world2camera(xN1[:,:3],camera_param_list[i])
        obs.append(uv)
    
    add_noise_to_uv(obs,std=3)

    if not like_bag:
        return obs,xN1,camera_param_list
    else:
        data_array = []
        for i in range(len(xN1)):
            this_cam_id = i%3
            this_uv = obs[this_cam_id][i]
            data_array.append([time_ticks[i],i%3,this_uv[0],this_uv[1]])
        return np.array(data_array),xN1,camera_param_list

def main():
    obs,xN1,camera_param_list = make_data()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(xN1[:,0], xN1[:,1],xN1[:,2], 'g', markerfacecolor='black', markersize=1,label='top spin')
    for i in range(3):
        draw_camera(camera_param_list[i].R, camera_param_list[i].t, color='blue', scale=1, ax=ax)
    set_axes_equal(ax)
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    plt.show()

    fig2 = plt.figure()
    ax = fig2.add_subplot()
    uv1 = obs[1]
    ax.scatter(uv1[:,0],uv1[:,1],1,'b','.')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    plt.show()

    fig3 = plt.figure()
    ax = fig3.add_subplot()
    for i in range(3):
        ax.plot(xN1[:,3+i])
    plt.show()

    

if __name__=='__main__':
    main()