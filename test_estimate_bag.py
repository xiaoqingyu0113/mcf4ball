
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from test_makedata import make_data

from mcf4ball.estimator import IsamSolver
from mcf4ball.predictor import predict_trajectory

from mcf4ball.draw_util import  set_axes_equal,comet
from mcf4ball.camera import  load_params, triangulation

CURRENT_DIR = os.path.dirname(__file__)

def load_data():
    with open('bag.csv', mode='r') as file:
        reader = csv.reader(file)
        data_list = [row for row in reader]
    data_array = np.array(data_list)
    return data_array

def init_camera_params():
    camera_param_list = []
    for i in range(6):
        camera_param = load_params(CURRENT_DIR + f"/camera_calibration_data/camera_param{i+1}_mar13.json")
        if i >= 3:
            camera_param.R=  camera_param.R @ np.array([[-1,0,0],[0,-1,0],[0,0,1]]).T
            camera_param.t = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) @ camera_param.t + np.array([0,-12.8,0])
        camera_param_list.append(camera_param)

    return camera_param_list


def main():
    data_array = load_data()
    camera_param_list = init_camera_params()
    print(f"make total number of {data_array.shape[0]} data")

    saved_p = []
    saved_v = []
    saved_w = []

    graph_minimum_size = 150
    gtsam_solver = IsamSolver(camera_param_list)
    total_iter = len(data_array)
    prev_cam_id = 0
    total_time = -time.time()
    for iter,data in enumerate(data_array):
        print(f"\niter = {iter}")
        if iter < 400:
            continue
        # print(f"{float(data[0])} > {gtsam_solver.t_max} = {float(data[0]) > gtsam_solver.t_max}")
        if  (int(data[1])==3) or (int(data[1])==4) or (int(data[1])==5) :
            continue
        if iter > 2000:
            break
        if float(data[0]) > gtsam_solver.t_max:
            if gtsam_solver.curr_node_idx < graph_minimum_size:
                gtsam_solver.update(data,optim=False)
            else:
                gtsam_solver.update(data,optim=True)
                rst = gtsam_solver.get_result()
                p_rst,v_rst,w_rst = rst
                # if np.linalg.norm(w_rst) > 1000:
                #     continue
                saved_p.append(p_rst)
                saved_w.append(w_rst)
                saved_v.append(v_rst)
        prev_cam_id = int(data[1])
    total_time += time.time()
    print('average inference time (hz) = ', (total_iter-graph_minimum_size)/total_time)


    saved_p = np.array(saved_p)
    saved_v = np.array(saved_v)
    saved_w = np.array(saved_w)

    print(saved_p.shape)
    # -------------------------------------- plain plot ------------------------------
    # comet(saved_p,saved_v,saved_w,predict_trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    ax.plot(saved_p[:,0], saved_p[:,1],saved_p[:,2], 'b.', markerfacecolor='black', markersize=3)
    ax.set_title('uses cameras 1-3')
    set_axes_equal(ax)
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    plt.show()


    fig, axs = plt.subplots(1, 2)  # This will create a 2x2 grid of Axes objects
    labels = ['x','y','z']
    gt_color = ['r','g','b']

    for i in range(3):
        axs[0].plot(saved_v[:,i],'o',label=labels[i])

        axs[1].plot(saved_w[:,i]/(2*np.pi),'o',label=labels[i]) # convert to herz

    axs[0].set_ylabel('linear velocity (m/s)')    
    axs[1].set_ylabel('angular velocity (Hz)')    

    axs[0].legend() 
    axs[1].legend() 

    plt.show()

if __name__ == '__main__':
    main()