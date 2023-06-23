
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from collections import deque

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

def check_optimizable(obs_buffer,graph_minimum_size = 150):
    prev_camera_id = int(obs_buffer[0][1])
    sum_change = 0
    N = len(obs_buffer)
    for idx in range(N):
        curr_camera_id = int(obs_buffer[idx][1])
        if prev_camera_id != curr_camera_id:
            sum_change += 1
    print(f'check optimizable ({sum_change/graph_minimum_size:.2f})')
    return sum_change/graph_minimum_size > 0.6

def add_buffer(obs_buffer:deque,data,graph_minimum_size = 150):
    if len(obs_buffer) >= graph_minimum_size:
        obs_buffer.popleft()
    obs_buffer.append(data)
        

def main():
    data_array = load_data()
    camera_param_list = init_camera_params()
    print(f"make total number of {data_array.shape[0]} data")

    saved_p = []
    saved_v = []
    saved_w = []

    graph_minimum_size = 150
    gtsam_solver = IsamSolver(camera_param_list,verbose=False)
    total_iter = len(data_array)
    total_time = -time.time()

    obs_buffer = deque()
    optimizable = False
    for iter,data in enumerate(data_array):
        print(f"\niter = {iter}")
        add_buffer(obs_buffer,data)
        if not optimizable: 
            optimizable = check_optimizable(obs_buffer)
        print(optimizable)
        if  (int(data[1])==3) or (int(data[1])==4) or (int(data[1])==5) :
            continue
        if iter > 2000:
            break
        if (float(data[0]) > gtsam_solver.t_max) and optimizable:
            # if iter < 0:
            #     continue
            print(gtsam_solver.curr_node_idx)
            if (gtsam_solver.curr_node_idx < graph_minimum_size):
                gtsam_solver.update(data,optim=False)
            else:
                gtsam_solver.update(data,optim=optimizable)
                rst = gtsam_solver.get_result()
                p_rst,v_rst,w_rst = rst
                if np.linalg.norm(w_rst) > 800:
                    continue
                saved_p.append(p_rst)
                saved_w.append(w_rst)
                saved_v.append(v_rst)
    total_time += time.time()
    print('average inference time (hz) = ', (total_iter-graph_minimum_size)/total_time)


    saved_p = np.array(saved_p)
    saved_v = np.array(saved_v)
    saved_w = np.array(saved_w)

    print(saved_p.shape)
    # -------------------------------------- plain plot ------------------------------
    # comet(saved_p[::3],saved_v[::3],saved_w[::3],predict_trajectory)
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