
import csv
import numpy as np
import yaml
import os
import glob
import matplotlib.pyplot as plt

import mcf4ball.parameters as param

from mcf4ball.estimator import IsamSolver
from mcf4ball.predictor import predict_trajectory

from mcf4ball.draw_util import  set_axes_equal,comet,draw_tennis_court, plot_spheres
from mcf4ball.camera import  CameraParam
from gtsam.symbol_shorthand import X,L,V,W

'''
iterate and minimize the error at the beginning and in the end
'''
CURRENT_DIR = os.path.dirname(__file__)


def convert2camParam(params):
    camera_params = []
    for p in params:
        K = np.array(p['camera_matrix']['data']).reshape(3,3)
        R = np.array(p['R_cam_world']).reshape(3,3)
        t = np.array(p['t_world_cam'])
        camera_params.append(CameraParam(K,R,t))
    return camera_params

def load_data(folder_name):
    with open(folder_name+'/detections.csv', mode='r') as file:
        reader = csv.reader(file)
        data_list = []
        for row in reader:
            if len(row) == 5:
                data_list.append(row)
    data_array = np.array(data_list)
    return data_array

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")


def run_label(folder_name):
    data_array = load_data(folder_name)
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)

    

    graph_minimum_size = 20


    with open(folder_name+'/d_sperate_id.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        separate_indices = np.array(list(reader),dtype=int)

    spin_priors = []
    traj_id = 0

    for s_index in separate_indices:
        angular_prior = np.array([0,0,0])*6.28
        error = np.inf
        start_idx, end_idx, start_iter, end_iter = s_index
        sol_count = 0
        while error > .5 * np.pi*2 and sol_count < 100:
            
            gtsam_solver = IsamSolver(camera_param_list,
                                    verbose = False,
                                    graph_minimum_size = graph_minimum_size,
                                    ez = param.ez,
                                    exy = param.exy,
                                    ground_z0=param.ground_z0,
                                    angular_prior = angular_prior)
            saved_p = [];saved_v = [];saved_w = [];saved_w0 = [];saved_iter = []; saved_gid = [];saved_time = []

            for d in data_array:
                iter = int(d[0])
                if iter < start_iter:
                    continue
                if iter >= end_iter:
                    break

                data = []
                data.append(float(d[1]))
                data.append(int(d[2])-1)
                data.append(float(d[3]))
                data.append(float(d[4]))


                gtsam_solver.push_back(data)
                rst = gtsam_solver.get_result()
                if rst is not None:
                    p_rst,v_rst,w_rst = rst
                    saved_p.append(p_rst)
                    saved_w0.append(gtsam_solver.current_estimate.atVector(W(0)))
                    saved_w.append(w_rst)
                    saved_v.append(v_rst)
                    saved_iter.append(iter)
                    saved_gid.append(gtsam_solver.curr_node_idx)
                    saved_time.append(float(d[1]))

            prev_angular_prior = angular_prior
            angular_prior = saved_w0[-1]
            error = np.linalg.norm(prev_angular_prior - angular_prior)
            sol_count += 1
            # print(f"-------------\ntraj={traj_id}, count = {sol_count}, error = {error}")
            # print(f"w0 beginning = {saved_w0[0]}")
            # print(f"prev_angular_prior = {prev_angular_prior}")
            # print(f"w0 end = {saved_w0[-1]}")

            
        spin_priors.append(angular_prior)
        traj_id +=1

        print(f'\n======trj={traj_id}==error={error}================= ')
        print(f"w0 beginning = {saved_w0[0]}")
        print(f"w0 end = {saved_w0[-1]}")
        print('=============')

    spin_priors = np.array(spin_priors)
    with open(folder_name+'/d_spin_priors.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(spin_priors)
    print('\nspin priors successfully written in ' + folder_name)
    for sp in spin_priors:
        print(sp)

    # save as numpy array
    # saved_p = np.array(saved_p)
    # saved_v = np.array(saved_v)
    # saved_w = np.array(saved_w)
    # saved_w0 = np.array(saved_w0)
    # saved_iter = np.array(saved_iter)[:,None]
    # saved_gid = np.array(saved_gid)[:,None]
    # saved_time = np.array(saved_time)[:,None]


    # fig, axs = plt.subplots(1, 2)  # This will create a 2x2 grid of Axes objects
    # labels = ['x','y','z']
    # gt_color = ['r','g','b']

    # for i in range(3):
    #     axs[0].plot(saved_v[:,i],'o',label=labels[i])
    #     axs[1].plot(saved_w0[:,i]/(2*np.pi),'o',label=labels[i]) # convert to herz

    # axs[0].set_ylabel('linear velocity (m/s)')    
    # axs[1].set_ylabel('angular velocity (Hz)')    

    # axs[0].legend() 
    # axs[1].legend() 

    # fig.savefig('velocities.jpg')
    
def show_label(folder_name):
    with open(folder_name+'/d_spin_priors.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        spins = np.array(list(reader),dtype=float)/(2*np.pi)
    indices = np.arange(len(spins))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(indices,spins[:,0],20)
    ax.scatter(indices, spins[:,1],20)
    ax.scatter(indices,spins[:,2],20)
    plt.show()
if __name__ == '__main__':
    # run all folder
    folders = glob.glob('dataset/tennis_*')
    for folder_name in folders:
        print('processing ' + folder_name)
        run_label(folder_name)
    # run all folder

    # folder_name = 'dataset/tennis_1'
    # run_label(folder_name)

    # folder_name = 'dataset/tennis_10'
    # show_label(folder_name)