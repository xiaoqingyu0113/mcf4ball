
import csv
import numpy as np
import yaml
import os
import glob
import matplotlib.pyplot as plt
import bisect

import mcf4ball.parameters as param

from mcf4ball.estimator import IsamSolver
from mcf4ball.predictor import predict_trajectory

from mcf4ball.draw_util import  set_axes_equal,comet,draw_tennis_court, plot_spheres
from mcf4ball.camera import  CameraParam
from gtsam.symbol_shorthand import X,L,V,W

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


def run_and_save_estimation(folder_name):
    data_array = load_data(folder_name)
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)

    saved_p = [];saved_v = [];saved_w = [];saved_w0 = [];saved_iter = []; saved_gid = [];saved_time = []

    graph_minimum_size = 20

    angular_prior = np.array([0,0,0])*6.28
    
    try:
        with open(folder_name+'/d_spin_priors.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            saved_priors = np.array(list(reader),dtype=float)
        with open(folder_name+'/d_sperate_id.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            separate_indices = np.array(list(reader),dtype=int)
        
    except:
        saved_priors = None
        separate_indices = None



    gtsam_solver = IsamSolver(camera_param_list,
                              verbose = False,
                              graph_minimum_size = graph_minimum_size,
                              ez = param.ez,
                              exy = param.exy,
                               ground_z0=param.ground_z0,
                              angular_prior = angular_prior)

    for d in data_array:
        iter = int(d[0])
        data = []
        data.append(float(d[1]))
        data.append(int(d[2])-1)
        data.append(float(d[3]))
        data.append(float(d[4]))

        if iter % 10000 == 0:
            print(f"iter = {iter}/{int(data_array[-1][0])}")
        # if iter > 5000:
        #     break

        if (separate_indices is not None) and (saved_priors is not None) :
            N_traj = len(saved_priors)
            just_before_iter_index = bisect.bisect_left(separate_indices[:,2], iter)
            just_before_iter_index = just_before_iter_index if just_before_iter_index < N_traj else N_traj-1
            gtsam_solver.angular_prior = saved_priors[just_before_iter_index,:]

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

    # save as numpy array
    saved_p = np.array(saved_p)
    saved_v = np.array(saved_v)
    saved_w = np.array(saved_w)
    saved_w0 = np.array(saved_w0)
    saved_iter = np.array(saved_iter)[:,None]
    saved_gid = np.array(saved_gid)[:,None]
    saved_time = np.array(saved_time)[:,None]

    saved_data = np.concatenate((saved_iter,saved_p,saved_v,saved_w,saved_w0,saved_gid,saved_time),axis=1)

    # write to harddrive
    with open(folder_name+'/d_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(saved_data)

def seperate_traj(gid):
    prev_node_id = gid[0]
    seperator = []
    for i, node_id in enumerate(gid):
        if node_id < prev_node_id:
            seperator.append(i)
        prev_node_id = node_id
    return seperator

def save_separate_idx(folder_name):
    # Open the CSV file in read mode
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)

    saved_iter = data[:,0].astype(int);saved_p = data[:,1:4];saved_v = data[:,4:7]
    saved_w = data[:,7:10]; saved_w0 = data[:,10:13]
    saved_gid = data[:,13]; saved_time = data[:,14]

    traj_seperator = seperate_traj(saved_gid)

    minimum_traj_size = 400
    save_indices= []

    for i in range(len(traj_seperator)-1):
        if (traj_seperator[i+1] - traj_seperator[i] > minimum_traj_size) and (np.abs(saved_p[traj_seperator[i],0] - saved_p[traj_seperator[i+1]-1,0]) > 5.0):
            save_indices.append([traj_seperator[i], traj_seperator[i+1],saved_iter[traj_seperator[i]],saved_iter[traj_seperator[i+1]]])
   
    with open(folder_name+'/d_sperate_id.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(save_indices)

def plot_results(folder_name):
    # Open the CSV file in read mode
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    
    with open(folder_name+'/d_sperate_id.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        separate_indices = np.array(list(reader),dtype=int)


    saved_iter = data[:,0];saved_p = data[:,1:4];saved_v = data[:,4:7]
    saved_w = data[:,7:10]; saved_w0 = data[:,10:13]
    saved_gid = data[:,13]; saved_time = data[:,14]
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i, j in separate_indices[:,:2]:
        ax.plot(saved_p[i:j,0], 
                saved_p[i:j,1],
                saved_p[i:j,2],
                '-', markerfacecolor='black', markersize=3)
    ax.set_title('uses cameras 1-3')
    set_axes_equal(ax)
    draw_tennis_court(ax)
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    plt.show()


def plot_spin(folder_name):
    # Open the CSV file in read mode
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    
    with open(folder_name+'/d_sperate_id.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        separate_indices = np.array(list(reader),dtype=int)


    saved_iter = data[:,0];saved_p = data[:,1:4];saved_v = data[:,4:7]
    saved_w = data[:,7:10]; saved_w0 = data[:,10:13]
    saved_gid = data[:,13]; saved_time = data[:,14]
    
    
    fig, axs = plt.subplots(4, 5, figsize=(12, 10))
    axs = axs.flatten()
    for trj, (i, j) in enumerate(separate_indices[:,:2]):

        # axs[trj].plot(np.arange(i,j),saved_w[i:j,0],'-')
        # axs[trj].plot(np.arange(i,j),saved_w[i:j,1],'-')
        # axs[trj].plot(np.arange(i,j),saved_w[i:j,2],'-')
 
        axs[trj].plot(np.arange(i,j),saved_w0[i:j,0],'--',label='x')
        axs[trj].plot(np.arange(i,j),saved_w0[i:j,1],'--',label='y')
        axs[trj].plot(np.arange(i,j),saved_w0[i:j,2],'--',label='z')
        axs[trj].set_title(f'traj = {trj}')
        axs[trj].legend()
    fig.savefig(f"results/{folder_name.split('/')[-1]}_spin")

def save_as_video():
    # Open the CSV file in read mode
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    saved_iter = data[:,0];saved_p = data[:,1:4];saved_v = data[:,4:7];saved_w = data[:,7:10]
    comet(saved_p[::3],saved_v[::3],saved_w[::3],predict_trajectory)

if __name__ == '__main__':
    folders = glob.glob('dataset/tennis_*')
    for folder_name in folders:
        print('processing ' + folder_name)

        run_and_save_estimation(folder_name)
        save_separate_idx(folder_name)
        plot_spin(folder_name)


    # folder_name = 'dataset/tennis_1'
    # run(folder_name) # save the estimation result

    # save_separate_idx(folder_name)
    # plot_results(folder_name)
    # plot_spin(folder_name)
    # save_as_video()