
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
from test_helper import convert2camParam, load_detections, read_yaml, read_csv, write_rows_to_csv

def run_localization_for_one_trajectory(detections,gtsam_solver,ds,de,return_all = False):

    if not return_all:
        w0 = np.zeros(3)
        num_w = 0
        for d in detections[ds:de,:]:
            data = [d[1], int(d[2]) - 1, d[3], d[4]] # time, camera index, u, v
            gtsam_solver.push_back(data)
            rst = gtsam_solver.get_result()
            count_landing1 = 0
            count_landing2 = 0
            if rst is not None:
                if num_w ==0 and gtsam_solver.num_w ==0:
                    w0 = gtsam_solver.current_estimate.atVector(W(0))
                if num_w == 0 and gtsam_solver.num_w ==1:
                    w0 = gtsam_solver.current_estimate.atVector(W(0))
                    count_landing1 += 1
                if 0<count_landing1 < 20:
                    w0 = gtsam_solver.current_estimate.atVector(W(0))
                    count_landing1 += 1
                if num_w ==1 and gtsam_solver.num_w ==2:
                    w0 =  gtsam_solver.current_estimate.atVector(W(0))
                    count_landing2 += 1 
                    count_landing1 = 0
                if (0<count_landing2<20) and num_w ==2 and gtsam_solver.num_w ==2:
                    w0 =  gtsam_solver.current_estimate.atVector(W(0))
                    count_landing2 += 1 

                num_w = gtsam_solver.num_w
        return w0
    else:
        saved_p = [];saved_v = [];saved_w = [];saved_w0 = [];saved_iter = []; saved_gid = [];saved_time = []
        for d in detections[ds:de,:]:
            data = [d[1], int(d[2]) - 1, d[3], d[4]] # time, camera index, u, v
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
        saved_p = np.array(saved_p);saved_v = np.array(saved_v);saved_w = np.array(saved_w)
        saved_w0 = np.array(saved_w0);saved_iter = np.array(saved_iter)[:,None];saved_gid = np.array(saved_gid)[:,None]
        saved_time = np.array(saved_time)[:,None]

        return saved_p,saved_v,saved_w,saved_w0,saved_iter,saved_gid,saved_time


def run_label(folder_name):
    detections = load_detections(folder_name)
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)
    separate_ind = read_csv(folder_name+'/d_separate_ind.csv').astype(int)
    N_traj = len(separate_ind)
    if N_traj == 0:
        return

    graph_minimum_size = 20
    gtsam_solver = IsamSolver(camera_param_list,
                                    verbose = False,
                                    graph_minimum_size = graph_minimum_size,
                                    ez = param.ez,
                                    exy = param.exy,
                                    ground_z0=param.ground_z0,
                                    angular_prior = np.zeros(3))
    spin_priors = []

    print(f'\n----------------------start iteration of folder {folder_name}-------------------------')
    for traj_ind, s_index in enumerate(separate_ind):
        angular_prior = np.random.rand(3)*0
        prev_angular_prior = angular_prior

        error = np.inf
        rs, re, its, ite,ds,de = s_index
        
        sol_count = 0
        while error > .5 * np.pi*2 and sol_count < 100:
            gtsam_solver.reset()
            gtsam_solver.set_angular_prior(angular_prior)
            angular_prior = run_localization_for_one_trajectory(detections,gtsam_solver,ds,de)
            error = np.linalg.norm(prev_angular_prior - angular_prior)
            prev_angular_prior = angular_prior
            sol_count += 1
            print(f"folder = {folder_name}\ttraj_ind = {traj_ind}/{N_traj}\terror={error}")
        spin_priors.append(angular_prior)
        print(f'solved angular prior = {angular_prior}')
    write_rows_to_csv(folder_name+'/d_spin_priors.csv',np.array(spin_priors))

    print('\nspin priors successfully written in ' + folder_name+'/d_spin_priors.csv')

def plot_traj_and_spin(folder_name, traj_ind):
    separate_ind = read_csv(folder_name+'/d_separate_ind.csv').astype(int)
    spin_pirors = read_csv(folder_name+'/d_spin_priors.csv').astype(float)
    rs, re, its, ite,ds,de = separate_ind[traj_ind,:]
    print(f'plotting {folder_name:} {traj_ind}/{len(spin_pirors)}')

    detections = load_detections(folder_name)
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)
    separate_ind = read_csv(folder_name+'/d_separate_ind.csv').astype(int)
    N_traj = len(separate_ind)
    if N_traj == 0:
        return
    graph_minimum_size = 20
    gtsam_solver = IsamSolver(camera_param_list,
                                    verbose = False,
                                    graph_minimum_size = graph_minimum_size,
                                    ez = param.ez,
                                    exy = param.exy,
                                    ground_z0=param.ground_z0,
                                    angular_prior = spin_pirors[traj_ind])
    
    saved_p,saved_v,saved_w,saved_w0,saved_iter,saved_gid,saved_time = run_localization_for_one_trajectory(detections,gtsam_solver,ds,de,return_all = True)

    fig = plt.figure()

    ax1 = fig.add_subplot(121,projection='3d')
    ax1.plot(saved_p[:,0],saved_p[:,1],saved_p[:,2])
    set_axes_equal(ax1)
    draw_tennis_court(ax1)
    ax1.set_xlabel('x');ax1.set_ylabel('y');ax1.set_zlabel('z')

    ax2 = fig.add_subplot(122)
    ax2.plot(saved_w0[:,0],label='wx')
    ax2.plot(saved_w0[:,1],label='wy')
    ax2.plot(saved_w0[:,2],label='wz')
    ax2.set_xlabel('# of camera detections')
    ax2.set_ylabel('initial spin prior')

    plt.show()
if __name__ == '__main__':
    # run all folder
    folders = glob.glob('dataset/tennis_*')
    for folder_name in folders:
        print('processing ' + folder_name)
        run_label(folder_name)
    
    # folder_name = 'dataset/tennis_6'
    # run_label(folder_name)

    # plot_traj_and_spin('dataset/tennis_19',0)