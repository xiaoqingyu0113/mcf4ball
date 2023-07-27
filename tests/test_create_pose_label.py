
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
import racketpose

from test_trajectory_label import convert2camParam, load_data, read_yaml

CURRENT_DIR = os.path.dirname(__file__)



def read_detections(folder_name):
    with open(folder_name+'/detections.csv', mode='r') as file:
        reader = csv.reader(file)
        data_list = []
        for row in reader:
            if len(row) == 5:
                data_list.append(row)
    data_array = np.array(data_list)
    return data_array

def read_calibration(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def seperate_traj(gid):
    prev_node_id = gid[0]
    seperator = []
    for i, node_id in enumerate(gid):
        if node_id < prev_node_id:
            seperator.append(i)
        prev_node_id = node_id
    return seperator


def run(folder_name):
    data_array = read_detections(folder_name)
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_calibration('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)

    saved_p = [];saved_v = [];saved_w = [];saved_w0 = [];saved_iter = []; saved_gid = [];saved_time = []

    graph_minimum_size = 20
    angular_prior = np.array([0,0,0])*6.28

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

        if iter % 1000 == 1:

            print(f"iter = {iter}/{len(data_array)}")
        # if iter > 5000:
        #     break

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




if __name__ == '__main__':
    folders = glob.glob('dataset/tennis_*')
    for folder_name in folders:
        print('processing ' + folder_name)
        run(folder_name)
