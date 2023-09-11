
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

from test_helper import convert2camParam, load_detections, read_yaml, save_list_to_csv,write_rows_to_csv,read_csv

def run_and_save_estimation(folder_name):
    '''
    1. compute the 3d estimation of balls using yolo detector
    2. save in d_results.csv
    '''
    detections = load_detections(folder_name)
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)

    saved_p = [];saved_v = [];saved_w = [];saved_w0 = [];saved_iter = []; saved_gid = [];saved_time = []; saved_detection_id= []

    graph_minimum_size = 20
    angular_prior = np.zeros(3)
    gtsam_solver = IsamSolver(camera_param_list,
                              verbose = False,
                              graph_minimum_size = graph_minimum_size,
                              ez = param.ez,
                              exy = param.exy,
                               ground_z0=param.ground_z0,
                              angular_prior = angular_prior)
    
    detection_length = len(detections)
    for detection_ind, d in enumerate(detections):
        iter = int(d[0]) 
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
            saved_detection_id.append(detection_ind)
        if detection_ind % 5000 == 4999:
            print(f"{detection_ind}/{detection_length}")

    # save as numpy array
    save_list_to_csv(folder_name + '/d_results.csv',saved_p,saved_v,saved_w,saved_w0,saved_iter,saved_gid,saved_time,saved_detection_id)
    print(f"saved to {folder_name} /d_results.csv")

if __name__ == '__main__':

    # run all folder
    # folders = glob.glob('dataset/tennis_*')
    # for folder_name in folders:
    #     print('processing ' + folder_name)
    #     run_and_save_estimation(folder_name)


    folder_name = 'dataset/tennis_1'
    run_and_save_estimation(folder_name) # save the estimation result

