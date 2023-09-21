
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

def run_backproj_errors(folder_name):
    '''
    1. compute the 3d estimation of balls using yolo detector
    2. save in d_results.csv
    '''
    detections = load_detections(folder_name)
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)

    saved_p = [];saved_v = [];saved_w = [];saved_w0 = [];saved_iter = []; saved_gid = [];saved_time = []; saved_detection_id= []

    saved_error_u = []; saved_error_v = []
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
            uv_backproj = camera_param_list[data[1]].proj2img(p_rst)
            saved_error_u.append(uv_backproj[0] - d[3])
            saved_error_v.append(uv_backproj[1] - d[4])
        if detection_ind % 5000 == 4999:
            print(f"{detection_ind}/{detection_length}")

    mean_u = np.mean(saved_error_u)
    std_u = np.std(saved_error_u)
    mean_v = np.mean(saved_error_v)
    std_v = np.std(saved_error_v)

    print(f"average error for u is {mean_u:.4f} +- {std_u}")
    print(f"average error for v is {mean_v:.4f} +- {std_v}")

if __name__ == '__main__':



    folder_name = 'dataset/tennis_2'
    run_backproj_errors(folder_name) # save the estimation result

