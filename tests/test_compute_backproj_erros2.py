from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import csv
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader, Subset, random_split
from torch import nn
from mcf4ball.pose2spin import *
import mcf4ball.parameters as param
from mcf4ball.estimator import IsamSolver
from mcf4ball.predictor import predict_trajectory
from mcf4ball.draw_util import  set_axes_equal,comet,draw_tennis_court, plot_spheres
from mcf4ball.camera import  CameraParam

from test_pose2spin_train import CustomDataset
from test_helper import convert2camParam, load_detections,read_yaml








def compute_backproj_error(data_array,camera_param_list,graph_minimum_size,angular_prior,s,e):
    gtsam_solver = IsamSolver(camera_param_list,
                              verbose = False,
                              graph_minimum_size = graph_minimum_size,
                              ez = param.ez,
                              Le = param.Le,
                                Cd=param.Cd,
                              exy = param.exy,
                               ground_z0=param.ground_z0,
                              angular_prior = angular_prior)
    saved_p = [];saved_v = [];saved_w = [];saved_w0 = [];saved_iter = []; saved_gid = [];saved_time = []; saved_landing_seq=[]
    saved_error_u = []; saved_error_v = []

    localize_ind = 0
    num_w = 0
    for d in data_array:
        iter = int(d[0])
        if s<iter and iter<e:
            data = [float(d[1]),int(d[2])-1,float(d[3]),float(d[4])]
            gtsam_solver.push_back(data)
            rst = gtsam_solver.get_result()
            if rst is not None:
                p_rst,v_rst,w_rst = rst
                uv_backproj = camera_param_list[data[1]].proj2img(p_rst)
                saved_error_u.append(uv_backproj[0] - d[3])
                saved_error_v.append(uv_backproj[1] - d[4])

            if num_w == 1 and gtsam_solver.num_w==2:
                break
            num_w = gtsam_solver.num_w
    mean_u = np.mean(saved_error_u)
    std_u = np.std(saved_error_u)
    mean_v = np.mean(saved_error_v)
    std_v = np.std(saved_error_v)

    print(f"average error for u is {mean_u:.4f} +- {std_u}")
    print(f"average error for v is {mean_v:.4f} +- {std_v}")

    # return saved_p, saved_v, saved_w, saved_iter,saved_landing_seq



def run_prediction(folder_name_only,i):
    dataset = CustomDataset('dataset', max_seq_size = 100, seq_size = 20)
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)
    # model = torch.load('trained/tcnn.pth').to(device)
    # model.eval()
    graph_minimum_size = 20

    dataset_dict = dataset.dataset_dict

    # folder_name = 'tennis_14'
    # human_poses = dataset_dict[folder_name]['poses']
    iters = dataset_dict[folder_name_only]['iters']
    labels = dataset_dict[folder_name_only]['labels']

    # i =1
    # hp = torch.from_numpy(human_poses[i]).float().to(device)[list(range(0,100,5))]
    s,e = iters[i]

    print(f'computing {folder_name_only} ({i}/{len(iters)})')
    # with torch.no_grad():
    #     angular_prior = model(hp[None,:])
    # angular_prior = angular_prior.to('cpu').numpy()[0]
    angular_prior = labels[i]
    data_array = load_detections('dataset/'+folder_name_only)
    compute_backproj_error(data_array,camera_param_list,graph_minimum_size,angular_prior,s,e)





if __name__ == '__main__':
    folder_name_only = 'tennis_4'
    i = 0
    run_prediction(folder_name_only,i)