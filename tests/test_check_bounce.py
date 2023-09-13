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
from test_helper import convert2camParam, load_detections,read_yaml,save_list_to_csv,read_csv
from test_prediction_error import computer_trajectory


def run_prediction(folder_name_only):
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

    v1 = [];w1=[];v2=[];w2=[]
    for i in range(len(labels)):
        s,e = iters[i]
        angular_prior = labels[i]
        data_array = load_detections('dataset/'+folder_name_only)
        saved_p, saved_v, saved_w, saved_iter,saved_landing_seq = computer_trajectory(data_array,camera_param_list,graph_minimum_size,angular_prior,s,e)

        offset = 1
        if len(saved_landing_seq)>0:
            landing_ind = saved_landing_seq[0]
            if landing_ind + offset < len(saved_p):
                v1.append(saved_v[landing_ind - offset,:])
                w1.append(saved_w[landing_ind - offset,:])
                v2.append(saved_v[landing_ind + offset,:])
                w2.append(saved_w[landing_ind + offset,:])
            # else:
            #     v1.append(saved_v[landing_ind - 2,:])
            #     w1.append(saved_w[landing_ind - 2,:])
            #     v2.append(saved_v[landing_ind + 2,:])
            #     w2.append(saved_w[landing_ind + 2,:])

    save_list_to_csv(f'dataset/{folder_name_only}/check_bounce.csv',v1,w1,v2,w2)
    print(f'successfuly saved to dataset/{folder_name_only}/check_bounce.csv')


def run_all():
    data_dir =Path('dataset')
    for folder_name in data_dir.iterdir():
        if folder_name.is_dir():
            print(f'processing {folder_name}')
            folder_name_only = str(folder_name).split('/')[-1]
            run_prediction(folder_name_only)

def run_one(folder_name_only):
    run_prediction(folder_name_only)

def show_results():
    data_dir =Path('dataset')
    file_paths = data_dir.glob('**/check_bounce.csv')

    fig = plt.figure()
    ax = fig.add_subplot()

    pz = [];qz = [];pm=[];qm = []
    for file_path in file_paths:
        data = read_csv(file_path)
        v2x = data[:,6]
        v1x = data[:,0]
        w1y = data[:,4]

        p = v2x - 0.6448 *v1x
        q = w1y
   
        if int(str(file_path).split('/')[-2].split('_')[-1])>10:
            pm.append(p); qm.append(q)
        else:
            pz.append(p); qz.append(q)

    pm = np.concatenate(pm); qm = np.concatenate(qm)
    pz = np.concatenate(pz); qz = np.concatenate(qz)

    idn = ~((pm>1.8)&(qm<20))
    pm = pm[idn]
    qm = qm[idn]

    idn = ~((pz>1.8)&(qz<20))
    pz= pz[idn]
    qz = qz[idn]

    ax.scatter(pm,qm,label='Matthew')
    ax.scatter(pz,qz,label='Zulfiqar')
    ax.legend()
    plt.show()

def show_results_for_each_folder():
    data_dir =Path('dataset')
    file_paths = data_dir.glob('**/check_bounce.csv')
    fig = plt.figure()
    ax = fig.add_subplot()
    pz = [];qz = [];pm=[];qm = []
    for file_path in file_paths:
        data = read_csv(file_path)
        v2x = data[:,6]
        v1x = data[:,0]
        w1y = data[:,4]
        p = v2x - 0.6448 *v1x
        q = w1y
        ax.scatter(p,q,label=str(file_path).split('/')[-2])
    ax.legend()
    plt.show()
if __name__ == '__main__':
    # run_all()
    show_results()
    # show_results_for_each_folder()