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


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def find_consecutive(lst):
    sequences = []
    start = lst[0]
    for i in range(1, len(lst)):
        if lst[i] - lst[i-1] != 1:
            if start != lst[i-1]:  # If sequence length is greater than 1.
                sequences.append(list(range(start, lst[i-1]+1)))
            start = lst[i]

    # For the last sequence in the list
    if start != lst[-1]:
        sequences.append(list(range(start, lst[-1]+1)))
    return sequences



def find_landing_position(saved_p):
    x,y,z = saved_p.T
    zmin = z.min()
    landing_indices = np.where(z<zmin+0.300)[0]
    # print(landing_indices)
    landing_seq = find_consecutive(landing_indices)

    landing_position = []
    for l in landing_seq:
        index = np.argmin(z[l])
        lx = x[l[index]]
        ly = y[l[index]]
        lz = z[l[index]]
        landing_position.append([lx,ly,lz])        
    landing_position = np.array(landing_position)

    return landing_position, landing_seq

def find_landing_position_from_prediction(saved_p):
    x,y,z = saved_p.T
    landing_position = []
    landing_seq = []
    for i in range(1,len(z)-1):
        if z[i-1]>z[i] and z[i+1] > z[i]:
            landing_position.append([x[i],y[i],z[i]])        
            landing_seq.append([i])
    landing_position = np.array(landing_position)
    return landing_position, landing_seq

def computer_trajectory(data_array,camera_param_list,graph_minimum_size,angular_prior,s,e):
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
                saved_p.append(p_rst)
                saved_w.append(w_rst)
                saved_v.append(v_rst)
                saved_iter.append(iter)
                
                if num_w == 0 and gtsam_solver.num_w ==1:
                    saved_landing_seq.append(localize_ind)
                if num_w ==1 and gtsam_solver.num_w ==2:
                    saved_landing_seq.append(localize_ind)
                    break
                num_w = gtsam_solver.num_w
                localize_ind += 1

    saved_p = np.array(saved_p)
    saved_v = np.array(saved_v)
    saved_w = np.array(saved_w)
    saved_iter = np.array(saved_iter)[:,None]
    return saved_p, saved_v, saved_w, saved_iter,saved_landing_seq

def compute_landing_prediction_error_over_time(saved_p,saved_v,saved_w,landing_positions, landing_seq):
    N_landing_points = len(landing_seq)
    N1 = landing_seq[0]
    if N_landing_points == 2:
        N2 = landing_seq[1]
    else:
        N2 = 0

    error1_spin = []
    error2_spin = []
    error1_nospin = []
    error2_nospin = []
    i = 0
    for p0,v0,w0 in zip(saved_p,saved_v,saved_w):
        if i< N1:
            _,xN_nospin = predict_trajectory(p0,v0,np.zeros(3),
                                                total_time=8.0,
                                                z0=param.ground_z0,
                                                ez=param.ez,
                                                Le = param.Le,
                                                exy=param.exy,
                                                Cd=param.Cd,
                                                verbose=False)
            _,xN_spin = predict_trajectory(p0,v0,w0,
                                            total_time=8.0,
                                            z0=param.ground_z0,
                                                ez=param.ez,
                                                Le = param.Le,
                                                exy=param.exy,
                                                Cd=param.Cd,
                                            verbose=False)
            
            landing_pred_nospin,_ = find_landing_position_from_prediction(xN_nospin[:,:3])
            landing_pred_spin,_ = find_landing_position_from_prediction(xN_spin[:,:3])

            error1_spin.append(np.linalg.norm(landing_pred_spin[0,:2]-landing_positions[0,:2]))
            if N_landing_points==2:
                error2_spin.append(np.linalg.norm(landing_pred_spin[1,:2]-landing_positions[1,:2]))
            
            error1_nospin.append(np.linalg.norm(landing_pred_nospin[0,:2]-landing_positions[0,:2]))
            if N_landing_points==2:
                error2_nospin.append(np.linalg.norm(landing_pred_nospin[1,:2]-landing_positions[1,:2]))

        elif N1<i and  i<N2:
            _,xN_nospin = predict_trajectory(p0,v0,np.zeros(3),
                                                total_time=8.0,
                                                z0=param.ground_z0,
                                                ez=param.ez,
                                                Le = param.Le,
                                                exy=param.exy,
                                                Cd=param.Cd,
                                                verbose=False)
            _,xN_spin = predict_trajectory(p0,v0,w0,
                                                total_time=8.0,
                                                z0=param.ground_z0,
                                                ez=param.ez,
                                                Le = param.Le,
                                                exy=param.exy,
                                                Cd=param.Cd,
                                                verbose=False)
            
            landing_pred_nospin,_ = find_landing_position_from_prediction(xN_nospin[:,:3])
            landing_pred_spin,_ = find_landing_position_from_prediction(xN_spin[:,:3])

            error2_spin.append(np.linalg.norm(landing_pred_spin[0,:2]-landing_positions[1,:2]))
            error2_nospin.append(np.linalg.norm(landing_pred_nospin[0,:2]-landing_positions[1,:2]))

        i += 1
    return error1_spin, error2_spin, error1_nospin,error2_nospin

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
    saved_p, saved_v, saved_w, saved_iter,saved_landing_seq = computer_trajectory(data_array,camera_param_list,graph_minimum_size,angular_prior,s,e)

    # plot_trajectory(saved_p,saved_p[saved_landing_seq,:])
    # raise

    error1_spin, error2_spin, error1_nospin,error2_nospin = compute_landing_prediction_error_over_time(saved_p,saved_v,saved_w,saved_p[saved_landing_seq,:], saved_landing_seq)

    plt.plot(error1_nospin[4:],'--',label='landing 1, nospin')
    plt.plot(error2_nospin[4:],'--',label='landing 2, nospin')
    plt.plot(error1_spin[4:],label='landing 1, spin')
    plt.plot(error2_spin[4:],label='landing 2, spin')

    # Show minor grid lines
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.2, color='gray')
    plt.xlabel('# of detection from camera')
    plt.ylabel('RMSE (m)')
    plt.title(f'{angular_prior}')
    # plt.ylim([0,3])
    plt.legend()
    plt.show()

def plot_trajectory(saved_p,landing_positions):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(saved_p[:,0], 
            saved_p[:,1],
            saved_p[:,2],
            '.', markerfacecolor='black', markersize=3)
    # landing_positions,landing_seq = find_landing_position(saved_p[:-2])
    # print(landing_positions)
    ax.scatter(landing_positions[:,0],
               landing_positions[:,1],
               landing_positions[:,2],
               s=40,c ='r',marker='.')
    ax.set_title('uses cameras 1-6')

    ax.view_init(19, -174)
    set_axes_equal(ax)

    draw_tennis_court(ax)
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    plt.show()

if __name__ == '__main__':
    folder_name_only = 'tennis_12'
    i = 5
    run_prediction(folder_name_only,i)