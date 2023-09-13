
import csv
import numpy as np
import yaml

import matplotlib.pyplot as plt

from mcf4ball.predictor import predict_trajectory

from mcf4ball.draw_util import  set_axes_equal,comet,draw_tennis_court, plot_spheres
from mcf4ball.camera import  CameraParam

def convert2camParam(params):
    camera_params = []
    for p in params:
        K = np.array(p['camera_matrix']['data']).reshape(3,3)
        R = np.array(p['R_cam_world']).reshape(3,3)
        t = np.array(p['t_world_cam'])
        camera_params.append(CameraParam(K,R,t))
    return camera_params

def load_detections(folder_name):
    with open(folder_name+'/detections.csv', mode='r') as file:
        reader = csv.reader(file)
        data_list = []
        for row in reader:
            if len(row) == 5:
                data_list.append(row)
    data_array = np.array(data_list).astype(float)
    return data_array

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def save_list_to_csv(file_path,*data_list):
    data_array = []
    for l in data_list:
        l =  np.array(l)
        if len(l.shape) == 1: # 1d array, change to column vector
            l = l[:,None]
        data_array.append(l)
    saved_localization = np.concatenate(data_array,axis=1)
    write_rows_to_csv(file_path,saved_localization)

def write_rows_to_csv(file_path,data):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def read_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    return data

def result_parser(file_name):
    # saved_p, saved_v, saved_w,
    # saved_w0, saved_iter, saved_gid,
    # saved_time, saved_detection_id
    data = read_csv(file_name)
    saved_p = data[:,:3]; saved_v = data[:,3:6]; saved_w = data[:,6:9]
    saved_w0 = saved_w = data[:,9:12]; saved_iter = data[:,12].astype(int); saved_gid = data[:,13].astype(int)
    saved_time = data[:,14]; saved_detection_id = data[:,15].astype(int)
    print('the dimension fo the result is ',data.shape)
    return saved_p,saved_v,saved_w,saved_w0,saved_iter,saved_gid,saved_time,saved_detection_id

def plot_separated_results(folder_name,save_fig=True):
    saved_p,saved_v,saved_w,saved_w0,saved_iter,saved_gid,saved_time,saved_detection_id = result_parser(folder_name+'/d_results.csv')
    separate_ind = read_csv(folder_name+'/d_separate_ind.csv').astype(int)
    result_ind = separate_ind[:,:2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i, j in result_ind:
        ax.plot(saved_p[np.arange(i,j),0], 
                saved_p[np.arange(i,j),1],
                saved_p[np.arange(i,j),2],
                '-', markerfacecolor='black', markersize=3)
    ax.set_title('uses cameras 1-6')
    set_axes_equal(ax)
    draw_tennis_court(ax)
    ax.set_xlabel('x (m)',fontsize=12);ax.set_ylabel('y (m)',fontsize=12);ax.set_zlabel('z (m)',fontsize=12)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(axis='both', labelsize=12)
    # Set pane colors to white (this will remove the gray background)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    if save_fig:
        fig.savefig(f'results/{folder_name.split("/")[-1]}_trajectories.png')
    else:
        plt.show()


def plot_whole_results(folder_name):
    saved_p,saved_v,saved_w,saved_w0,saved_iter,saved_gid,saved_time,saved_detection_id = result_parser(folder_name+'/d_results.csv')


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(saved_p[:,0], saved_p[:,1],saved_p[:,2],s=1, c='b')
    ax.set_title('uses cameras 1-6')
    set_axes_equal(ax)
    draw_tennis_court(ax)
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    plt.show()