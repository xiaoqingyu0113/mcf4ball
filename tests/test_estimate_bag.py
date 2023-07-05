
import csv
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

from mcf4ball.estimator import IsamSolver
from mcf4ball.predictor import predict_trajectory

from mcf4ball.draw_util import  set_axes_equal,comet,draw_tennis_court, plot_spheres
from mcf4ball.camera import  CameraParam

CURRENT_DIR = os.path.dirname(__file__)
folder_name = 'from_bag_1'

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def convert2camParam(params):
    camera_params = []
    for p in params:
        K = np.array(p['camera_matrix']['data']).reshape(3,3)
        R = np.array(p['R_cam_world']).reshape(3,3)
        t = np.array(p['t_world_cam'])
        camera_params.append(CameraParam(K,R,t))
    return camera_params

def load_data():
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

def convert2camParam(params):
    camera_params = []
    for p in params:
        K = np.array(p['camera_matrix']['data']).reshape(3,3)
        R = np.array(p['R_cam_world']).reshape(3,3)
        t = np.array(p['t_world_cam'])
        camera_params.append(CameraParam(K,R,t))
    return camera_params


def run():
    data_array = load_data()
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)

    saved_p = [];saved_v = [];saved_w = [];saved_iter = []

    graph_minimum_size = 30
    gtsam_solver = IsamSolver(camera_param_list,verbose=True,graph_minimum_size=graph_minimum_size)

    for d in data_array:
        iter = int(d[0])
        data = []
        data.append(float(d[1]))
        data.append(int(d[2])-1)
        data.append(float(d[3]))
        data.append(float(d[4]))


        print(f"\niter = {iter}")

        if iter > 10000:
            break
        # if iter < 0:
        #     continue
        gtsam_solver.push_back(data)
        rst = gtsam_solver.get_result()
        if rst is not None:
            p_rst,v_rst,w_rst = rst
            # if np.linalg.norm(w_rst) > 1000:
            #     continue
            saved_p.append(p_rst)
            saved_w.append(w_rst)
            saved_v.append(v_rst)
            saved_iter.append(iter)

    saved_p = np.array(saved_p)
    saved_v = np.array(saved_v)
    saved_w = np.array(saved_w)
    saved_iter = np.array(saved_iter)

    saved_data = np.concatenate((saved_iter[:,None],saved_p,saved_v,saved_w),axis=1)


    with open(folder_name+'/d_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(saved_data)

def plot_results():
    # Open the CSV file in read mode
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    
    saved_iter = data[:,0];saved_p = data[:,1:4];saved_v = data[:,4:7];saved_w = data[:,7:]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(saved_p[:,0], saved_p[:,1],saved_p[:,2], 'b.', markerfacecolor='black', markersize=3)
    ax.set_title('uses cameras 1-3')
    set_axes_equal(ax)
    draw_tennis_court(ax)
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    plt.show()


    fig, axs = plt.subplots(1, 2)  # This will create a 2x2 grid of Axes objects
    labels = ['x','y','z']
    gt_color = ['r','g','b']

    for i in range(3):
        axs[0].plot(saved_v[:,i],'o',label=labels[i])
        axs[1].plot(saved_w[:,i]/(2*np.pi),'o',label=labels[i]) # convert to herz

    axs[0].set_ylabel('linear velocity (m/s)')    
    axs[1].set_ylabel('angular velocity (Hz)')    

    axs[0].legend() 
    axs[1].legend() 

    plt.show()

    # -------------------------------------- plain plot ------------------------------
    # 
def save_as_video():
    # Open the CSV file in read mode
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    saved_iter = data[:,0];saved_p = data[:,1:4];saved_v = data[:,4:7];saved_w = data[:,7:]
    comet(saved_p[::3],saved_v[::3],saved_w[::3],predict_trajectory)


if __name__ == '__main__':
    # run()
    plot_results()
    # save_as_video()