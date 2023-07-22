
import csv
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

from mcf4ball.estimator import IsamSolver
from mcf4ball.predictor import predict_trajectory

from mcf4ball.draw_util import  set_axes_equal,comet,draw_tennis_court, plot_spheres
from mcf4ball.camera import  CameraParam
from gtsam.symbol_shorthand import X,L,V,W

CURRENT_DIR = os.path.dirname(__file__)
folder_name = 'tennis_3'


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


def run():
    data_array = load_data()
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)

    saved_p = [];saved_v = [];saved_w = [];saved_w0 = [];saved_iter = []; saved_gid = [];saved_time = []

    graph_minimum_size = 20

    angular_prior = np.array([-5,-16,-10])*6.28


    gtsam_solver = IsamSolver(camera_param_list,verbose=False,graph_minimum_size=graph_minimum_size,ez=0.7,exy=0.7,angular_prior=angular_prior)

    for d in data_array:
        iter = int(d[0])
        data = []
        data.append(float(d[1]))
        data.append(int(d[2])-1)
        data.append(float(d[3]))
        data.append(float(d[4]))


        print(f"\niter = {iter}")
        if iter > 5000:
            break

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


    saved_p = np.array(saved_p)
    saved_v = np.array(saved_v)
    saved_w = np.array(saved_w)
    saved_w0 = np.array(saved_w0)
    saved_iter = np.array(saved_iter)[:,None]
    saved_gid = np.array(saved_gid)[:,None]
    saved_time = np.array(saved_time)[:,None]

    saved_data = np.concatenate((saved_iter,saved_p,saved_v,saved_w,saved_w0,saved_gid,saved_time),axis=1)


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


def plot_results():
    # Open the CSV file in read mode
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)

    saved_iter = data[:,0];saved_p = data[:,1:4];saved_v = data[:,4:7]
    saved_w = data[:,7:10]; saved_w0 = data[:,10:13]
    saved_gid = data[:,13]; saved_time = data[:,14]
    traj_seperator = seperate_traj(saved_gid)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    minimum_traj_size = 400
    if len(traj_seperator) == 0:
        if len(saved_p) > minimum_traj_size:
            ax.plot(saved_p[:,0], saved_p[:,1],saved_p[:,2], '.', markerfacecolor='black', markersize=3)
    else:
        if traj_seperator[0]> minimum_traj_size:
            ax.plot(saved_p[:traj_seperator[0],0], saved_p[:traj_seperator[0],1],saved_p[:traj_seperator[0],2], '.', markerfacecolor='black', markersize=3)
        if len(saved_p) - traj_seperator[-1] > minimum_traj_size:
            ax.plot(saved_p[traj_seperator[-1]:,0], saved_p[traj_seperator[-1]:,1],saved_p[traj_seperator[-1]:,2], '.', markerfacecolor='black', markersize=3)
        if len(traj_seperator)>1:
            for i in range(len(traj_seperator)-1):
                if traj_seperator[i+1] - traj_seperator[i] > minimum_traj_size:
                    ax.plot(saved_p[traj_seperator[i]:traj_seperator[i+1],0], 
                            saved_p[traj_seperator[i]:traj_seperator[i+1],1],
                            saved_p[traj_seperator[i]:traj_seperator[i+1],2],
                            '.', markerfacecolor='black', markersize=3)


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
        axs[1].plot(saved_w0[:,i]/(2*np.pi),'o',label=labels[i]) # convert to herz

    axs[0].set_ylabel('linear velocity (m/s)')    
    axs[1].set_ylabel('angular velocity (Hz)')    

    axs[0].legend() 
    axs[1].legend() 

    fig.savefig('velocities.jpg')

def save_as_video():
    # Open the CSV file in read mode
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    saved_iter = data[:,0];saved_p = data[:,1:4];saved_v = data[:,4:7];saved_w = data[:,7:10]
    comet(saved_p[::3],saved_v[::3],saved_w[::3],predict_trajectory)

if __name__ == '__main__':
    run()
    plot_results()
    save_as_video()