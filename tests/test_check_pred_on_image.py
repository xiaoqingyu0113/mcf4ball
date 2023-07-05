import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import glob
import yaml
from mcf4ball.camera import CameraParam
from mcf4ball.predictor import predict_trajectory

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

def load_img():
    jpg_files = glob.glob(folder_name+'/*.jpg')
    jpg_files.sort()
    return jpg_files

def get_iter_from_img(jpg_file):
    sp = jpg_file.split('_')
    return int(sp[-1][:-4])

def find_closest_value(target, values):
    closest_value = min(values, key=lambda x: abs(x - target))
    return int(closest_value)
def find_closest_value_index(target, values):
    closest_value_index = min(range(len(values)), key=lambda i: abs(values[i] - target))
    return int(closest_value_index)



def save_as_image_video():
    # LOAD camera
    camera_id = 1
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)
    camera_param = camera_param_list[camera_id]

    # load image
    jpg_files = load_img()
    jpg_iters = [get_iter_from_img(f) for f in jpg_files]

    # load result
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    
    saved_iter = data[:,0];saved_p = data[:,1:4];saved_v = data[:,4:7];saved_w = data[:,7:]

    # backproj
    est_uv = camera_param.proj2img(saved_p)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    est_point, = ax.plot([], [], 'b', marker='o', markersize=2,label='est')
    pred_line, = ax.plot([], [], 'orange', lw=2,label='pred')
    ball_piont, = ax.plot([], [], 'r',marker='o', markersize=5,label='ball')
    
    def init():
        est_point.set_data([], [])
        pred_line.set_data([], [])
        ball_piont.set_data([], [])
        return est_point, pred_line,ball_piont,
    
    def update(frame):
        ax.clear()
        ax.axis('off')

        image = plt.imread(jpg_files[frame])
        ax.imshow(image)

        curr_iter = jpg_iters[frame]
        rst_idx = find_closest_value_index(curr_iter,saved_iter)
        offset = 0
        if rst_idx - offset < 0:
            rst_idx = 0
        else:
            rst_idx = rst_idx - offset

        print("rst_idx = ",rst_idx)
        print("curr_iter = ", curr_iter)
        print("rst_iter = ", saved_iter[rst_idx])
        ax.plot(est_uv[:rst_idx,0], est_uv[:rst_idx,1], 'b', marker='.', markersize=1,label='est')

        
        p0 = saved_p[rst_idx,:];v0 = saved_v[rst_idx,:];w0 = saved_w[rst_idx,:]
        _,xN = predict_trajectory(p0,v0,w0,total_time=2.0,z0=0)

        pred_uv = camera_param.proj2img(xN[:,:3])
        pred_uv_onimage = []
        for uv in pred_uv:
            if (0<=uv[0]<1280) and (0<=uv[1]<1080):
                pred_uv_onimage.append(uv)
        pred_uv_onimage = np.array(pred_uv_onimage)

        if len(pred_uv_onimage)>0:
            ax.plot(pred_uv_onimage[:,0], pred_uv_onimage[:,1], 'orange', lw=1,label='pred')
            # ball_piont, = ax.plot(est_uv[rst_idx,0], est_uv[rst_idx,1], 'r',marker='o', markersize=5,label='ball')
        else:
            ax.plot([], [], 'orange', lw=2,label='pred')
            # ball_piont, = ax.plot([], [], 'r',marker='o', markersize=5,label='ball')

        return est_point, pred_line,ball_piont,

    ani = animation.FuncAnimation(fig, update, frames=200, init_func=init, blit=True,interval=1)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    ani.save('image.mp4', writer=writer)

if __name__ == '__main__':
    save_as_image_video()