import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import json
import glob
import yaml
from mcf4ball.camera import CameraParam
from mcf4ball.predictor import predict_trajectory
import mcf4ball.parameters as param
import os


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

def load_img(folder_name):
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

def draw_human_keypoints(ax,pose):

    # https://github.com/Fang-Haoshu/Halpe-FullBody
    # skipped some points at feet
    links = [(10,8), (8,6), (6,18),(18,5), (5,7), (7,9),
             (4,2),(2,0),(0,1),(1,3),(17,18),
             (18,19),
             (19,12),(12,14),(14,16),(16,21),
             (19,11),(11,13),(13,15),(15,20)]
    
    pts = []
    for two in links:
        for one in two:
            if one not in pts:
                pts.append(one)
    r_pts = [8,14,16]
    l_pts = [7,13,15]


    # draw keypoints
    point_size = 2
    for rst in pose['result']:
        uvs = rst['keypoints']
        for pt in pts:
            x, y = uvs[pt]
            # ax.scatter(x , y, 2, color='r')

        # Link the points
        for s,e in links:
            if (s in r_pts) or (e in r_pts):
                color= '#FF5733'
            elif (s in l_pts) or (e in l_pts):
                color = '#FF5733'
            else:
                color = '#FF5733'
            ax.plot([uvs[s][0], uvs[e][0]],[uvs[s][1], uvs[e][1]], color=color, linewidth=2)
        
        # label
        # cx,cy,cw,ch = rst['bbox']
        # plt.text(cx+cw//3, cy+ch,  rst['pose_label'], fontsize=12, color='y')

def save_as_image_video(folder_name):
    # LOAD camera
    camera_id = 1
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)
    camera_param = camera_param_list[camera_id]

    # load image
    jpg_files = load_img(folder_name)
    jpg_iters = [get_iter_from_img(f) for f in jpg_files]

    # load result
    with open(folder_name+'/d_results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    
    saved_iter = data[:,0];saved_p = data[:,1:4];saved_v = data[:,4:7];saved_w = data[:,7:10]

    # backproj
    est_uv = camera_param.proj2img(saved_p)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    est_point, = ax.plot([], [], color = 'orange', marker='o', markersize=2,label='est')
    pred_line, = ax.plot([], [], 'orange', lw=2,label='pred')
    ball_piont, = ax.plot([], [], 'r',marker='.', markersize=10,label='ball')

    start_id =  [0]

    
    def update(frame):
        frame = 8000+frame *2
        print('frame = ', frame)
        ax.clear()
        ax.axis('off')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        ax.set_xlim([0,1280])
        ax.set_ylim([1024,0])
        image = plt.imread(jpg_files[frame])
        ax.imshow(image)

        curr_iter = jpg_iters[frame]
        rst_idx = find_closest_value_index(curr_iter,saved_iter)

        if np.linalg.norm(est_uv[rst_idx] - est_uv[rst_idx-1]) > 200:
            start_id.append(rst_idx)
        start_idx = start_id[-1]

        ax.plot(est_uv[start_idx:rst_idx,0], est_uv[start_idx:rst_idx,1], 'orange', marker='.', markersize=1,label='est')

        
        p0 = saved_p[rst_idx,:];v0 = saved_v[rst_idx,:];w0 = saved_w[rst_idx,:]
        if -v0[0] < 2:
            w0 = np.zeros(3)  
        _,xN = predict_trajectory(p0,v0,w0,
                                  total_time=2.0,
                                  z0=param.ground_z0,
                                  ez=param.ez,
                                  exy=param.exy,
                                  verbose=False)
        
        _,xN_nospin = predict_trajectory(p0,v0,np.zeros(3),
                                         total_time=2.0,
                                         z0=param.ground_z0,
                                         ez=param.ez,
                                         exy=param.exy,
                                         verbose=False)

        pred_uv = camera_param.proj2img(xN[:,:3])
        pred_uv_onimage = []
        for uv in pred_uv:
            if (0<=uv[0]<1280) and (0<=uv[1]<1024):
                pred_uv_onimage.append(uv)
        pred_uv_onimage = np.array(pred_uv_onimage)

        pred_uv_nospin = camera_param.proj2img(xN_nospin[:,:3])
        pred_uv_onimage_nospin = []
        for uv in pred_uv_nospin:
            if (0<=uv[0]<1280) and (0<=uv[1]<1024):
                pred_uv_onimage_nospin.append(uv)
        pred_uv_onimage_nospin = np.array(pred_uv_onimage_nospin)

        if len(pred_uv_onimage)>0:
            ax.plot(pred_uv_onimage[:,0], pred_uv_onimage[:,1], 'green', lw=1,label='spin')
            ax.plot(pred_uv_onimage_nospin[:,0], pred_uv_onimage_nospin[:,1], 'yellow', lw=1,label='no spin')
        else:
            ax.plot([], [], 'orange', lw=2,label='pred')
        ax.legend()

        # enable pose
        label_name = jpg_files[frame][:-3] + 'json'
        if os.path.exists(label_name):
            with open(label_name,'r') as file:
                pose = json.load(file)
            draw_human_keypoints(ax,pose)

        return est_point, pred_line,ball_piont,

    ani = animation.FuncAnimation(fig, update, frames=4000, blit=True,interval=2)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    ani.save('results/'+folder_name.split('/')[-1]+'.mp4', writer=writer)

if __name__ == '__main__':
    folder_name = 'dataset/tennis_9'
    save_as_image_video(folder_name)