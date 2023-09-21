
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
from test_check_pred_on_image import load_img,get_iter_from_img
from test_Zulfiar_Kalman import computer_trajectory, find_closest
import json

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
                color= 'y'
            elif (s in l_pts) or (e in l_pts):
                color = 'y'
            else:
                color = 'y'
            ax.plot([uvs[s][0], uvs[e][0]],[uvs[s][1], uvs[e][1]], color=color, linewidth=2)
        
        # label
        # cx,cy,cw,ch = rst['bbox']
        # plt.text(cx+cw//3, cy+ch,  rst['pose_label'], fontsize=12, color='y')



def project2image(saved_p,camera_param):
    pred_uv = camera_param.proj2img(saved_p)
    pred_uv_onimage = []
    for uv in pred_uv:
        if (0<=uv[0]<1280) and (0<=uv[1]<1024):
            pred_uv_onimage.append(uv)
    pred_uv_onimage = np.array(pred_uv_onimage)
    return pred_uv_onimage

def draw_prediction_on_image(ax,p0,v0,w0,camera_param,label,color):
    _,xN = predict_trajectory(p0,v0,w0,
                                total_time=2.0,
                                z0=param.ground_z0,
                                ez=param.ez,
                                exy=param.exy,
                                verbose=False)
    uv = camera_param.proj2img(xN[:,:3])
    ax.plot(uv[:,0],uv[:,1],label=label,color=color)


def run(folder_name_only, traj):
    dataset = CustomDataset('dataset', max_seq_size = 100, seq_size = 100)
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_param_list = convert2camParam(raw_params)
    camera_param = camera_param_list[1] # camera 2

    graph_minimum_size = 20

    dataset_dict = dataset.dataset_dict
    iters = dataset_dict[folder_name_only]['iters']
    s,e = iters[traj]
    print(s,e)

    # print(iters)
    labels = dataset_dict[folder_name_only]['labels']

    data_array = load_detections('dataset/'+folder_name_only)
    saved_p, saved_v, saved_w, saved_iter,saved_landing_seq = computer_trajectory(data_array,camera_param_list,graph_minimum_size,labels[traj],s,e)
    pred_uv_onimage = project2image(saved_p,camera_param)


    # current state:
    jpg_iter = 44646 #44598
    jpg_name = f'dataset/{folder_name_only}/cam2_{jpg_iter:06d}.jpg'
    if not os.path.exists(jpg_name):
        raise
    im = plt.imread(f'dataset/{folder_name_only}/cam2_{jpg_iter:06d}.jpg')
    data_array[data_array[:,2]!=2,0] = 0
    ind = find_closest(jpg_iter,data_array[:,0])
    curr_iter, timestamp, camid,ub,vb = data_array[ind,:]

    curr_index_in_p = find_closest(curr_iter,saved_iter)
    p0 = saved_p[curr_index_in_p,:]
    v0 = saved_v[curr_index_in_p,:]
    w0 = saved_w[curr_index_in_p,:]
    # _,xN = predict_trajectory(p0,v0,w0,
    #                             total_time=2.0,
    #                             z0=param.ground_z0,
    #                             ez=param.ez,
    #                             exy=param.exy,
    #                             verbose=False)
    # enable pose
    with open(f'dataset/{folder_name_only}/cam2_{jpg_iter:06d}.json','r') as file:
        pose = json.load(file)

    fig,ax = plt.subplots()
    draw_human_keypoints(ax,pose)

    draw_prediction_on_image(ax,p0,v0,w0,camera_param,'prediction with spin','#09ad3b')
    draw_prediction_on_image(ax,p0,v0,np.zeros(3),camera_param,'prediction with no spin','#0381bc')

    ax.plot(pred_uv_onimage[curr_index_in_p:,0],pred_uv_onimage[curr_index_in_p:,1],label='ground truth',color='orange')

    im = np.clip(im*2.0,0,255).astype(np.uint8)
   
    ax.imshow(im)
    ax.scatter(ub,vb,s=20,c='r',zorder=5,label='tennis ball')

    ax.set_xlim([0,1280])
    ax.set_ylim([1080,0])
    # Remove x and y axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0.1, y=0.1)  # Adjust the margins as needed

# Remove the spines
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)
        ax.legend()
    fig.savefig('prediction_from_humanpose.png',dpi=400,bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    folder_only = 'tennis_9'
    traj = 8
    run(folder_only,traj)

    