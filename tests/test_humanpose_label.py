from pathlib import Path
import bisect
import csv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import racketpose
from racketpose.alphapose.demo_api import SingleImageAlphaPose
from racketpose.labeling.util import pose_detect,draw_pose,Settings,draw_human_keypoints_plt



def get_image_iters(folder_name):
    dataset_path  =  Path('dataset')/folder_name
    img_path = sorted(dataset_path.glob('cam2*.jpg'))
    return [int(p.stem[-6:]) for p in img_path]

def get_human_pose_iters(iters, num, sequence_size = 8, skip_iter = 5):
    just_before_iter_index = bisect.bisect_left(iters, num)
    return [iters[i] for i in range(just_before_iter_index - sequence_size*skip_iter, just_before_iter_index,skip_iter)]

def get_image_from_iters(folder_name, iters):
    return [Path(f'dataset/{folder_name}/cam2_{iter:06d}.jpg') for iter in iters]

def filt_poses(poses):
    for idx,rst in enumerate(poses['result']):
        kpts = rst['keypoints'].numpy()
        head_pts = kpts[17]
        left_feet = kpts[20]
        # print(np.linalg.norm(head_pts - left_feet))
        if np.linalg.norm(head_pts - left_feet)<100:
            del poses['result'][idx]


def run_and_show_in_image(folder_name ):
    settings = Settings()
    pose_detector = SingleImageAlphaPose(settings, settings.get_cfg())
    iters = get_image_iters(folder_name)
    
    
    with open(Path(f'dataset/{folder_name}/d_sperate_id.csv'),'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        separate_indices = np.array(list(reader),dtype=int)
    
    begin_iter = separate_indices[0][2]
    human_pose_iters = get_human_pose_iters(iters, begin_iter,sequence_size = 8, skip_iter= 10)
    img_names = get_image_from_iters(folder_name, human_pose_iters)

    image_vis = image = Image.open(img_names[-1])
    for img_name in img_names:
        image = Image.open(img_name)        
        pose = pose_detect(image,pose_detector)
        filt_poses(pose)
        image_vis =  draw_pose(image_vis,pose)
    image_vis.show()

def run_and_save(folder_name,seq_size = 24, skip_iter = 3):
    settings = Settings()
    pose_detector = SingleImageAlphaPose(settings, settings.get_cfg())
    iters = get_image_iters(folder_name)
    
    
    with open(Path(f'dataset/{folder_name}/d_sperate_id.csv'),'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        separate_indices = np.array(list(reader),dtype=int)
    
    saved_poses = []
    offset = 0
    for separate_ind in separate_indices:
        begin_iter = separate_ind[2] + offset
        human_pose_iters = get_human_pose_iters(iters, begin_iter,sequence_size = seq_size, skip_iter= skip_iter)
        img_names = get_image_from_iters(folder_name, human_pose_iters)

        curr_poses = []
        for img_name in img_names:
            image = Image.open(img_name)        
            pose = pose_detect(image,pose_detector)
            filt_poses(pose)
            curr_poses.append(pose['result'][0]['keypoints'].numpy())
        curr_poses = np.concatenate(curr_poses)
        saved_poses.append(curr_poses)
    saved_poses = np.concatenate(saved_poses)

    with open(Path(f'dataset/{folder_name}/d_human_poses.csv'),'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(saved_poses)
    print("poses written to " + f'dataset/{folder_name}/d_human_poses.csv')
    print(f'total trajectories = {len(separate_indices)}')

def show_detections(folder_name,seq_size = 24):
    with open(Path(f'dataset/{folder_name}/d_sperate_id.csv'),'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        separate_indices = np.array(list(reader),dtype=int)
    with open(Path(f'dataset/{folder_name}/d_human_poses.csv'),'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        human_poses = np.array(list(reader),dtype=float)

    N = len(separate_indices)
    
    human_poses = human_poses.reshape(N,seq_size,26,2) # traj num, seq size, key pts, uv
    human_poses = human_poses - human_poses[:,:,19,None,:] # centerize

    fig, axs = plt.subplots(4, 5, figsize=(15, 10))
    axs = axs.flatten()
    for traj_idx in range(20):
        ax = axs[traj_idx]
        
        if traj_idx < N:
            # draw skeleton
            for ind, ps in enumerate(human_poses[traj_idx]):
                draw_human_keypoints_plt(ax,ps,al = 0.0 + 1.0*ind/(seq_size-1))
            
            # draw stroke
            for st_idx in range(1,seq_size):
                ax.plot(human_poses[traj_idx,[st_idx-1, st_idx],10,0],human_poses[traj_idx,[st_idx-1,st_idx],10,1],color = 'purple', linewidth=4, alpha=1.0*st_idx/(seq_size-1))
            
            ax.text(0.5, -0.1, str(traj_idx), ha='center', va='center', transform=ax.transAxes)

        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(f'results/{folder_name}_poses.png')

if __name__ == '__main__':
    
    for i in range(1,11):
        folder_name = 'tennis_' + str(i)
        seq_size = 8
        # run_and_save(folder_name,seq_size=seq_size,skip_iter=3)
        show_detections(folder_name,seq_size=seq_size)