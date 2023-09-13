import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import glob
from test_helper import result_parser, load_detections


def load_img(folder_name,cam_id):
    jpg_files = glob.glob(folder_name+'/*.jpg')
    jpg_files.sort()
    filtered_jpg_files = list(filter(lambda s: cam_id in s, jpg_files))
    return filtered_jpg_files

def get_iter_from_img(jpg_file):
    sp = jpg_file.split('_')
    return int(sp[-1][:-4])

def find_closest_value(target, values):
    closest_value = min(values, key=lambda x: abs(x - target))
    return int(closest_value)

def find_closest_value_index(target, values):
    closest_value_index = min(range(len(values)), key=lambda i: abs(values[i] - target))
    return int(closest_value_index)



def save_as_image_video(folder_name):
  

    # load image
    jpg_files = load_img(folder_name,'cam2')
    jpg_iters = [get_iter_from_img(f) for f in jpg_files]
    jpg_length = len(jpg_iters)

    #load data
    detections = load_detections(folder_name)
    detections_cam2 = detections[detections[:,2].astype(int)==2,:]
    iter_list = detections_cam2[:,0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ball_points, = ax.plot([], [], 'r',marker='o', markersize=5,label='ball')
    speed_up = 2

    def update(frame):
        frame = frame *speed_up
        curr_iter = jpg_iters[frame]
        detection_ind = find_closest_value_index(curr_iter,iter_list)
        if frame %1000 ==0:
            print('frame = ', frame,'/',jpg_length)
        ax.clear()
        ax.axis('off')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        ax.set_xlim([0,1280])
        ax.set_ylim([1024,0])
        image = plt.imread(jpg_files[frame])
        ax.imshow(image)
        u,v = detections_cam2[detection_ind,3:5]
        ball_point = ax.plot([u], [v], 'r',marker='o', markersize=5,label='ball')
        curr_iter
        return ball_points,

    ani = animation.FuncAnimation(fig, update, frames=jpg_length//speed_up, blit=True,interval=2)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    ani.save('results/'+folder_name.split('/')[-1]+'_yolocheck.mp4', writer=writer)

if __name__ == '__main__':
    folders = glob.glob('dataset/tennis_*')
    for folder_name in folders:
        print('processing ' + folder_name)
        save_as_image_video(folder_name)

    # folder_name = 'dataset/tennis_14'
    # save_as_image_video(folder_name)