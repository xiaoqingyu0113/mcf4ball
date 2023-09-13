import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import glob
from test_helper import result_parser, load_detections

'''
plot detection of each image, and save as mp4
'''
def check_yolo_in_all_camera(folder_name):


    #load data
    detections = load_detections(folder_name)

    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    ball_point_art = []
    for ax in axes:
        b, = ax.plot([], [], 'r',marker='.', markersize=5,label='ball')
        ball_point_art.append(b)
        ax.set_xlim([0,1280])
        ax.set_ylim([1024,0])
    speed_up = 1

    def update(frame):
        frame = frame *speed_up
        if frame %100 ==0:
            print(f"frame = {frame}/{len(detections)} ({frame/len(detections)*100:.2f}%)")
        u,v = detections[frame,3:5]
        camera_id = int(detections[frame,2])-1
        ball_point_art[camera_id].set_data([u],[v])
        
        return tuple(ball_point_art)

    ani = animation.FuncAnimation(fig, update, frames=len(detections)//speed_up, blit=True,interval=1)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=300, metadata=dict(artist='Me'), bitrate=1800)

    ani.save('results/'+folder_name.split('/')[-1]+'_yolocheck2.mp4', writer=writer)

if __name__ == '__main__':
    # folders = glob.glob('dataset/tennis_*')
    # for folder_name in folders:
    #     print('processing ' + folder_name)
    #     check_yolo_in_all_camera(folder_name)

    folder_name = 'dataset/tennis_12'
    check_yolo_in_all_camera(folder_name)