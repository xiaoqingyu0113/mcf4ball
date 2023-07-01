import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import glob
from collections import deque
from PIL import Image, ImageDraw
import cv2

from test_makedata import make_data

from mcf4ball.estimator import IsamSolver
from mcf4ball.predictor import predict_trajectory

from mcf4ball.draw_util import  set_axes_equal,comet
from mcf4ball.camera import  load_params, triangulation

CURRENT_DIR = os.path.dirname(__file__)

def load_data():
    with open('from_bag_1/detections.csv', mode='r') as file:
        reader = csv.reader(file)
        data_list = [row for row in reader]
    
    for i,data in enumerate(data_list):
        if len(data)!=5:
            print(i)
            print(len(data))
    data_array = np.array(data_list)
    return data_array

def load_img():
    jpg_files = glob.glob('from_bag_1/*.jpg')
    jpg_files.sort()
    return jpg_files

def get_iter_from_img(jpg_file):
    sp = jpg_file.split('_')
    return int(sp[-1][:-4])

def find_closest_value(target, values):
    closest_value = min(values, key=lambda x: abs(x - target))
    return closest_value

def main():
    data_array = load_data()
    jpg_files = load_img()
    jpg_iters = [get_iter_from_img(f) for f in jpg_files]


    for data in data_array:
        iter = int(data[0])
        camera_id = int(data[2])
        u = int(float(data[3]));v = int(float(data[4]))
        if iter > 1350:
            break
        if camera_id == 2:
            image_iter = find_closest_value(iter,jpg_iters)
            image_path = f'from_bag_1/img_{image_iter:06d}.jpg'
            image = cv2.imread(image_path)
            cv2.circle(image, (u, v), 10, (0, 0, 255), 2)  # Draws a red circle with radius 10 and thickness 2
            # Display the image (optional)
            cv2.imshow('Image', image)
            cv2.waitKey(1000)  # Adjust the delay between frames (1ms here)
            # Save the frame to the video
            # video_writer.write(image)

    # Release the video writer
    # video_writer.release()
if __name__ == '__main__':
    main()