import yaml
import csv
import numpy as np
from mcf4ball.camera import CameraParam,draw_camera
from mcf4ball.draw_util import set_axes_equal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
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
        R = np.array(p['R_world_cam']).reshape(3,3)
        t = np.array(p['t_world_cam'])
        camera_params.append(CameraParam(K,R,t))
    return camera_params


def main():
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_params = convert2camParam(raw_params)

    detections = [[914,1687821197.7938619,2,624.51171875,133.4375],
                   [ 915,1687821197.8091333,5,976.46875,428.5],
                    [916,1687821197.8161385,1,1115.140625,233.0],
                    [918,1687821197.8225596,3,209.62890625,118.015625],
                    [919,1687821197.807658,2,623.85546875,136.25],
                    [920,1687821197.8195708,6,1249.171875,127.75]]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = ['r','g','b','r','g','b']
    for cn,c,cp in zip(camera_names, colors,camera_params):
        draw_camera(cp.R.T,cp.t,color = c,ax= ax)
        ax.text(cp.t[0], cp.t[1], cp.t[2], cn,color=c)
    set_axes_equal(ax)
    ax.set_xlabel('x (m)');    ax.set_ylabel('y (m)');    ax.set_zlabel('z (m)')
    plt.show()
if __name__ == '__main__':
    main()