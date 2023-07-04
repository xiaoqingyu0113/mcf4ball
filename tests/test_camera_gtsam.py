import os
import numpy as np
import gtsam 
import yaml
from mcf4ball.camera import CameraParam,world2camera

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


def test():
    p0 = np.array([10,0,1])
    camera_names = ['22495525','22495526','22495527','23045007','23045008','23045009']
    raw_params = [read_yaml('camera_calibration_data/'+cname+'_calibration.yaml') for cname in camera_names]
    camera_params = convert2camParam(raw_params)

    camera_param = camera_params[5]
    K1_gtsam,pose1 = camera_param.to_gtsam()
    camera1_gtsam = gtsam.PinholeCameraCal3_S2(pose1, K1_gtsam)

    print(camera1_gtsam.project(p0))
    print(world2camera(p0,camera_param))

if __name__ == '__main__':
    test()