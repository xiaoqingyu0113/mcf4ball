import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time

from test_makedata import make_data

from mcf4ball.estimator import IsamSolver
from mcf4ball.draw_util import axis_bgc_white, set_axes_equal,axis_equal
from mcf4ball.predictor import predict_trajectory

def comet(saved_p, saved_v, saved_w):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    axis_equal(ax,saved_p[:,0],saved_p[:,1],saved_p[:,2])
    axis_bgc_white(ax)

    est_point, = ax.plot([], [], [], 'b', marker='o', markersize=2,label='est')
    pred_line, = ax.plot([], [], [], 'g', lw=2,label='pred')
    ball_piont, = ax.plot([], [], [], 'r',marker='o', markersize=5,label='ball')
    ax.view_init(elev=19, azim=145)

    def init():
        est_point.set_data([], [])
        est_point.set_3d_properties([])

        pred_line.set_data([], [])
        pred_line.set_3d_properties([])

        ball_piont.set_data([], [])
        ball_piont.set_3d_properties([])

        return est_point, pred_line,ball_piont,
    
    def update(frame):
        frame = frame*5
        est_point.set_data(saved_p[:frame,0], saved_p[:frame,1])
        est_point.set_3d_properties(saved_p[:frame,2])
        
        trust_steps = 150
        if frame > trust_steps:
            p0 = saved_p[frame,:];v0 = saved_v[frame,:];w0 = saved_w[frame,:]
        else:
            p0 = saved_p[frame,:];v0 = saved_v[frame,:];w0 = saved_w[frame,:]*frame/trust_steps
        _,xN = predict_trajectory(p0,v0,w0,total_time=3.0,z0=0)

        ball_piont.set_data([p0[0]], [p0[1]])
        ball_piont.set_3d_properties([p0[2]])

        pred_line.set_data(xN[:,0], xN[:,1])
        pred_line.set_3d_properties(xN[:,2])

        return est_point, pred_line,ball_piont,

    ani = animation.FuncAnimation(fig, update, frames=len(saved_p)//5, init_func=init, blit=True,interval=1)

    ani.save('animation.gif', writer='pillow')

def main():
    data_array,x_gt,camera_param_list = make_data(total_time=2.5,like_bag=True)
    print(f"make total number of {data_array.shape[0]} data")
    saved_p = []
    saved_v = []
    saved_w = []

    graph_minimum_size = 20
    gtsam_solver = IsamSolver(camera_param_list)
    total_iter = len(data_array)
    total_time = -time.time()
    for iter,data in enumerate(data_array):
        print(f"iter = {iter}")
        if iter < graph_minimum_size:
            gtsam_solver.update(data,optim=False)
        else:
            gtsam_solver.update(data,optim=True)
            rst = gtsam_solver.get_result()
            p_rst,v_rst,w_rst = rst
            if np.linalg.norm(w_rst) > 400:
                w_rst = np.zeros(3)
            saved_p.append(p_rst)
            saved_w.append(w_rst)
            saved_v.append(v_rst)

    total_time += time.time()
    print('average inference time (hz) = ', (total_iter-graph_minimum_size)/total_time)
    print('\t- average add graph time (s) = ', gtsam_solver.total_addgraph_time/(total_iter-graph_minimum_size))
    print('\t- average optim time (s) = ', gtsam_solver.total_optimize_time/(total_iter-graph_minimum_size))


    saved_p = np.array(saved_p)
    saved_v = np.array(saved_v)
    saved_w = np.array(saved_w)

    comet(saved_p[::3],saved_v[::3],saved_w[::3])

  

if __name__ == '__main__':
    main()