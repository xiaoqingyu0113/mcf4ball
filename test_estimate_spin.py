import numpy as np
import matplotlib.pyplot as plt
import time

from test_makedata import make_data

from mcf4ball.estimator import IsamSolver
from mcf4ball.draw_util import  set_axes_equal




def main():
    data_array,x_gt,camera_param_list = make_data(total_time=2.5,like_bag=True)
    print(f"make total number of {data_array.shape[0]} data")
    saved_p = []
    saved_v = []
    saved_w = []

    graph_minimum_size = 50
    gtsam_solver = IsamSolver(camera_param_list)
    total_iter = len(data_array)
    total_time = -time.time()
    for iter,data in enumerate(data_array):
        print(f"\niter = {iter}")
        if iter < graph_minimum_size:
            gtsam_solver.update(data,optim=False)
        else:
            gtsam_solver.update(data,optim=True)
            rst = gtsam_solver.get_result()
            p_rst,v_rst,w_rst = rst
            if np.linalg.norm(w_rst) > 400:
                w_rst = np.array([0,0,0])
            saved_p.append(p_rst)
            saved_w.append(w_rst)
            saved_v.append(v_rst)

    total_time += time.time()
    print('average inference time (hz) = ', (total_iter-graph_minimum_size)/total_time)
    # print('\t- average add graph time (s) = ', gtsam_solver.total_addgraph_time/(total_iter-graph_minimum_size))
    # print('\t- average optim time (s) = ', gtsam_solver.total_optimize_time/(total_iter-graph_minimum_size))

    saved_p = np.array(saved_p)
    saved_v = np.array(saved_v)
    saved_w = np.array(saved_w)

    print(saved_p.shape)
    # -------------------------------------- plain plot ------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    ax.plot(saved_p[:,0], saved_p[:,1],saved_p[:,2], 'b.', markerfacecolor='black', markersize=3)
    ax.plot(x_gt[:,0], x_gt[:,1],x_gt[:,2], 'r.', markerfacecolor='black', markersize=1)
    ax.set_title('uses cameras 1-3')
    set_axes_equal(ax)
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    plt.show()


    fig, axs = plt.subplots(1, 2)  # This will create a 2x2 grid of Axes objects
    labels = ['x','y','z']
    gt_color = ['r','g','b']

    for i in range(3):
        axs[0].plot(saved_v[:,i],'o',label=labels[i])
        axs[0].plot(x_gt[-saved_p.shape[0]:,i+3],'-',label=labels[i],color=gt_color[i],markersize=10)

        axs[1].plot(saved_w[:,i]/(2*np.pi),'o',label=labels[i]) # convert to herz
        axs[1].plot(x_gt[-saved_p.shape[0]:,i+6]/(2*np.pi),'-',label=labels[i],color=gt_color[i])

    axs[0].set_ylabel('linear velocity (m/s)')    
    axs[1].set_ylabel('angular velocity (Hz)')    

    axs[0].legend() 
    axs[1].legend() 

    plt.show()

if __name__ == '__main__':
    main()