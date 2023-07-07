
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mcf4ball.draw_util import set_axes_equal, axis_bgc_white
from mcf4ball.predictor import predict_trajectory



def compute_trajectory():
    mag_vel = 26.8
    ang = 30
    p0 = np.array([0,0,1])
    v0  =np.array([mag_vel*np.sin(np.radians(ang)),0,mag_vel*np.cos(np.radians(ang))])
    w0 = np.array([0,0.0,0.1])*2*np.pi
    time_ticks,xN1 = predict_trajectory(p0,v0,w0,z0=0,Cd=0.55,Le=1.5,ez=1.0,total_time=10)
    return xN1

def show_trajectory(xN1):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(xN1[:,0], xN1[:,1],xN1[:,2], 'g', markerfacecolor='black', markersize=1,label='top spin')
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')

    axis_bgc_white(ax)
    ax.set_title('compare spin trajectory')
    set_axes_equal(ax)
    plt.legend()
    plt.show()


def show_vel(xN1):
    saved_v = xN1[:,3:6]
    saved_w = xN1[:,6:]

    fig, axs = plt.subplots(1, 2)  # This will create a 2x2 grid of Axes objects
    labels = ['x','y','z']
    gt_color = ['r','g','b']

    for i in range(3):
        axs[0].plot(saved_v[:,i],'o',label=labels[i])
        axs[1].plot(saved_w[:,i]/(2*np.pi),'o',label=labels[i]) # convert to herz

    axs[0].set_ylabel('linear velocity (m/s)')    
    axs[1].set_ylabel('angular velocity (Hz)')    

    axs[0].legend() 
    axs[1].legend() 
    plt.show()
if __name__ == '__main__':
    xN1 = compute_trajectory()

    show_trajectory(xN1)
    show_vel(xN1)
    
  