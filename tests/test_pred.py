
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mcf4ball.draw_util import set_axes_equal, axis_bgc_white
from mcf4ball.predictor import predict_trajectory
if __name__ == '__main__':
    mag_vel = 26.8
    ang = 30
    p0 = np.array([0,0,1])
    v0  =np.array([mag_vel*np.sin(np.radians(ang)),0,mag_vel*np.cos(np.radians(ang))])
    w0 = np.array([0,0,10])*2*np.pi


    time_ticks,xN1 = predict_trajectory(p0,v0,w0,z0=0)
    time_ticks,xN1 = predict_trajectory(p0,v0,w0,z0=0)

    # plain plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(xN1[:,0], xN1[:,1],xN1[:,2], 'g', markerfacecolor='black', markersize=1,label='top spin')

    axis_bgc_white(ax)
    ax.set_title('compare spin trajectory')
    set_axes_equal(ax)
    plt.legend()
    plt.show()