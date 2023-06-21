import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

def axis_equal(ax,X,Y,Z):
   # Set the limits of the axes to be equal

    x = np.array(X).flatten()
    y = np.array(Y).flatten()
    z = np.array(Z).flatten()

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def axis_bgc_white(ax):
    ax.set_facecolor('white')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(color='black')

# def comet(x,y,z):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     set_axes_equal(ax)
#     axis_bgc_white(ax)
#     line, = ax.plot([], [], [], 'b', lw=2)
#     point, = ax.plot([], [], [], 'ro')
#     def init():
#         line.set_data([], [])
#         line.set_3d_properties([])
#         point.set_data([], [])
#         point.set_3d_properties([])
#         return line, point,
#     def update(frame):
#         line.set_data(x[:frame], y[:frame])
#         line.set_3d_properties(z[:frame])
#         point.set_data(x[frame], y[frame])
#         point.set_3d_properties(z[frame])
#         return line, point,
#     ani = animation.FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True,interval=50)
#     plt.show()

# def comet_with_rot(x,y,z,wx,wy,wz,save=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     set_axes_equal(ax)
#     axis_bgc_white(ax)
#     line, = ax.plot([], [], [], 'b', lw=2)
#     point, = ax.plot([], [], [], 'ro')

#     R = [np.eye(3)]  # Turn R into a list
#     w_, = ax.plot([], [], [], 'g', lw=2)


#     def init():
#         line.set_data([], [])
#         line.set_3d_properties([])
#         point.set_data([], [])
#         point.set_3d_properties([])
#         w_.set_data([], [])
#         w_.set_3d_properties([])

#         return line, point,w_
#     def update(frame):
#         scale = 0.2
#         line.set_data(x[:frame], y[:frame])
#         line.set_3d_properties(z[:frame])
#         point.set_data(x[frame], y[frame])
#         point.set_3d_properties(z[frame])
#         w_.set_data([x[frame],x[frame] + wx[frame]*scale], [y[frame],y[frame]+ wy[frame]*scale])
#         w_.set_3d_properties([z[frame],z[frame]+wz[frame]*scale])

#         return line, point,w_
#     ani = animation.FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True,interval=0.01)
#     # if save is not None:
#     #     ani.save(save, writer='pillow')
#     plt.show()


def comet(saved_p, saved_v, saved_w,predict_trajectory):
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