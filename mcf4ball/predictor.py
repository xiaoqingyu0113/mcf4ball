import numpy as np
import time

from mcf4ball.dynamics.ball_dynamics_with_spin_aero_friction import dynamic_forward,dynamic_jacobian
from mcf4ball.camera import axis_equal

from mcf4ball.dynamics import ball_bouncing as bounce
from mcf4ball.dynamics import ball_dynamics_with_spin_aero_friction as aero


def rk4_nextStep(fun,t0,y0,tf):
    y0 = np.array(y0)
    h = tf-t0
    k1 = h * fun(y0)[0]
    k2 = h * fun(y0+k1/2)[0]
    k3 = h * fun(y0+k2/2)[0]
    k4 = h * fun(y0+k3)[0]
    k = (k1+2*k2+2*k3+k4)/6

    yn = y0 + k
    return yn

def predict_trajectory(p0,v0,w0,total_time=10,z0=0,Cd=0.55,Le=1.5,verbose=True):
    if verbose:
        t_sim_walltime = -time.time()
    N_steps = int(total_time*100)
    time_ticks= np.linspace(0,total_time,N_steps)

    x0 = np.concatenate((p0,v0,w0))

    x0.dtype = np.float64
    xN = []
    xN.append(x0)
    rotmN = []
    for i in range(1,N_steps):
        x0 = rk4_nextStep(lambda x: aero.dynamic_forward(x,[Cd,Le]),time_ticks[i-1],x0,time_ticks[i])
        if (x0[2] < z0) and (x0[5]<0):
            x0[3:] = bounce.dynamic_forward(x0[3:],0.79)
        xN.append(x0)

    xN = np.array(xN)
    if verbose:
        t_sim_walltime += time.time()
        print('total simulation time = ', t_sim_walltime, ' sec')
    return time_ticks,xN


