import numba
from numpy import *
@numba.jit(cache=True,nopython=True)
def dynamic_forward(_Dummy_22, ez):
    [v1x, v1y, v1z, w1x, w1y, w1z] = _Dummy_22
    return array([[0.644760213143872*v1x + 0.0117229129662522*w1y, 0.644760213143872*v1y - 0.0117229129662522*w1x, -ez*v1z, -19.538188277087*v1y + 0.355239786856128*w1x, 19.538188277087*v1x + 0.355239786856128*w1y, w1z]])

@numba.jit(cache=True,nopython=True)
def dynamic_jacobian(_Dummy_23, ez):
    [v1x, v1y, v1z, w1x, w1y, w1z] = _Dummy_23
    return array([[0.644760213143872, 0, 0, 0, 0.0117229129662522, 0], [0, 0.644760213143872, 0, -0.0117229129662522, 0, 0], [0, 0, -ez, 0, 0, 0], [0, -19.538188277087, 0, 0.355239786856128, 0, 0], [19.538188277087, 0, 0, 0, 0.355239786856128, 0], [0, 0, 0, 0, 0, 1]])

