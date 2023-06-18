import numba
from numpy import *
@numba.jit(cache=True,nopython=True)
def dynamic_forward(_Dummy_22, _Dummy_23):
    [p_1, p_2, p_3, v_1, v_2, v_3, w_1, w_2, w_3] = _Dummy_22
    [Cd, Le] = _Dummy_23
    return array([[v_1, v_2, v_3, -0.0373221207246467*Cd*v_1*sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*(-v_2*w_3 + v_3*w_2)*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10), -0.0373221207246467*Cd*v_2*sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*(v_1*w_3 - v_3*w_1)*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10), -0.0373221207246467*Cd*v_3*sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*(-v_1*w_2 + v_2*w_1)*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) - 9.81, 0, 0, 0]])

@numba.jit(cache=True,nopython=True)
def dynamic_jacobian(_Dummy_24, _Dummy_25):
    [p_1, p_2, p_3, v_1, v_2, v_3, w_1, w_2, w_3] = _Dummy_24
    [Cd, Le] = _Dummy_25
    return array([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, -0.0373221207246467*Cd*v_1**2/sqrt(v_1**2 + v_2**2 + v_3**2) - 0.0373221207246467*Cd*sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*v_1*(-v_2*w_3 + v_3*w_2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(v_1**2 + v_2**2 + v_3**2)) - 0.00123162998391334*Le*(-v_2*w_3 + v_3*w_2)*(w_2*(v_1*w_2 - v_2*w_1) + w_3*(v_1*w_3 - v_3*w_1))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), -0.0373221207246467*Cd*v_1*v_2/sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*v_2*(-v_2*w_3 + v_3*w_2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(v_1**2 + v_2**2 + v_3**2)) - 0.00123162998391334*Le*w_3*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) - 0.00123162998391334*Le*(-v_2*w_3 + v_3*w_2)*(-w_1*(v_1*w_2 - v_2*w_1) + w_3*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), -0.0373221207246467*Cd*v_1*v_3/sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*v_3*(-v_2*w_3 + v_3*w_2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(v_1**2 + v_2**2 + v_3**2)) + 0.00123162998391334*Le*w_2*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) - 0.00123162998391334*Le*(-v_2*w_3 + v_3*w_2)*(-w_1*(v_1*w_3 - v_3*w_1) - w_2*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), 0.00123162998391334*Le*w_1*(-v_2*w_3 + v_3*w_2)*sqrt(v_1**2 + v_2**2 + v_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(w_1**2 + w_2**2 + w_3**2)) - 0.00123162998391334*Le*(-v_2*w_3 + v_3*w_2)*(-v_2*(v_1*w_2 - v_2*w_1) - v_3*(v_1*w_3 - v_3*w_1))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), 0.00123162998391334*Le*v_3*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) + 0.00123162998391334*Le*w_2*(-v_2*w_3 + v_3*w_2)*sqrt(v_1**2 + v_2**2 + v_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(w_1**2 + w_2**2 + w_3**2)) - 0.00123162998391334*Le*(v_1*(v_1*w_2 - v_2*w_1) - v_3*(v_2*w_3 - v_3*w_2))*(-v_2*w_3 + v_3*w_2)*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), -0.00123162998391334*Le*v_2*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) + 0.00123162998391334*Le*w_3*(-v_2*w_3 + v_3*w_2)*sqrt(v_1**2 + v_2**2 + v_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(w_1**2 + w_2**2 + w_3**2)) - 0.00123162998391334*Le*(v_1*(v_1*w_3 - v_3*w_1) + v_2*(v_2*w_3 - v_3*w_2))*(-v_2*w_3 + v_3*w_2)*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2))], [0, 0, 0, -0.0373221207246467*Cd*v_1*v_2/sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*v_1*(v_1*w_3 - v_3*w_1)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(v_1**2 + v_2**2 + v_3**2)) + 0.00123162998391334*Le*w_3*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) - 0.00123162998391334*Le*(v_1*w_3 - v_3*w_1)*(w_2*(v_1*w_2 - v_2*w_1) + w_3*(v_1*w_3 - v_3*w_1))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), -0.0373221207246467*Cd*v_2**2/sqrt(v_1**2 + v_2**2 + v_3**2) - 0.0373221207246467*Cd*sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*v_2*(v_1*w_3 - v_3*w_1)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(v_1**2 + v_2**2 + v_3**2)) - 0.00123162998391334*Le*(v_1*w_3 - v_3*w_1)*(-w_1*(v_1*w_2 - v_2*w_1) + w_3*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), -0.0373221207246467*Cd*v_2*v_3/sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*v_3*(v_1*w_3 - v_3*w_1)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(v_1**2 + v_2**2 + v_3**2)) - 0.00123162998391334*Le*w_1*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) - 0.00123162998391334*Le*(v_1*w_3 - v_3*w_1)*(-w_1*(v_1*w_3 - v_3*w_1) - w_2*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), -0.00123162998391334*Le*v_3*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) + 0.00123162998391334*Le*w_1*(v_1*w_3 - v_3*w_1)*sqrt(v_1**2 + v_2**2 + v_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(w_1**2 + w_2**2 + w_3**2)) - 0.00123162998391334*Le*(v_1*w_3 - v_3*w_1)*(-v_2*(v_1*w_2 - v_2*w_1) - v_3*(v_1*w_3 - v_3*w_1))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), 0.00123162998391334*Le*w_2*(v_1*w_3 - v_3*w_1)*sqrt(v_1**2 + v_2**2 + v_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(w_1**2 + w_2**2 + w_3**2)) - 0.00123162998391334*Le*(v_1*w_3 - v_3*w_1)*(v_1*(v_1*w_2 - v_2*w_1) - v_3*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), 0.00123162998391334*Le*v_1*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) + 0.00123162998391334*Le*w_3*(v_1*w_3 - v_3*w_1)*sqrt(v_1**2 + v_2**2 + v_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(w_1**2 + w_2**2 + w_3**2)) - 0.00123162998391334*Le*(v_1*w_3 - v_3*w_1)*(v_1*(v_1*w_3 - v_3*w_1) + v_2*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2))], [0, 0, 0, -0.0373221207246467*Cd*v_1*v_3/sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*v_1*(-v_1*w_2 + v_2*w_1)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(v_1**2 + v_2**2 + v_3**2)) - 0.00123162998391334*Le*w_2*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) - 0.00123162998391334*Le*(-v_1*w_2 + v_2*w_1)*(w_2*(v_1*w_2 - v_2*w_1) + w_3*(v_1*w_3 - v_3*w_1))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), -0.0373221207246467*Cd*v_2*v_3/sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*v_2*(-v_1*w_2 + v_2*w_1)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(v_1**2 + v_2**2 + v_3**2)) + 0.00123162998391334*Le*w_1*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) - 0.00123162998391334*Le*(-v_1*w_2 + v_2*w_1)*(-w_1*(v_1*w_2 - v_2*w_1) + w_3*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), -0.0373221207246467*Cd*v_3**2/sqrt(v_1**2 + v_2**2 + v_3**2) - 0.0373221207246467*Cd*sqrt(v_1**2 + v_2**2 + v_3**2) + 0.00123162998391334*Le*v_3*(-v_1*w_2 + v_2*w_1)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(v_1**2 + v_2**2 + v_3**2)) - 0.00123162998391334*Le*(-v_1*w_2 + v_2*w_1)*(-w_1*(v_1*w_3 - v_3*w_1) - w_2*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), 0.00123162998391334*Le*v_2*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) + 0.00123162998391334*Le*w_1*(-v_1*w_2 + v_2*w_1)*sqrt(v_1**2 + v_2**2 + v_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(w_1**2 + w_2**2 + w_3**2)) - 0.00123162998391334*Le*(-v_1*w_2 + v_2*w_1)*(-v_2*(v_1*w_2 - v_2*w_1) - v_3*(v_1*w_3 - v_3*w_1))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), -0.00123162998391334*Le*v_1*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/(sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10) + 0.00123162998391334*Le*w_2*(-v_1*w_2 + v_2*w_1)*sqrt(v_1**2 + v_2**2 + v_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(w_1**2 + w_2**2 + w_3**2)) - 0.00123162998391334*Le*(-v_1*w_2 + v_2*w_1)*(v_1*(v_1*w_2 - v_2*w_1) - v_3*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2)), 0.00123162998391334*Le*w_3*(-v_1*w_2 + v_2*w_1)*sqrt(v_1**2 + v_2**2 + v_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)*sqrt(w_1**2 + w_2**2 + w_3**2)) - 0.00123162998391334*Le*(-v_1*w_2 + v_2*w_1)*(v_1*(v_1*w_3 - v_3*w_1) + v_2*(v_2*w_3 - v_3*w_2))*sqrt(v_1**2 + v_2**2 + v_3**2)*sqrt(w_1**2 + w_2**2 + w_3**2)/((sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2) + 1.0e-10)**2*sqrt((v_1*w_2 - v_2*w_1)**2 + (v_1*w_3 - v_3*w_1)**2 + (v_2*w_3 - v_3*w_2)**2))], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
