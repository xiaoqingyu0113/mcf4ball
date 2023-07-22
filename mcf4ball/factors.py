import gtsam
from typing import Optional, List
import numpy as np
from mcf4ball.dynamics.dynamic_errors import *

def assign_jacobians(jacobians,J):
     for i, JJ in enumerate(J):
          jacobians[i] = JJ
     

class PriorFactor3(gtsam.CustomFactor):
    def __init__(self, noiseModel, key1,mu):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            w1 = values.atVector(key1)
            error = w1 - mu
            if jacobians is not None:
                    jacobians[0] = np.eye(3)
            return error
        super().__init__(noiseModel, [key1], error_function) # may change to partial

class PositionFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, key1, key2, key3, t1, t2):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            l1, l2, v1 = values.atVector(key1), values.atVector(key2), values.atVector(key3)
            error = pos_error(l1,l2,v1,t1,t2)
            if jacobians is not None:
                    assign_jacobians(jacobians,pos_jacobian(l1,l2,v1,t1,t2))
            return error
        super().__init__(noiseModel, [key1, key2, key3], error_function) # may change to partial

class LinearFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel,key1, key2, key3, t1, t2,params,z0=0):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            v1, w1, v2 =  values.atVector(key1), values.atVector(key2), values.atVector(key3)

            error = vel_linear_error(v1,w1,v2,t1,t2,params)
            if jacobians is not None:
                assign_jacobians(jacobians,vel_linear_jacobian(v1,w1,v2,t1,t2,params))
            return error
        super().__init__(noiseModel, [key1, key2, key3], error_function) # may change to partial

class BounceLinearFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, key1, key2, key3,ez,exy):
       
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            v1, w1, v2 =  values.atVector(key1), values.atVector(key2), values.atVector(key3)
            error = bounce_vel_linear_error(v1,w1,v2,ez,exy)
            if jacobians is not None:
                    # print(bounce_vel_linear_jacobian(v1,w1,v2))
                    assign_jacobians(jacobians,bounce_vel_linear_jacobian(v1,w1,v2,ez,exy))
            return error
        
        super().__init__(noiseModel, [key1, key2, key3], error_function) # may change to partial

class BounceAngularFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, key1, key2, key3,ez,exy):
       
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            v1, w1, w2 =  values.atVector(key1), values.atVector(key2), values.atVector(key3)

            error = bounce_vel_angular_error(v1,w1,w2,ez,exy)
            if jacobians is not None:
                    assign_jacobians(jacobians,bounce_vel_angular_jacobian(v1,w1,w2,ez,exy))
            return error
        
        super().__init__(noiseModel, [key1, key2, key3], error_function) # may change to partial

class TestFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, key1, key2, key3, t1, t2):
       
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            l1,l2,v1 = values.atVector(key1), values.atVector(key2), values.atVector(key3)
            error = l2 - (l1 + v1 * (t2-t1))
            if jacobians is not None:
                jacobians[0] = -np.eye(3)
                jacobians[1] = np.eye(3)
                jacobians[2] = -np.eye(3)*(t2-t1)
            return error
        super().__init__(noiseModel, [key1, key2, key3], error_function) 