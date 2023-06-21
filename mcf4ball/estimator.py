import numpy as np
# import time 
import gtsam
from gtsam.symbol_shorthand import X,L,V,W
from mcf4ball.factors import PositionFactor,LinearFactor, BounceLinearFactor, BounceAngularFactor
from mcf4ball.camera import triangulation
        
class IsamSolver:
    def __init__(self,camera_param_list,Cd = 0.55,Le=1.5,ez=0.79, verbose = True):

        self.camera_param_list = camera_param_list
        self.Cd = Cd
        self.Le = Le
        self.ez = ez
        self.verbose = verbose

        self.uv_noise = gtsam.noiseModel.Isotropic.Sigma(2, 3.0)  # 2 pixels error
        self.camera_calibration_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])) 
        self.pos_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001]))
        self.linear_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001]))
        self.angular_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001])*2*np.pi)

        self.reset()

    def reset(self):
        self.t_max = -np.inf
   
        self.num_optim = 0
        self.start_optim = False

        parameters = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(parameters)

        self.initial_estimate = gtsam.Values()
        self.current_estimate = None
        self.graph = gtsam.NonlinearFactorGraph()

        self.curr_node_idx = -1
        self.num_w = 0
        self.bounce_smooth_step = 10
        self.curr_bounce_idx = 0

        self.total_addgraph_time = 0
        self.total_optimize_time = 0

        self.obs_start_buffer = []

    def update(self,data, optim = False):
        if self.verbose:
            print('updating!')
        self.curr_node_idx += 1

        t, camera_id, u,v = data
        t = float(t); camera_id = int(camera_id);u = float(u);v = float(v)

        if t > self.t_max:
            # self.total_addgraph_time -= time.time()
            self.add_subgraph(data) # modify self.graph and self.intial
            # self.total_addgraph_time += time.time()
            if optim:
                if self.num_optim ==0:
                    self.obs_start_buffer.append([t, camera_id, u,v])
                    self.warm_start()

                # self.total_optimize_time -= time.time()
                self.optimize() 
                # self.total_optimize_time += time.time()
                self.clear()
                self.num_optim += 1
            else:
                self.obs_start_buffer.append([t, camera_id, u,v])

            self.t_max = t # keep at bottom

        elif self.verbose:
            print('ERROR: t < self.t_max!!!!!!!!')

    def warm_start(self):
        if self.verbose:
            print("\t- warm starting")
        t0,camera_id0,u0,v0 = self.obs_start_buffer[0]
        save_idx = []
        save_p_guess = []
        t_save = []
        for idx, (t,camera_id,u,v) in enumerate(self.obs_start_buffer):
            if camera_id != camera_id0:
                p_guess ,_= triangulation(np.array([u0,v0]),np.array([u,v]),np.array([-1,-1]),
                                                            self.camera_param_list[camera_id0],
                                            self.camera_param_list[camera_id],
                                            self.camera_param_list[camera_id])
                save_idx.append(idx)
                save_p_guess.append(p_guess)
                t_save.append(t)
            t0,camera_id0,u0,v0 = t,camera_id,u,v
        
        save_p_guess = np.array(save_p_guess)
        t_save = np.array(t_save)
        save_v_guess = np.diff(save_p_guess,axis=0)/np.diff(t_save)[:,None]

        pmean = save_p_guess.mean(0)

        vmean = np.random.rand(3)

        curr_idx = save_idx[0]
        for i in range(len(self.obs_start_buffer)):
            if i > curr_idx:
                curr_idx += 1
            if curr_idx < len(save_idx):
                # self.initial_estimate.insert(L(i),pmean + np.random.rand(3)-0.5)
                self.initial_estimate.insert(L(i),save_p_guess[curr_idx])
            else:
                # self.initial_estimate.insert(L(i),pmean+ np.random.rand(3)-0.5)
                self.initial_estimate.insert(L(i),save_p_guess[-1])
            if curr_idx < len(save_v_guess):
                self.initial_estimate.insert(V(i),np.random.rand(3)-0.5)
            else:
                self.initial_estimate.insert(V(i),np.random.rand(3)-0.5)

     
    def add_subgraph(self,data):
        t, camera_id, u,v = data
        t = float(t); camera_id = int(camera_id);u = float(u);v = float(v)

        j = self.curr_node_idx
        
        K_gtsam, pose_gtsam = self.camera_param_list[camera_id].to_gtsam()
        self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(np.array([u,v]), self.uv_noise, X(j), L(j), K_gtsam)) # add noise
        self.graph.push_back(gtsam.PriorFactorPose3(X(j), pose_gtsam, self.camera_calibration_noise)) # add prior
        if self.verbose:
            print(f"add pixel detection X({j}) -> L({j})")
            print(f"add prior X({j})")
        if j >0:
            self.graph.push_back(PositionFactor(self.pos_noise,L(j-1),L(j),V(j-1),self.t_max,t))
            if self.verbose:
                print(f"add position factor L({j-1}), L({j}) -> V({j-1})")

            if self.current_estimate is not None:
                z_prev = self.current_estimate.atVector(L(j-1))[2]
                if (z_prev < 0) and (j > self.curr_bounce_idx + self.bounce_smooth_step):
                    if self.verbose:
                        print('add bounce')
                        print(f'\t- adding bounce factor (v({j-1}), w({self.num_w}) -> v({j}))')
                        print(f'\t- adding bounce factor (v({j-1}), w({self.num_w}) -> w({self.num_w+1}))')
                    self.graph.push_back(BounceLinearFactor(self.linear_noise,V(j-1),W(self.num_w),V(j),self.ez))
                    self.graph.push_back(BounceAngularFactor(self.angular_noise,V(j-1),W(self.num_w),W(self.num_w+1),self.ez))
                    self.initial_estimate.insert(W(self.num_w+1), self.current_estimate.atVector(W(self.num_w)))
                    self.num_w += 1
                    self.curr_bounce_idx = j
                elif (z_prev < 0) and (j <= self.curr_bounce_idx + self.bounce_smooth_step):
                    if self.verbose:
                        print(f'bounce within the smooth, move on. Current bounce idx move to {self.curr_bounce_idx}')
                    self.graph.push_back(LinearFactor(self.linear_noise,V(j-1),W(self.num_w),V(j),self.t_max,t,[self.Cd, self.Le]))
                    self.curr_bounce_idx = j
                else:
                    self.graph.push_back(LinearFactor(self.linear_noise,V(j-1),W(self.num_w),V(j),self.t_max,t,[self.Cd, self.Le]))
                    if self.verbose:
                        print(f"add Linear factor V({j-1}), W({self.num_w}) -> V({j})")

            else:
                self.graph.push_back(LinearFactor(self.linear_noise,V(j-1),W(self.num_w),V(j),self.t_max,t,[self.Cd, self.Le]))
                if self.verbose:
                    print(f"add Linear factor V({j-1}), W({self.num_w}) -> V({j})")

        self.initial_estimate.insert(X(j),pose_gtsam)
        if j ==0:
            # self.graph.push_back(PriorFactor3(gtsam.noiseModel.Diagonal.Sigmas(np.array([100, 100, 100])),W(0),np.array([0,0,0])))
            self.initial_estimate.insert(W(self.num_w),np.array([0.1,0.1,0.1]))
        if self.current_estimate is None:
            # self.initial_estimate.insert(L(j),np.random.rand(3)+np.array([0,0,2]))
            # self.initial_estimate.insert(V(j),np.random.rand(3))
            pass
        else:
            self.initial_estimate.insert(L(j),self.current_estimate.atVector(L(j-1)))
            self.initial_estimate.insert(V(j),self.current_estimate.atVector(V(j-1)))

    def optimize(self):
        if self.verbose:
            print('\t- optimizing!')
        self.isam.update(self.graph, self.initial_estimate)
        if self.num_optim <10:
            for _ in range(10):
                self.isam.update() # more iteration
        else:
            for _ in range(2):
                self.isam.update() # more iteration
        self.current_estimate = self.isam.calculateEstimate()
        

    def clear(self):
        self.graph.resize(0)
        self.initial_estimate.clear()
        

    def get_result(self):
        if self.current_estimate is None:
            if self.verbose:
                print('get result: None')
            return None
        else:
            l = self.current_estimate.atVector(L(self.curr_node_idx))
            v = self.current_estimate.atVector(V(self.curr_node_idx))
            w = self.current_estimate.atVector(W(self.num_w))
            if self.verbose:
                print(f'get result: (L({self.curr_node_idx}),V({self.curr_node_idx}),W({self.num_w}))')
        return l,v,w


