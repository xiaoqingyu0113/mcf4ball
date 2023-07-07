import numpy as np
from collections import deque
import gtsam
from gtsam.symbol_shorthand import X,L,V,W
from mcf4ball.factors import PositionFactor,LinearFactor, BounceLinearFactor, BounceAngularFactor,PriorFactor3
from mcf4ball.camera import triangulation
        
class IsamSolver:
    def __init__(self,camera_param_list,Cd = 0.55,Le=1.5,ez=1.0, graph_minimum_size=150,ground_z0=0,verbose = True):

        self.camera_param_list = camera_param_list
        self.Cd = Cd
        self.Le = Le
        self.ez = ez
        self.graph_minimum_size = graph_minimum_size
        self.ground_z0 = ground_z0
        self.verbose = verbose

        self.bp_error = 80
        self.cam_mix_ratio = 0.4
        self.uv_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)  # 2 pixels error
        self.camera_calibration_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1e-6) 
        self.pos_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-3)
        self.linear_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-4)
        self.angular_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-4)
        self.angular_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*5)
        self.reset()

    def reset(self):
        self.t_max = -np.inf
        self.dt = None
        self.num_optim = 0
        self.start_optim = False
        self.prev_L = deque()

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

        self.obs_buffer = deque()
        self.opt_buffer = deque()
        self.optimizable = False
        self.end_optim = False

        if self.verbose:
            print('reset!')

    def add_opt_buffer(self,data):
        if len(self.opt_buffer) >= self.graph_minimum_size:
            self.opt_buffer.popleft()
        self.opt_buffer.append(data)

    def check_optimizable(self):
        prev_camera_id = int(self.opt_buffer[0][1])
        sum_change = 0
        N = len(self.opt_buffer)
        for idx in range(N):
            curr_camera_id = int(self.opt_buffer[idx][1])
            if prev_camera_id != curr_camera_id:
                sum_change += 1
        if self.verbose:
            print(f'check optimizable ({sum_change/self.graph_minimum_size:.2f})')
        if not self.optimizable and len(self.opt_buffer) == self.graph_minimum_size:
            self.optimizable =  sum_change/self.graph_minimum_size > self.cam_mix_ratio

    def check_end_optim(self):
        rst = self.get_result()
        if rst is not None:
            t, camera_id, u,v = self.opt_buffer[-1]
            camera_param = self.camera_param_list[camera_id]
            uv1 = camera_param.proj2img(rst[0]) 
            error = np.sqrt((uv1[0] - u)**2 + (uv1[1]-v)**2)
            if error > self.bp_error:
                self.end_optim = True
                
            if self.verbose:
                print(f"\t- check end optim: input:({int(u)},{int(v)}), backproj: ({int(uv1[0])},{int(uv1[1])})")
                print(f"\t- the error is {error:.2f}")

    def push_back(self,data):
        t, camera_id, u,v = data
        data = [float(t), int(camera_id), float(u),float(v)]
        if t < self.t_max:
            return
        self.dt = t - self.t_max
        self.add_opt_buffer(data)
        self.check_optimizable()
        self.check_end_optim()
        if self.end_optim:
            self.reset()
        if self.optimizable:
            self.obs_buffer.append(data)
            if (self.curr_node_idx < self.graph_minimum_size):
                self.update(data,optim=False)
            else:
                try:
                    self.update(data,optim=True)
                except:
                    self.reset()
        self.t_max = float(t) # keep at bottom

    def update(self,data, optim = False):
        if self.verbose:
            print(f'update from camera:{data[1]}!')

        self.curr_node_idx += 1
        self.add_subgraph(data) # modify self.graph and self.intial

        if optim:
            if self.num_optim ==0:
                self.warm_start()
            self.optimize() 
            self.clear()
            self.num_optim += 1


    def warm_start(self):
        '''
        initialize the results using triangulation (size + 1)
        '''
        if self.verbose:
            print("\t- warm starting")
        t0,camera_id0,u0,v0 = self.obs_buffer[0]
        save_idx = []
        save_p_guess = []
        t_save = []
        for idx, (t,camera_id,u,v) in enumerate(self.obs_buffer):
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
        curr_idx = save_idx[0]
        for i in range(len(self.obs_buffer)):
            if i > curr_idx:
                curr_idx += 1
            if curr_idx < len(save_idx):
                self.initial_estimate.insert(L(i),save_p_guess[curr_idx])
            else:
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
        if j == 0:
            self.graph.push_back(PriorFactor3(self.angular_prior_noise,W(0),np.array([0,0,20])*6.28))
        if j >0:
            self.graph.push_back(PositionFactor(self.pos_noise,L(j-1),L(j),V(j-1),self.t_max,t))
            if self.verbose:
                print(f"add position factor L({j-1}), L({j}) -> V({j-1})")

            if self.current_estimate is not None:
                z_prev = self.current_estimate.atVector(L(j-1))[2]
                if (z_prev < self.ground_z0) and (j > self.curr_bounce_idx + self.bounce_smooth_step):
                    if self.verbose:
                        print('add bounce')
                        print(f'\t- adding bounce factor (v({j-1}), w({self.num_w}) -> v({j}))')
                        print(f'\t- adding bounce factor (v({j-1}), w({self.num_w}) -> w({self.num_w+1}))')
                    self.graph.push_back(BounceLinearFactor(self.linear_noise,V(j-1),W(self.num_w),V(j),self.ez))
                    self.graph.push_back(BounceAngularFactor(self.angular_noise,V(j-1),W(self.num_w),W(self.num_w+1),self.ez))
                    self.initial_estimate.insert(W(self.num_w+1), self.current_estimate.atVector(W(self.num_w)))
                    self.num_w += 1
                    self.curr_bounce_idx = j
                elif (z_prev < self.ground_z0) and (j <= self.curr_bounce_idx + self.bounce_smooth_step):
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

        # add guess
        self.initial_estimate.insert(X(j),pose_gtsam)
        if j ==0:
            self.initial_estimate.insert(W(self.num_w),np.array([0.1,0.1,0.1]))
        if self.current_estimate is not None:
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
            for _ in range(3):
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

            if len(self.prev_L)>10:
                self.prev_L.popleft()
            self.prev_L.append(l)

            prev_l = self.prev_L[0]

            if self.verbose:
                print(f'get result: (L({self.curr_node_idx}),V({self.curr_node_idx}),W({self.num_w}))')

            if len(self.prev_L) >= 10 and np.linalg.norm(prev_l - l) < 0.500: # some filtering
                return l,v,w
            else:
                return None

