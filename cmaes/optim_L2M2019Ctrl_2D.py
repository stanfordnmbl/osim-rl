from osim.env import L2M2019CtrlEnv
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl

import sys
import numpy as np

trial_name = 'trial_190501_L2M2019CtrlEnv_2D_'

class CMATrain(object):
    def __init__(self, ):
        self.n_f_call = 0
        self.best_total_reward = -np.inf

        self.flag_model = '2D'
        self.flag_ctrl_mode = '2D' # use 2D
        self.seed = None
        self.difficulty = 0
        self.sim_dt = 0.01
        self.sim_t = 20
        self.timstep_limit = int(round(self.sim_t/self.sim_dt))

        self.locoCtrl = OsimReflexCtrl(mode=self.flag_ctrl_mode, dt=self.sim_dt)
        self.env = L2M2019CtrlEnv(locoCtrl=self.locoCtrl, seed=self.seed, difficulty=self.difficulty, visualize=False)
        self.env.change_model(model=self.flag_model, difficulty=self.difficulty, seed=self.seed)
        
    def f(self, params):
        self.n_f_call += 1

        try:
            observation = self.env.reset(project=True, seed=self.seed)
        except Exception as e_msg:
            print("simulation error!!!")
            print(e_msg)
            #import pdb; pdb.set_trace()
            return 0
        self.env.spec.timestep_limit = self.timstep_limit+100

        total_reward = 0
        error_sim = 0;
        t = 0
        for i in range(self.timstep_limit+100):
            t += self.sim_dt

            observation, reward, done, info = self.env.step(params, project=True)
            total_reward += reward

            if done:
                break

        print('    sim #{}: score={} time={}sec #step={}'.format(self.n_f_call, total_reward, t, self.env.footstep['n']))

        if self.best_total_reward < total_reward:
            filename = "./data/cma/" + trial_name + "best.txt"
            print("")
            print("----")
            print("update the best score!!!!")
            print("\tprev = %.8f" % self.best_total_reward)
            print("\tcurr = %.8f" % total_reward)
            print("\tsave to [%s]" % filename)
            print("----")
            print("")
            self.best_total_reward = total_reward
            np.savetxt(filename, params)

        return -total_reward  # minimization

params = np.ones(37)
#params = np.loadtxt('./data/cma/trial_181029_walk_3D_noStand_8_best.txt')

if __name__ == '__main__':
    prob = CMATrain()

    from cmaes.solver_cma import CMASolver
    solver = CMASolver(prob)

    solver.options.set("popsize", 16) # 8 = 4 + floor(3*log(37))
    solver.options.set("maxiter", 400)
    solver.options.set("verb_filenameprefix", 'data/cma/' + trial_name)
    solver.set_verbose(True)

    x0 = params
    sigma = .01

    res = solver.solve(x0, sigma)
    print(res)
