from osim.env import L2M2019CtrlEnv
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
from joblib import Parallel, delayed

import sys
import numpy as np

trial_name = 'trial_190501_L2M2019CtrlEnv_2D_2_'

n_pop = 16
n_par = 2
        
def f_ind(n_gen, i_worker, params, best_total_reward):
    flag_model = '2D'
    flag_ctrl_mode = '2D' # use 2D
    seed = None
    difficulty = 0
    sim_dt = 0.01
    sim_t = 20
    timstep_limit = int(round(sim_t/sim_dt))

    locoCtrl = OsimReflexCtrl(mode=flag_ctrl_mode, dt=sim_dt)
    env = L2M2019CtrlEnv(locoCtrl=locoCtrl, seed=seed, difficulty=difficulty, visualize=False)
    env.change_model(model=flag_model, difficulty=difficulty, seed=seed)

    try:
        observation = env.reset(project=True, seed=seed)
    except Exception as e_msg:
        print("simulation error!!!")
        print(e_msg)
        #import pdb; pdb.set_trace()
        return 0
    env.spec.timestep_limit = timstep_limit+100

    total_reward = 0
    error_sim = 0;
    t = 0
    for i in range(timstep_limit+100):
        t += sim_dt

        observation, reward, done, info = env.step(params, project=True)
        total_reward += reward

        if done:
            break

    print('    par#={} gen#={}: score={} time={}sec #step={}'.format(i_worker, n_gen, total_reward, t, env.footstep['n']))

    if best_total_reward < total_reward:
        filename = "./data/cma/" + trial_name + "best_w" + str(i_worker).zfill(2) + ".txt"
        print("")
        print("----")
        print("update the best score!!!!")
        print("\tprev = %.8f" % best_total_reward)
        print("\tcurr = %.8f" % total_reward)
        print("\tsave to [%s]" % filename)
        print("----")
        print("")
        best_total_reward = total_reward
        np.savetxt(filename, params)

    return -total_reward  # minimization


class CMATrainPar(object):
    def __init__(self, ):
        self.n_gen = 0
        self.best_reward_par = -np.inf*np.ones(n_pop)

    def f(self, v_params):
        self.n_gen += 1
        v_cost = Parallel(n_jobs=n_par)(delayed(f_ind)(self.n_gen, i, p, self.best_reward_par[i]) for i, p in enumerate(v_params))
        return v_cost

params = np.ones(37)
#params = np.loadtxt('./data/cma/trial_181029_walk_3D_noStand_8_best.txt')

if __name__ == '__main__':
    prob = CMATrainPar()

    from cmaes.solver_cma import CMASolverPar
    solver = CMASolverPar(prob)

    solver.options.set("popsize", n_pop) # 8 = 4 + floor(3*log(37))
    solver.options.set("maxiter", 400)
    solver.options.set("verb_filenameprefix", 'data/cma/' + trial_name)
    solver.set_verbose(True)

    x0 = params
    sigma = .01

    res = solver.solve(x0, sigma)
    print(res)
