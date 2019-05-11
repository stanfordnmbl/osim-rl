from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
from joblib import Parallel, delayed

import sys
import numpy as np

trial_name = 'trial_190510_L2M2019CtrlEnv_2D_d0_'

params = np.ones(37)
#params = np.loadtxt('./optim_data/cma/trial_181029_walk_3D_noStand_8_best.txt')
N_POP = 16 # 8 = 4 + floor(3*log(37))
N_PROC = 2
TIMEOUT = 10*60

init_pose = np.array([1.5, .9, 10*np.pi/180, # forward speed, pelvis height, trunk lean
        -3*np.pi/180, -30*np.pi/180, -10*np.pi/180, 10*np.pi/180, # [right] hip abduct, hip extend, knee extend, ankle extend
        -3*np.pi/180, 5*np.pi/180, -40*np.pi/180, -0*np.pi/180]) # [left] hip abduct, hip extend, knee extend, ankle extend
        
def f_ind(n_gen, i_worker, params):
    flag_model = '2D'
    flag_ctrl_mode = '2D' # use 2D
    seed = None
    difficulty = 0
    sim_dt = 0.01
    sim_t = 20
    timstep_limit = int(round(sim_t/sim_dt))

    init_error = True
    error_count = 0
    while init_error:
        try:
            locoCtrl = OsimReflexCtrl(mode=flag_ctrl_mode, dt=sim_dt)
            env = L2M2019Env(seed=seed, difficulty=difficulty, visualize=False)
            env.change_model(model=flag_model, difficulty=difficulty, seed=seed)
            obs_dict = env.reset(project=True, seed=seed, init_pose=init_pose, obs_as_dict=True)
            init_error = False
        except Exception as e_msg:
            error_count += 1
            print('\ninitialization error (x{})!!!'.format(error_count))
            #print(e_msg)
            #import pdb; pdb.set_trace()
    env.spec.timestep_limit = timstep_limit+100

    total_reward = 0
    error_sim = 0;
    t = 0
    while True:
        t += sim_dt

        locoCtrl.set_control_params(params)
        action = locoCtrl.update(obs_dict)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=True)
        total_reward += reward

        if done:
            break

    print('\n    gen#={} sim#={}: score={} time={}sec #step={}'.format(n_gen, i_worker, total_reward, t, env.footstep['n']))

    return total_reward  # minimization


class CMATrainPar(object):
    def __init__(self, ):
        self.n_gen = 0
        self.best_total_reward = -np.inf

    def f(self, v_params):
        self.n_gen += 1
        timeout_error = True
        error_count = 0
        while timeout_error:
            try:
                v_total_reward = Parallel(n_jobs=N_PROC, timeout=TIMEOUT)\
                (delayed(f_ind)(self.n_gen, i, p) for i, p in enumerate(v_params))
                timeout_error = False
            except Exception as e_msg:
                error_count += 1
                print('\ntimeout error (x{})!!!'.format(error_count))
                #print(e_msg)

        for total_reward in v_total_reward:
            if self.best_total_reward  < total_reward:
                filename = "./optim_data/cma/" + trial_name + "best_w.txt"
                print("\n")
                print("----")
                print("update the best score!!!!")
                print("\tprev = %.8f" % self.best_total_reward )
                print("\tcurr = %.8f" % total_reward)
                print("\tsave to [%s]" % filename)
                print("----")
                print("")
                self.best_total_reward  = total_reward
                np.savetxt(filename, params)

        return [-r for r in v_total_reward]

if __name__ == '__main__':
    prob = CMATrainPar()

    from cmaes.solver_cma import CMASolverPar
    solver = CMASolverPar(prob)

    solver.options.set("popsize", N_POP)
    solver.options.set("maxiter", 400)
    solver.options.set("verb_filenameprefix", 'optim_data/cma/' + trial_name)
    solver.set_verbose(True)

    x0 = params
    sigma = .01

    res = solver.solve(x0, sigma)
    print(res)
