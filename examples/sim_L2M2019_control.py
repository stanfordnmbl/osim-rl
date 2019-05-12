from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np

mode = '2D'
difficulty = 2
visualize=True
seed=None
sim_dt = 0.01
sim_t = 10
timstep_limit = int(round(sim_t/sim_dt))

init_pose = np.array([1.5, .9, 10*np.pi/180, # forward speed, pelvis height, trunk lean
                -3*np.pi/180, -30*np.pi/180, -10*np.pi/180, 10*np.pi/180, # [right] hip abduct, hip extend, knee extend, ankle extend
                -3*np.pi/180, 5*np.pi/180, -40*np.pi/180, -0*np.pi/180]) # [left] hip abduct, hip extend, knee extend, ankle extend

locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
env = L2M2019Env(visualize=visualize, seed=seed, difficulty=difficulty)
env.change_model(model=mode, difficulty=difficulty, seed=seed)
obs_dict = env.reset(project=True, seed=seed, init_pose=init_pose, obs_as_dict=True)
env.spec.timestep_limit = timstep_limit

# set control parameters
if mode is '2D':
    params = np.ones(37)
    #params = np.loadtxt('./optim_data/cma/trial_190505_L2M2019CtrlEnv_2D_d0_best_w.txt')
    #xrecentbest = open("./optim_data/cma/trial_190505_L2M2019CtrlEnv_2D_d0_xrecentbest.dat", "r")
    #last = np.fromstring(line, sep=' ')
    #params = np.array(last[5:])

    #params_3 = np.append(params, [1, 1, 1, 1, 1, 1, 1, 1])
    #np.savetxt('params_3D.txt', params_3)
elif mode is '3D':
    params = np.ones(45)
    #params = np.loadtxt('./optim_data/cma/trial_190505_L2M2019CtrlEnv_d0_best_w.txt')
    #params = np.loadtxt('./optim_data/params_3D_init.txt')
    #xrecentbest = open("./optim_data/cma/trial_190505_L2M2019CtrlEnv_d0_xrecentbest.dat", "r")
    #last = np.fromstring(line, sep=' ')
    #params = np.array(last[5:])

total_reward = 0
t = 0
i = 0
while True:
    i += 1
    t += sim_dt

    # chage params
    # params = myHigherLayeyController(obs_dict)

    locoCtrl.set_control_params(params)
    action = locoCtrl.update(obs_dict)
    obs_dict, reward, done, info = env.step(action, project = True, obs_as_dict=True)
    total_reward += reward
    if done:
        break
