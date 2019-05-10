from osim.env import L2M2019CtrlEnv
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np

mode = '2D'
difficulty = 2
seed=None
sim_dt = 0.01
sim_t = 10
timstep_limit = int(round(sim_t/sim_dt))

locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
env = L2M2019CtrlEnv(locoCtrl=locoCtrl, seed=seed, difficulty=difficulty)
env.change_model(model=mode, difficulty=difficulty, seed=seed)
observation = env.reset(project=True, seed=seed)
env.spec.timestep_limit = timstep_limit+100

if mode is '2D':
    #params = np.loadtxt('./optim_data/cma/trial_190505_L2M2019CtrlEnv_2D_d0_best_w.txt')
    xrecentbest = open("./optim_data/cma/trial_190505_L2M2019CtrlEnv_2D_d0_xrecentbest.dat", "r")
    #params = np.ones(37)

    #params_3 = np.append(params, [1, 1, 1, 1, 1, 1, 1, 1])
    #np.savetxt('params_3D.txt', params_3)
elif mode is '3D':
    #params = np.loadtxt('./optim_data/cma/trial_190505_L2M2019CtrlEnv_d0_best_w.txt')
    params = np.loadtxt('./optim_data/params_3D_init.txt')
    #xrecentbest = open("./optim_data/cma/trial_190505_L2M2019CtrlEnv_d0_xrecentbest.dat", "r")
    #params = np.ones(45)

try:
    for line in xrecentbest:
        pass
    last = np.fromstring(line, sep=' ')
    params = np.array(last[5:])
except:
    pass

# visualize v_tgt --------------------------------------------------------------
import matplotlib.pyplot as plt
flag_visualize_vtgt = 1
if flag_visualize_vtgt:
    vtgt_obj = env.vtgt.vtgt_obj
    fig,axes = plt.subplots(2,1, figsize=(4, 6))
    X = vtgt_obj.map[0]
    Y = vtgt_obj.map[1]
    U = vtgt_obj.vtgt[0]
    V = vtgt_obj.vtgt[1]
    R = np.sqrt(U**2 + V**2)
    q0 = axes[0].quiver(X, Y, U, V, R)
    axes[0].axis('equal')
# visualize v_tgt --------------------------------------------------------------

total_reward = 0
t = 0
i = 0
while True:
    i += 1
    t += sim_dt
    #import pdb; pdb.set_trace()
    #observation, reward, done, info = env.step(np.ones(locCtrl.n_par), project = True)
    observation, reward, done, info = env.step(params, project = True)
    total_reward += reward
    obs_dict = env.get_observation_dict()
    if done:
        break

# visualize v_tgt --------------------------------------------------------------
    if flag_visualize_vtgt and i%50==0:
        if env.flag_new_v_tgt_field:
            q0.remove()
            X = vtgt_obj.map[0]
            Y = vtgt_obj.map[1]
            U = vtgt_obj.vtgt[0]
            V = vtgt_obj.vtgt[1]
            R = np.sqrt(U**2 + V**2)
            q0 = axes[0].quiver(X, Y, U, V, R)
            axes[0].axis('equal')

        pose = env.pose            
        axes[0].plot(pose[0], pose[1], 'k.')
        
        X, Y = vtgt_obj._generate_grid(vtgt_obj.rng_get, vtgt_obj.res_get)
        U = env.v_tgt_field[0]
        V = env.v_tgt_field[1]
        R = np.sqrt(U**2 + V**2)
        axes[1].clear()
        axes[1].quiver(X, Y, U, V, R)
        axes[1].plot(0, 0, 'k.')
        axes[1].axis('equal')
        
        plt.pause(0.0001)
# visualize v_tgt --------------------------------------------------------------

print('    score={} time={}sec'.format(total_reward, t))

# visualize v_tgt --------------------------------------------------------------
if flag_visualize_vtgt:
    plt.show()
# visualize v_tgt --------------------------------------------------------------
