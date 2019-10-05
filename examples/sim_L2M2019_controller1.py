from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np

mode = '2D'
difficulty = 3
visualize=True
seed=None
sim_dt = 0.01
sim_t = 10
timstep_limit = int(round(sim_t/sim_dt))


INIT_POSE = np.array([
    1.699999999999999956e+00, # forward speed
    .5, # rightward speed
    9.023245653983965608e-01, # pelvis height
    2.012303881285582852e-01, # trunk lean
    0*np.pi/180, # [right] hip adduct
    -6.952390849304798115e-01, # hip flex
    -3.231075259785813891e-01, # knee extend
    1.709011708233401095e-01, # ankle flex
    0*np.pi/180, # [left] hip adduct
    -5.282323914341899296e-02, # hip flex
    -8.041966456860847323e-01, # knee extend
    -1.745329251994329478e-01]) # ankle flex

if mode is '2D':
    params = np.loadtxt('./osim/control/params_2D.txt')
elif mode is '3D':
    params = np.loadtxt('./osim/control/params_3D_init.txt')

locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
env = L2M2019Env(visualize=visualize, seed=seed, difficulty=difficulty)
env.change_model(model=mode, difficulty=difficulty, seed=seed)
obs_dict = env.reset(project=True, seed=seed, obs_as_dict=True, init_pose=INIT_POSE)
env.spec.timestep_limit = timstep_limit

total_reward = 0
t = 0
i = 0
while True:
    i += 1
    t += sim_dt

    locoCtrl.set_control_params(params)
    action = locoCtrl.update(obs_dict)
    obs_dict, reward, done, info = env.step(action, project = True, obs_as_dict=True)
    total_reward += reward
    if done:
        break
print('    score={} time={}sec'.format(total_reward, t))