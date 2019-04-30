from osim.env import L2M2019CtrlEnv
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np

mode = '3D'

locoCtrl = OsimReflexCtrl(mode=mode)
env = L2M2019CtrlEnv(locoCtrl=locoCtrl, seed=5, difficulty=2)
env.change_model(model=mode, difficulty=2, seed=11)
observation = env.reset(project=True)
for i in range(300):
    #import pdb; pdb.set_trace()
    observation, reward, done, info = env.step(np.ones(locoCtrl.n_par), project = True)
    obs_dict = env.get_observation_dict()
    if done:
        break
