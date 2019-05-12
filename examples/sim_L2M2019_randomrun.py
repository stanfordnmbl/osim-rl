from osim.env import L2M2019Env
import numpy as np

mode = '3D'
difficulty = 2
seed = None
project = True
obs_as_dict = True

env = L2M2019Env(seed=seed, difficulty=difficulty)
env.change_model(model=mode, difficulty=difficulty, seed=seed)
obs_dict = env.reset(project=project, seed=seed, obs_as_dict=obs_as_dict)

while True:
    obs_dict, reward, done, info=env.step(env.action_space.sample(), project=project, obs_as_dict=obs_as_dict)
    if done:
        break
