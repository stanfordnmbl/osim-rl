from osim.env import L2M2019Env
import numpy as np

env = L2M2019Env(seed=5, difficulty=2)
env.change_model(model='3D', difficulty=2, seed=11)
observation = env.reset(project=False)
for i in range(300):
    observation, reward, done, info = env.step(env.action_space.sample(), project = True)
    obs_dict = env.get_observation_dict()
    if done:
        break
