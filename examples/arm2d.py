import os
from osim.env import Arm2DEnv
import pprint
import numpy as np

env = Arm2DEnv()
if __name__ == '__main__':
    observation = env.reset()
    print(env.osim_model.list_elements())
    for i in range(3000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        print(len(observation))
        if done:
            env.reset()
