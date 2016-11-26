from gym import spaces
import numpy as np
from rllab.envs.gym_env import convert_gym_space

def convert_to_gym(space):
    return spaces.Box(np.array(space[0]), np.array(space[1]) )

class Specification:
    timestep_limit = None
    def __init__(self, timestep_limit):
        self.timestep_limit = timestep_limit

def gymify_env(env):
    env.action_space = convert_to_gym(env.action_space)
    env.observation_space = convert_to_gym(env.observation_space)

    env.spec = Specification(env.timestep_limit)
    env.spec.action_space = env.action_space
    env.spec.observation_space = env.observation_space

    return env

def rllabify_env(env):
    env = gymify_env(env)
    print(env.action_space)
    env.action_space = convert_gym_space(env.action_space)
    env.observation_space = convert_gym_space(env.observation_space)
    env.horizon = env.timestep_limit
    return env
