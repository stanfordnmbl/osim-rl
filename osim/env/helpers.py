from gym import spaces
from rllab.envs.gym_env import convert_gym_space
import numpy as np

def convert_to_gym(space):
    spaces.Box(np.array(space[0]), np.array(space[1]) )

def gymify_env(env):
    env.action_space = convert_to_gym(env.action_space)
    env.observation_space = convert_to_gym(env.observation_space)
    return env

def rllabify_env(env):
    env = gymify_env(env)
    env.action_space = convert_gym_space(env.action_space)
    env.observation_space = convert_gym_space(env.action_space)
    return env
