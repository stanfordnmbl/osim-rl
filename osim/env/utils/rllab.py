import numpy as np
from .gym import convert_gym_space, gymify_env

def rllabify_env(env):
    env = gymify_env(env)
    print(env.action_space)
    env.action_space = convert_gym_space(env.action_space)
    env.observation_space = convert_gym_space(env.observation_space)
    env.horizon = env.timestep_limit
    return env
