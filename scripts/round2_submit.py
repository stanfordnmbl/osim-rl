#!/usr/bin/env python

import opensim as osim

from osim.redis.client import Client
from osim.env import *
import numpy as np
import argparse
import os

"""
Please ensure that `visualize=False`, else there might be unexpected errors in your submission
"""
env = RunEnv(visualize=False)
client = Client()

# Create environment
observation = client.env_create()

"""
The grader runs N simulations of at most 1000 steps each. We stop after the last one
A new simulation start when `clinet.env_step` returns `done==True`
and all the simulatiosn end when the subsequent `client.env_reset()` returns a False
"""
while True:
    _action = env.action_space.sample().tolist()
    [observation, reward, done, info] = client.env_step(_action)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
