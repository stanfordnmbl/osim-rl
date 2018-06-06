import opensim as osim
from osim.http.client import Client
from osim.env import *
import numpy as np
import argparse
from pprint import pprint

# Settings
remote_base = 'http://grader.crowdai.org:1729'
remote_base = 'http://0.0.0.0:5000'

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--token', dest='token', action='store', required=True)
args = parser.parse_args()

client = Client(remote_base)

# Create environment
observation = client.env_create(args.token)
env = ProstheticsEnv()

# Run a single step
#
# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
while True:
    pprint(observation)
#    v = np.array(observation).reshape((-1,1,env.observation_space.shape[0]))
    [observation, reward, done, info] = client.env_step(env.action_space.sample().tolist())
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
