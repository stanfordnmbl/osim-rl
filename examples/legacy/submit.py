import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv
import numpy as np
import argparse

# Settings
# remote_base = 'http://grader.crowdai.org:1729' # Submission to Round-1
remote_base = 'http://grader.crowdai.org:1730' # Submission to Round-2

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--token', dest='token', action='store', required=True)
args = parser.parse_args()

client = Client(remote_base)

# Create environment
observation = client.env_create(args.token, env_id="ProstheticsEnv")
env = ProstheticsEnv()

# Run a single step
# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
while True:
    print(observation)
    [observation, reward, done, info] = client.env_step(env.action_space.sample().tolist())
    if done:
        observation = client.env_reset()
        if not observation:
            break
            
client.submit()
