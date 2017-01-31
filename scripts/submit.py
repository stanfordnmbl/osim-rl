import opensim as osim
from osim.http.client import Client
from osim.env import *
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
import numpy as np
import argparse

# Settings
CROWDAI_TOKEN = "518ec33d7af656bddfcb83ab614ba079"
remote_base = 'http://54.154.84.135:80'

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--model', dest='model', action='store', default="example_actor.h5f")
args = parser.parse_args()

env = GaitEnv(visualize=False)

nb_actions = env.action_space.shape[0]

# Load the acton
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
actor.load_weights(args.model)

client = Client(remote_base)

# Create environment
env_id = "Gait"
instance_id = client.env_create(env_id, CROWDAI_TOKEN)

# Run a single step
client.env_monitor_start(instance_id, directory='tmp', force=True)
observation = client.env_reset(instance_id)
for i in range(500):
    v = np.array(observation).reshape((-1,1,env.observation_space.shape[0]))
    [observation, reward, done, info] = client.env_step(instance_id, actor.predict(v)[0].tolist(), True)
    if done:
        break

client.env_monitor_close(instance_id)
client.env_close(instance_id)

