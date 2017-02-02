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
parser.add_argument('--token', dest='token', action='store', default=CROWDAI_TOKEN)
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
observation = client.env_create(args.token)

# Run a single step
for i in range(501):
    v = np.array(observation).reshape((-1,1,env.observation_space.shape[0]))
    [observation, reward, done, info] = client.env_step(args.token, actor.predict(v)[0].tolist(), True)
    if done:
        break

client.submit(args.token)

