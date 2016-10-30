import opensim as osim
import numpy as np
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam

import numpy as np

from rl.agents import ContinuousDQNAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from environment import Environment

import argparse
import math

# Some meta parameters
nallsteps = 10000000

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--output', dest='output', action='store', default="model.h5f")
args = parser.parse_args()

env = Environment(args.visualize)

# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + (env.ninput, )))
V_model.add(Dense(200))
V_model.add(Activation('relu'))
V_model.add(Dense(200))
V_model.add(Activation('relu'))
#V_model.add(Dense(16))
#V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + (env.ninput, )))
mu_model.add(Dense(200))
mu_model.add(Activation('relu'))
mu_model.add(Dense(200))
#mu_model.add(Activation('relu'))
#mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(env.noutput))
#mu_model.add(Activation('linear'))
mu_model.add(Activation('sigmoid'))
print(mu_model.summary())

action_input = Input(shape=(env.noutput,), name='action_input')
observation_input = Input(shape=(1,) + (env.ninput, ), name='observation_input')
x = merge([action_input, Flatten()(observation_input)], mode='concat')
x = Dense(400)(x)
x = Activation('relu')(x)
x = Dense(400)(x)
x = Activation('relu')(x)
# x = Dense(32)(x)
# x = Activation('relu')(x)
x = Dense(((env.noutput * env.noutput + env.noutput) / 2))(x)
x = Activation('linear')(x)
L_model = Model(input=[action_input, observation_input], output=x)
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=1, mu=0., sigma=.05, size=env.noutput)
agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
                           memory=memory, nb_steps_warmup=1000, random_process=random_process,
                           gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=True, verbose=1, nb_max_episode_steps=env.nepisodesteps)
    # After training is done, we save the final weights.
    agent.save_weights(args.output, overwrite=True)

if not args.train:
    agent.load_weights(args.output)
    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)


