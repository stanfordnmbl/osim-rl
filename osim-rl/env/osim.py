import opensim as osim
import math
import numpy as np
from gym import spaces
import os
from rllab.envs.gym_env import convert_gym_space
from rllab.envs.base import Env


class Specification:
    timestep_limit = 500

class OsimEnv(object):
    # Initialize simulation
    model = None
    manager = None
    state = None
    state0 = None

    stepsize = 0.01
    integration_accuracy = 1e-3
    timestep_limit = 500

    istep = 0

    joints = []

    def angular_dist(self, t,s):
        x = (t-s) % (2*math.pi)
        return min(x, 2*math.pi-x)

    def compute_reward(self):
        return 0.0

    def is_done(self):
        return False

    def terminate(self):
        pass

    def __init__(self, visualize = True):
        # Get the model
        self.model = osim.Model(self.model_path)

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        # Get the muscles
        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.noutput = self.muscleSet.getSize()

        # OpenAI Gym compatibility
        self.action_space = convert_gym_space(spaces.Box(0.0, 1.0, shape=(self.noutput,) ))
        self.observation_space = convert_gym_space(spaces.Box(-3*math.pi, 3*math.pi, shape=(self.ninput,) ))

        self.spec = Specification()
        self.spec.action_space = self.action_space
        self.spec.observation_space = self.observation_space
        self.horizon = self.spec.timestep_limit
        
        self.reset()

    def reset(self):
        self.istep = 0
        if not self.state0:
            self.state0 = self.model.initSystem()
            self.manager = osim.Manager(self.model)
            self.state = osim.State(self.state0)
        else:
            self.state = osim.State(self.state0)

        self.model.equilibrateMuscles(self.state)

        # nullacttion = np.array([0] * self.noutput, dtype='f')
        # for i in range(0, int(math.floor(0.2 / self.stepsize) + 1)):
        #     self.step(nullacttion)

        return [0.0] * self.ninput # self.get_observation()

    def sanitify(self, x):
        if math.isnan(x):
            return 0.0
        BOUND = 1000.0
        if x > BOUND:
            x = BOUND
        if x < -BOUND:
            x = -BOUND
        return x

    def step(self, action):
        # action = action[0]
        for j in range(self.noutput):
            muscle = self.muscleSet.get(j)
            muscle.setActivation(self.state, action[j] )

        # Integrate one step
        self.manager.setInitialTime(self.stepsize * self.istep)
        self.manager.setFinalTime(self.stepsize * (self.istep + 1))

        # print (self.get_observation())
        # print (action)
        try:
            self.manager.integrate(self.state)
        except Exception as e:
            print (e)
            return self.get_observation(), -500, True, {}

        self.istep = self.istep + 1

        return self.get_observation(), self.compute_reward(), self.is_done(), {}

    def render(self, *args, **kwargs):
        return

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass
