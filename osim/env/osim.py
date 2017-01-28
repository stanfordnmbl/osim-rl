import opensim
import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym

class Osim(object):
    # Initialize simulation
    model = None
    manager = None
    state = None
    state0 = None
    joints = []

    def __init__(self, model_path, visualize):
        self.model = opensim.Model(model_path)

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()

    def reset(self):
        if not self.state0:
            self.state0 = self.model.initSystem()
            self.manager = opensim.Manager(self.model)

        self.state = opensim.State(self.state0)

class Spec(object):
    def __init__(self, *args, **kwargs):
        self.id = 0
        self.timestep_limit = 500

class OsimEnv(gym.Env):
    stepsize = 0.01
    integration_accuracy = 1e-3
    timestep_limit = 500
    test = False

    action_space = None
    observation_space = None
    osim_model = None
    istep = 0

    model_path = ""
    visualize = False
    ninput = 0
    noutput = 0

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 50
    }

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['osim_model']
        print ("HERE1")
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)
        self.osim_model = Osim(self.model_path, True)
        self.configure()

    def angular_dist(self, t,s):
        x = (t-s) % (2*math.pi)
        return min(x, 2*math.pi-x)

    def compute_reward(self):
        return 0.0

    def is_done(self):
        return False

    def terminate(self):
        pass

    def __init__(self, visualize = True, noutput = None):
        self.visualize = visualize
        self.osim_model = Osim(self.model_path, self.visualize)
        
        self.noutput = noutput
        if not noutput:
            self.noutput = self.osim_model.muscleSet.getSize()
        if not self.action_space:
            self.action_space = ( [0.0] * self.noutput, [1.0] * self.noutput )
        if not self.observation_space:
            self.observation_space = ( [-math.pi] * self.ninput, [math.pi] * self.ninput )

        self.action_space = convert_to_gym(self.action_space)
        self.observation_space = convert_to_gym(self.observation_space)
        self.horizon = self.timestep_limit

        self.configure()
        self.reset()

        self.spec = Spec()

    def configure(self):
        pass

    def _reset(self):
        self.istep = 0
        self.osim_model.reset()
        return [0.0] * self.ninput

    def sanitify(self, x):
        if math.isnan(x):
            return 0.0
        BOUND = 1000.0
        if x > BOUND:
            x = BOUND
        if x < -BOUND:
            x = -BOUND
        return x

    def activate_muscles(self, action):
        muscleSet = self.osim_model.model.getMuscles()
        for j in range(self.noutput):
            muscle = muscleSet.get(j)
            muscle.setActivation(self.osim_model.state, float(action[j]))

    def _step(self, action):
        # Integrate one step
        self.osim_model.manager.setInitialTime(self.stepsize * self.istep)
        self.osim_model.manager.setFinalTime(self.stepsize * (self.istep + 1))

        # action = action[0]
        self.activate_muscles(action)

        try:
            self.osim_model.manager.integrate(self.osim_model.state)
        except Exception as e:
            print (e)
            return self.get_observation(), -500, True, {}

        self.istep = self.istep + 1

        res = [ self.get_observation(), self.compute_reward(), self.is_done(), {} ]
        return res


    def _render(self, mode='human', close=False):
        return

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass
