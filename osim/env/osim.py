import opensim
import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym

class Osim(object):
    # Initialize simulation
    model = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize):
        self.model = opensim.Model(model_path)
        self.model.initSystem()
        self.brain = opensim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()
        
        for j in range(self.muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.model.addController(self.brain)

    def set_strength(self, strength):
        self.curforces = strength
        for i in range(len(self.curforces)):
            self.muscleSet.get(i).setMaxIsometricForce(self.curforces[i] * self.maxforces[i])

    def get_body(self, name):
        return self.bodySet.get(name)

    def get_joint(self, name):
        return self.jointSet.get(name)

    def get_muscle(self, name):
        return self.muscleSet.get(name)

    def get_contact_geometry(self, name):
        return self.contactGeometrySet.get(name)

    def get_force(self, name):
        return self.forceSet.get(name)

    def initializeState(self):
        self.state = self.model.initializeState()

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
    last_action = None

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
#        self.reset()

        self.spec = Spec()

    def configure(self):
        pass

    def _reset(self):
        self.istep = 0
        self.osim_model.initializeState()
        return self.get_observation()

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
        if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")
        brain = opensim.PrescribedController.safeDownCast(self.osim_model.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue( float(action[j]) )

    def _step(self, action):
        self.last_action = action

        self.activate_muscles(action)

        # Integrate one step
        manager = opensim.Manager(self.osim_model.model)
        manager.setInitialTime(self.stepsize * self.istep)
        manager.setFinalTime(self.stepsize * (self.istep + 1))

        try:
            manager.integrate(self.osim_model.state)
        except Exception as e:
            print (e)
            return self.get_observation(), -500, True, {}

        self.istep = self.istep + 1

        res = [ self.get_observation(), self.compute_reward(), self.is_done(), {} ]
        return res

    def _render(self, mode='human', close=False):
        return
