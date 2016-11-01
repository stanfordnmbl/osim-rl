import opensim as osim
import math
import numpy as np
from gym import spaces
import os

class Specification:
    timestep_limit = 500

def angular_dist(t,s):
    t = t % math.pi
    s0 = -2*math.pi + (s % (2*math.pi))
    s1 = s % (2*math.pi)
    s2 = 2*math.pi + s % (2*math.pi)
    return min(abs(t - s0),abs(t - s1),abs(t-s2))

class ArmEnv:
    # Initialize simulation
    model = None
    manager = None
    state = None
    state0 = None

    stepsize = 0.01
    ninput = 8
    noutput = 6
    integration_accuracy = 1e-3
    timestep_limit = 500

    istep = 0

    joints = []

    spec = Specification()

    # OpenAI Gym compatibility
    action_space = spaces.Box(0.0, 1.0, shape=(noutput,) )
    observation_space = spaces.Box(-100000.0, 100000.0, shape=(ninput,) )

    def compute_reward(self):
        obs = self.get_observation()
#        print (obs[0], obs[1])
        return (2*(math.pi**2) - angular_dist(obs[0],math.pi)**2 - angular_dist(obs[1],0.0)**2)/(2*math.pi**2)

    def is_done(self):
        return False

    def __init__(self, visualize = True):
#        setattr(self, '', 200)
        # Get the model

        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, '../../models/Arm26_Optimize.osim')
        self.model = osim.Model(filename)

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        # Get the muscles
        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()

        self.noutput = self.muscleSet.getSize()

        # Print bodies
        print("Bodies")
        for i in xrange(self.bodySet.getSize()):
            print(self.bodySet.get(i).getName())

        print("Joints")
        for i in xrange(self.jointSet.getSize()):
            print(self.jointSet.get(i).getName())

        self.joints.append(osim.CustomJoint.safeDownCast(self.jointSet.get(0)))
        self.joints.append(osim.CustomJoint.safeDownCast(self.jointSet.get(1)))

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

    def get_observation(self):
        invars = np.array([0] * self.ninput, dtype='f')

        invars[0] = self.joints[0].getCoordinate(0).getValue(self.state)
        invars[1] = self.joints[1].getCoordinate(0).getValue(self.state)

        pos = self.model.calcMassCenterPosition(self.state)
        vel = self.model.calcMassCenterVelocity(self.state)
        
        invars[2] = pos[0]
        invars[3] = pos[1]
        invars[4] = pos[2]

        invars[5] = vel[0]
        invars[6] = vel[1]
        invars[7] = vel[2]

        return invars

    def step(self, action):
        for j in range(self.noutput):
            muscle = self.muscleSet.get(j)
            muscle.setActivation(self.state, action[j] * 1.0)

        # Integrate one step
        self.manager.setInitialTime(self.stepsize * self.istep)
        self.manager.setFinalTime(self.stepsize * (self.istep + 1))

        try:
            self.manager.integrate(self.state, self.integration_accuracy)
        except Exception:
            print (self.get_observation())
            print (action)
            return self.get_observation(), -500, True, {}
            

        self.istep = self.istep + 1

        return self.get_observation(), self.compute_reward(), self.is_done(), {}

    def render(self, *args, **kwargs):
        return

