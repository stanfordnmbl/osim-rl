import opensim as osim
import math
import numpy as np
from gym import spaces
import os

class Specification:
    timestep_limit = 500

def angular_dist(t,s):
    x = (t-s) % (2*math.pi)
    return min(x, 2*math.pi-x)

class ArmEnv:
    # Initialize simulation
    model = None
    manager = None
    state = None
    state0 = None

    stepsize = 0.01
    ninput = 12
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
        up = (2*(math.pi**2) - angular_dist(obs[0],math.pi)**2 - angular_dist(obs[1],0.0)**2)/(2*math.pi**2)
        still = (obs[2]**2 + obs[3]**2) / 400
#        print still
        return up - still

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

    def sanitify(self, x):
        if math.isnan(x):
            return 0.0
        BOUND = 1000.0
        if x > BOUND:
            x = BOUND
        if x < -BOUND:
            x = -BOUND
        return x

    def get_observation(self):
        invars = np.array([0] * self.ninput, dtype='f')

        invars[0] = self.joints[0].getCoordinate(0).getValue(self.state)
        invars[1] = self.joints[1].getCoordinate(0).getValue(self.state)

        invars[2] = self.joints[0].getCoordinate(0).getSpeedValue(self.state)
        invars[3] = self.joints[1].getCoordinate(0).getSpeedValue(self.state)

        invars[4] = self.sanitify(self.joints[0].getCoordinate(0).getAccelerationValue(self.state))
        invars[5] = self.sanitify(self.joints[1].getCoordinate(0).getAccelerationValue(self.state))

        pos = self.model.calcMassCenterPosition(self.state)
        vel = self.model.calcMassCenterVelocity(self.state)
        
        invars[6] = pos[0]
        invars[7] = pos[1]
        invars[8] = pos[2]

        invars[9] = vel[0]
        invars[10] = vel[1]
        invars[11] = vel[2]

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

