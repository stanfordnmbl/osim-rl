import opensim as osim
import math
import numpy as np
from gym import spaces
import os

class Specification:
    timestep_limit = 500

class GaitEnv:
    # Initialize simulation
    model = None
    manager = None
    state = None
    state0 = None

    stepsize = 0.01
    ninput = 24
    noutput = 18
    integration_accuracy = 1e-3
    nepisodesteps = 500

    istep = 0
    prev_reward = 0

    spec = Specification()

    # OpenAI Gym compatibility
    action_space = spaces.Box(0.0, 1.0, shape=(noutput,) )
    observation_space = spaces.Box(-100000.0, 100000.0, shape=(ninput,) )

    def sanitify(self, x):
        if math.isnan(x):
            return 0.0
        BOUND = 1000.0
        if x > BOUND:
            x = BOUND
        if x < -BOUND:
            x = -BOUND
        return x

    def compute_reward_standing(self):
        y = self.ground_pelvis.getCoordinate(2).getValue(self.state)
        x = self.ground_pelvis.getCoordinate(1).getValue(self.state)

        pos = self.model.calcMassCenterPosition(self.state)
        vel = self.model.calcMassCenterVelocity(self.state)
        acc = self.model.calcMassCenterAcceleration(self.state)

        rew = 100 - abs(acc[0])**2 - abs(acc[1])**2 - abs(acc[2])**2 - abs(vel[0])**2 - abs(vel[1])**2 - abs(vel[2])*2*t

        obs = self.get_observation()
        ext = 100 * sum([x**2 for x in obs]) / self.noutput
        rew = rew - ext
        
        if rew < -100:
            rew = -100
        return rew / 100.0

    def compute_reward(self):
        obs = self.get_observation()
        return self.ground_pelvis.getCoordinate(1).getValue(self.state)

    def is_head_too_low(self):
        y = self.ground_pelvis.getCoordinate(2).getValue(self.state)
        return (y < 0.5)
    
    def is_done(self):
        return self.is_head_too_low()

    def __init__(self, visualize = True):
        setattr(self, 'timestep_limit', 200)
        # Get the model

        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, '../../models/gait9dof18musc_Thelen_BigSpheres_20161017.osim')
        self.model = osim.Model(filename)

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        # Get the muscles
        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()

        # Print bodies
        for i in xrange(13):
            print(self.bodySet.get(i).getName())

        for i in xrange(13):
            print(self.jointSet.get(i).getName())

        self.ground_pelvis = osim.PlanarJoint.safeDownCast(self.jointSet.get(0))

        self.hip_r = osim.PinJoint.safeDownCast(self.jointSet.get(1))
        self.knee_r = osim.CustomJoint.safeDownCast(self.jointSet.get(2))
        self.ankle_r = osim.PinJoint.safeDownCast(self.jointSet.get(3))
        self.subtalar_r = osim.WeldJoint.safeDownCast(self.jointSet.get(4))
        self.mtp_r = osim.WeldJoint.safeDownCast(self.jointSet.get(5))

        self.hip_l = osim.PinJoint.safeDownCast(self.jointSet.get(6))
        self.knee_l = osim.CustomJoint.safeDownCast(self.jointSet.get(7))
        self.ankle_l = osim.PinJoint.safeDownCast(self.jointSet.get(8))
        self.subtalar_l = osim.WeldJoint.safeDownCast(self.jointSet.get(9))
        self.mtp_l = osim.WeldJoint.safeDownCast(self.jointSet.get(10))

        self.back = osim.PinJoint.safeDownCast(self.jointSet.get(11))
        self.back1 = osim.WeldJoint.safeDownCast(self.jointSet.get(12))

        self.head = self.bodySet.get(12)

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
        self.prev_reward = 0

        nullacttion = np.array([0] * self.noutput, dtype='f')
        for i in range(0, int(math.floor(0.2 / self.stepsize) + 1)):
            self.step(nullacttion)
 
        return [0.0] * self.ninput # self.get_observation()

    def get_observation(self):
        invars = np.array([0] * self.ninput, dtype='f')

        invars[0] = self.ground_pelvis.getCoordinate(0).getValue(self.state)
        invars[1] = self.ground_pelvis.getCoordinate(1).getValue(self.state)
        invars[2] = self.ground_pelvis.getCoordinate(2).getValue(self.state)

        invars[3] = self.hip_r.getCoordinate(0).getValue(self.state)
        invars[4] = self.hip_l.getCoordinate(0).getValue(self.state)

        invars[5] = self.ankle_r.getCoordinate(0).getValue(self.state)
        invars[6] = self.ankle_l.getCoordinate(0).getValue(self.state)

        invars[7] = self.knee_r.getCoordinate(0).getValue(self.state)
        invars[8] = self.knee_l.getCoordinate(0).getValue(self.state)
        

        invars[9] = self.ground_pelvis.getCoordinate(0).getSpeedValue(self.state)
        invars[10] = self.ground_pelvis.getCoordinate(1).getSpeedValue(self.state)
        invars[11] = self.ground_pelvis.getCoordinate(2).getSpeedValue(self.state)

        invars[12] = self.hip_r.getCoordinate(0).getSpeedValue(self.state)
        invars[13] = self.hip_l.getCoordinate(0).getSpeedValue(self.state)

        invars[14] = self.ankle_r.getCoordinate(0).getSpeedValue(self.state)
        invars[15] = self.ankle_l.getCoordinate(0).getSpeedValue(self.state)

        invars[16] = self.knee_r.getCoordinate(0).getSpeedValue(self.state)
        invars[17] = self.knee_l.getCoordinate(0).getSpeedValue(self.state)

        pos = self.model.calcMassCenterPosition(self.state)
        vel = self.model.calcMassCenterVelocity(self.state)
        
        invars[18] = pos[0]
        invars[19] = pos[1]
        invars[20] = pos[2]

        invars[21] = vel[0]
        invars[22] = vel[1]
        invars[23] = vel[2]

        for i in range(0,self.ninput):
            invars[i] = self.sanitify(invars[i])

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

