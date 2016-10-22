import opensim as osim
import numpy as np
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Some meta parameters
stepsize = 0.05
nsteps = 10
ninput = 4
noutput = 18
visualize = False

class Policy:
    net = Sequential()

    def __init__(self):
        # DEFINE THE BRAIN
        self.net.add(Dense(output_dim=noutput, input_dim=ninput))
        self.net.add(Activation("relu"))
        self.net.add(Dense(output_dim=noutput))
        self.net.add(Activation("softmax"))
        self.net.compile(loss='mean_squared_error', optimizer='sgd')

    def get_action(self, state):
        # activations = F(params, invars)
        # THE NEURAL NET COMES HERE!
        self.activations = np.array([0] * noutput)
        self.activations = self.net.predict(state.reshape(-1,4))
        return self.activations[0]

    def get_params(self)

class Objective:
    rewards = []

    def __init__(self):
        pass

    def reset(self):
        self.sum_rewards = 0
    
    def update(self, agent):
        reward = - agent.ground_pelvis.getCoordinate(1).getValue(agent.state)
        self.sum_rewards = self.sum_rewards + reward
        return reward

    def reward(self):
        return self.sum_rewards

class Agent:
    # Initialize simulation
    model = None
    state = None
    state0 = None

    obj = Objective()

    step = 0
    
    def __init__(self):
        # Get the model
        self.model = osim.Model("../models/gait9dof18musc_Thelen_BigSpheres_20161017.osim")

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
        self.mtp_l = osim.PinJoint.safeDownCast(self.jointSet.get(10))

        self.back = osim.PinJoint.safeDownCast(self.jointSet.get(11))
        self.back1 = osim.WeldJoint.safeDownCast(self.jointSet.get(12))

        self.reset()


    def reset(self):
        self.step = 0
        if not self.state0:
            self.state0 = self.model.initSystem()
        self.state = osim.State(self.state0)

        self.model.equilibrateMuscles(self.state)
        self.manager = osim.Manager(self.model)
        self.obj.reset()


    def simulate_step(self, policy):
        invars = np.array([0] * ninput)

        invars[0] = self.ground_pelvis.getCoordinate(1).getValue(self.state)
        invars[1] = self.ground_pelvis.getCoordinate(2).getValue(self.state)
        invars[2] = self.hip_r.getCoordinate(0).getValue(self.state)
        invars[3] = self.hip_l.getCoordinate(0).getValue(self.state)
    
        activations = policy.get_action(invars)
    
        for j in range(noutput):
            muscle = self.muscleSet.get(j)
            muscle.setActivation(self.state, activations[j] * 10)

        reward = self.obj.update(self)

        # Integrate one step
        self.manager.setInitialTime(stepsize * self.step)
        self.manager.setFinalTime(stepsize * (self.step + 1))
        self.manager.integrate(self.state)

        self.step = self.step + 1

        return reward

    def simulate_episode(self, policy):
        self.reset()
        for i in range(0,nsteps):
            self.simulate_step(policy)
        return self.obj.reward()

policy = Policy()
agent = Agent()
    
print(agent.simulate_episode(policy))
print(agent.simulate_episode(policy))
print(agent.simulate_episode(policy))
print(agent.simulate_episode(policy))
print(agent.simulate_episode(policy))
