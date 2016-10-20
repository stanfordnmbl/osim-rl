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

# Get the model
model = osim.Model("../models/gait9dof18musc_Thelen_BigSpheres_20161017.osim")

# Enable the visualizer
model.setUseVisualizer(False)

# Get the muscles
muscleSet = model.getMuscles()
forceSet = model.getForceSet()
bodySet = model.getBodySet()
jointSet = model.getJointSet()

# Initialize simulation
state = model.initSystem()
model.equilibrateMuscles(state)
manager = osim.Manager(model)

# Print bodies
for i in xrange(13):
    print(bodySet.get(i).getName())

for i in xrange(13):
    print(jointSet.get(i).getName())

ground_pelvis = osim.PlanarJoint.safeDownCast(jointSet.get(0))

hip_r = osim.PinJoint.safeDownCast(jointSet.get(1))
knee_r = osim.CustomJoint.safeDownCast(jointSet.get(2))
ankle_r = osim.PinJoint.safeDownCast(jointSet.get(3))
subtalar_r = osim.WeldJoint.safeDownCast(jointSet.get(4))
mtp_r = osim.WeldJoint.safeDownCast(jointSet.get(5))

hip_l = osim.PinJoint.safeDownCast(jointSet.get(6))
knee_l = osim.CustomJoint.safeDownCast(jointSet.get(7))
ankle_l = osim.PinJoint.safeDownCast(jointSet.get(8))
subtalar_l = osim.WeldJoint.safeDownCast(jointSet.get(9))
mtp_l = osim.PinJoint.safeDownCast(jointSet.get(10))

back = osim.PinJoint.safeDownCast(jointSet.get(11))
back1 = osim.WeldJoint.safeDownCast(jointSet.get(12))

# DEFINE THE BRAIN
net = Sequential()
net.add(Dense(output_dim=noutput, input_dim=ninput))
net.add(Activation("relu"))
net.add(Dense(output_dim=noutput))
net.add(Activation("softmax"))
net.compile(loss='mean_squared_error', optimizer='sgd')

def brain_model(invars):
    # activations = F(params, invars)
    # THE NEURAL NET COMES HERE!
    activations = np.array([0] * noutput)
    activations = net.predict(invars.reshape(-1,4))
    return activations[0]

class Objective:
    sumOfObjectives = 0

    def __init__(self):
        pass

    def reset(self):
        self.sumOfObjectives = 0
    
    def update(self, state):
        upd = - ground_pelvis.getCoordinate(1).getValue(state)
        self.sumOfObjectives = self.sumOfObjectives + upd
        return upd

    def value(self):
        return self.sumOfObjectives

obj = Objective()
    
# Simulatie
for i in range(0,nsteps):
    invars = np.array([0] * ninput)

    # for j in range(18):
    #     invars[j] = muscleSet.get(j).getExcitation(state)
    invars[0] = ground_pelvis.getCoordinate(1).getValue(state)
    invars[1] = ground_pelvis.getCoordinate(2).getValue(state)
    invars[2] = hip_r.getCoordinate(0).getValue(state)
    invars[3] = hip_l.getCoordinate(0).getValue(state)
    
    activations = brain_model(invars)
    
    for j in range(noutput):
        muscle = muscleSet.get(j)
        muscle.setActivation(state, activations[j] * 10)

    obj.update(state)

    # Integrate one step
    manager.setInitialTime(stepsize * i)
    manager.setFinalTime(stepsize * (i + 1))
    manager.integrate(state)

    net.train_on_batch(np.array([invars]), np.array([[1]]))

print(obj.value())
