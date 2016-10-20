import opensim as osim
import numpy as np
import sys

# Some meta parameters
stepsize = 0.05
nsteps = 100

# Get the model
model = osim.Model("../models/gait9dof18musc_Thelen_BigSpheres_20161017.osim")

# Enable the visualizer
model.setUseVisualizer(True)

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

head = jointSet.get(0)
head = osim.PlanarJoint.safeDownCast(head)

def brain_model(invars):
    # activations = F(params, invars)
    activations = np.array([0] * 18)
    return activations

class Objective:
    sumOfObjectives = 0

    def __init__(self):
        pass

    def reset(self):
        self.sumOfObjectives = 0
    
    def update(self, state):
        self.sumOfObjectives = self.sumOfObjectives - head.getCoordinate(1).getValue(state)

    def value(self):
        return self.sumOfObjectives

obj = Objective()
    
# Simulatie
for i in range(0,nsteps):
    invars = np.array([0] * (26 * 3))

#    for j in xrange(26):
#        print(forceSet.get(j))
    
    activations = brain_model(invars)
    
    for j in range(18):
        muscle = muscleSet.get(j % 18)
        muscle.setActivation(state, activations[j])

    obj.update(state)

    # Integrate one step
    manager.setInitialTime(stepsize * i)
    manager.setFinalTime(stepsize * (i + 1))
    manager.integrate(state)
