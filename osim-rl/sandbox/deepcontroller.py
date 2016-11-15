import opensim as osim

import sys

# Some meta parameters
stepsize = 0.02
nsteps = 100

# Get the model
model = osim.Model("../models/gait9dof18musc_Thelen_BigSpheres_20161017.osim")

# Enable the visualizer
model.setUseVisualizer(True)

# Get the muscles
muscleSet = model.getMuscles()

# Initialize simulation
state = model.initSystem()
model.equilibrateMuscles(state)
manager = osim.Manager(model)

# Simulatie
for i in range(0,nsteps):
    # activate the muscles corresponding to the current step
    # yes, this makes no sense, it's just a test
    for j in range(0,18):
        muscle = muscleSet.get(j)
        muscle.setActivation(state, 1.0 *(-1) ** j)

    # Integrate one step
    manager.setInitialTime(stepsize * i)
    manager.setFinalTime(stepsize * (i + 1))
    manager.integrate(state)
