import opensim as osim

import sys

# -- CONFIG --
stepsize = 0.05
nsteps = 100

# Get the model
model = osim.Model("../models/gait9dof18musc_Thelen_BigSpheres_20161017.osim")

# Enable the visualizer
model.setUseVisualizer(True)

# Construct the controller
muscleSet = model.getMuscles()

# Initialize simulation
state = model.initSystem()
model.equilibrateMuscles(state)
manager = osim.Manager(model)

actControls = osim.Vector(1, 0.0);
time = osim.Vector(1, state.getTime());

actControls[0] = 1

# Simulatie
for i in range(0,nsteps):
    # activate the muscles corresponding to the current step
    # yes, this makes no sense, it's just a test
    muscle = muscleSet.get(i % 18)
    muscle.setActivation(state, 20)
    manager.setInitialTime(stepsize * i)
    manager.setFinalTime(stepsize * (i + 1))
    manager.integrate(state)
