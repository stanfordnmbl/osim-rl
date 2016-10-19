import opensim as osim

import sys

# Get the model
model = osim.Model("../models/gait9dof18musc_Thelen_BigSpheres_20161017.osim")

# Enable the visualizer
model.setUseVisualizer(True)

# Construct the controller
brain = osim.PrescribedController()
muscleSet = model.getMuscles();
hamstrings_r = muscleSet.get(0);
brain.addActuator(hamstrings_r);
brain.prescribeControlForActuator("hamstrings_r",
                                  osim.StepFunction(0.5, 3.0, 0.3, 1.0))

# Add it to the model
model.addController(brain)

# Initialize simulation
state = model.initSystem()
model.equilibrateMuscles(state)
manager = osim.Manager(model)
manager.setInitialTime(0)
manager.setFinalTime(10.0)

# Simulatie
manager.integrate(state)
