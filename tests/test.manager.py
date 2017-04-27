# This script goes through OpenSim funcionalties
# required for OpenSim-RL
import opensim

# Settings
stepsize = 0.01

# Load existing model
model_path = "../osim/models/arm2dof6musc.osim"
model = opensim.Model(model_path)
model.setUseVisualizer(True)

# Build the controller (we will iteratively
# update it at every step of the simulation)
brain = opensim.PrescribedController()
controllers = []

# Add actuators to the controller
state = model.initSystem() # we need to initialize the system (?)
muscleSet = model.getMuscles()
for j in range(muscleSet.getSize()):
    func = opensim.Constant(1.0)
    controllers.append(func)
    brain.addActuator(muscleSet.get(j))
    brain.prescribeControlForActuator(j, func)
model.addController(brain)

# Reinitialize the system with the new controller
state0 = model.initSystem()
state = opensim.State(state0)

for i in range(1000):
    # Set some excitation values
    for j in range(muscleSet.getSize()):
        controllers[j].setValue( ((i + j) % 10) * 0.1)

    # Integrate
    t = state.getTime()
    manager = opensim.Manager(model)
    manager.integrate(state, t + stepsize)

    # Report activations and excitations
    model.realizeDynamics(state)
    print("%f %f" % (t,muscleSet.get(0).getActivation(state)))
    print("%f %f" % (t,muscleSet.get(0).getExcitation(state)))

    # Restart the model every 100 frames
    if (i + 1) % 100 == 0:
        state = opensim.State(state0)
