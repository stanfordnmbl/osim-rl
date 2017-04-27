# This script goes through OpenSim funcionalties
# required for OpenSim-RL
import opensim

# Settings
stepsize = 0.01

# Load existing model
model_path = "../osim/models/gait9dof18musc.osim"
model = opensim.Model(model_path)
model.setUseVisualizer(True)

# Build the controller (we will iteratively
# update it at every step of the simulation)
brain = opensim.PrescribedController()
controllers = []

# Add actuators to the controller
state = model.initSystem() # we need to initialize the system (?)
muscleSet = model.getMuscles()
forceSet = model.getForceSet()

for j in range(muscleSet.getSize()):
    func = opensim.Constant(1.0)
    controllers.append(func)
    brain.addActuator(muscleSet.get(j))
    brain.prescribeControlForActuator(j, func)

model.addController(brain)

blockos = opensim.Body('blockos', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
pj = opensim.PinJoint("pinblock",
                         model.getGround(), # PhysicalFrame
                         opensim.Vec3(-0.5, 0, 0),
                         opensim.Vec3(0, 0, 0),
                         blockos, # PhysicalFrame
                         opensim.Vec3(0, 0, 0),
                         opensim.Vec3(0, 0, 0))

bodyGeometry = opensim.Ellipsoid(0.1, 0.1, 0.1)
bodyGeometry.setColor(opensim.Gray)
blockos.attachGeometry(bodyGeometry)

model.addComponent(pj)
model.addComponent(blockos)
#block = opensim.ContactMesh('block.obj',opensim.Vec3(0,0,0), opensim.Vec3(0,0,0), blockos);
block = opensim.ContactSphere(0.4, opensim.Vec3(0,0,0), blockos);
model.addContactGeometry(block)

# Reinitialize the system with the new controller
state0 = model.initSystem()
state = opensim.State(state0)

# Change max force
muscleSet.get(0).setMaxIsometricForce(100000.0)

# Get ligaments
ligamentSet = []
for j in range(20, 26):
    ligamentSet.append(opensim.CoordinateLimitForce.safeDownCast(forceSet.get(j)))

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

    # Ligaments
    for lig in ligamentSet:
        print(lig.calcLimitForce(state))

    # Restart the model every 100 frames
    if (i + 1) % 100 == 0:
        state = opensim.State(state0)
