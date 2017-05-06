# This script goes through OpenSim funcionalties
# required for OpenSim-RL
import opensim

# Settings
stepsize = 0.01

# Load existing model
model_path = "../osim/models/gait9dof18musc.osim"
model = opensim.Model(model_path)
model.setUseVisualizer(True)

# Create the ball
r = 0.000001
ballBody = opensim.Body('ball', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
ballGeometry = opensim.Ellipsoid(r, r, r)
ballGeometry.setColor(opensim.Gray)
ballBody.attachGeometry(ballGeometry)

# Attach ball to the model
ballJoint = opensim.FreeJoint("weldball",
                         model.getGround(), # PhysicalFrame
                         opensim.Vec3(0, 0, 0),
                         opensim.Vec3(0, 0, 0),
                         ballBody, # PhysicalFrame
                         opensim.Vec3(0, 0, 0),
                         opensim.Vec3(0, 0, 0))
model.addComponent(ballJoint)
model.addComponent(ballBody)

# Add contact
ballContact = opensim.ContactSphere(r, opensim.Vec3(0,0,0), ballBody);
model.addContactGeometry(ballContact)

# Reinitialize the system with the new controller
state = model.initSystem()

for i in range(6):
    ballJoint.getCoordinate(i).setLocked(state, True)

# Simulate
for i in range(100):
    t = state.getTime()
    manager = opensim.Manager(model)
    manager.integrate(state, t + stepsize)

    # Restart the model every 10 frames, with the new position of the ball
    if (i + 1) % 10 == 0:
        newloc = opensim.Vec3(float(i) / 5, 0, 0)
        opensim.PhysicalOffsetFrame.safeDownCast(ballJoint.getChildFrame()).set_translation(newloc)

        r = i * 0.005
        #ballGeometry.set_radii(opensim.Vec3(r,r,r))
        #ballBody.scale(opensim.Vec3(r, r, r))
        ballContact.setRadius(r)

        state = model.initializeState()
        
        ballJoint.getCoordinate(3).setValue(state, i / 100.0)
        for i in range(6):
            ballJoint.getCoordinate(i).setLocked(state, True)
