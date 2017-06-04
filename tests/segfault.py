from osim.env.osim import Osim
import os
import opensim

opensim.Body('block', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
opensim.Body('block', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );

model_path = os.path.join(os.path.dirname(__file__), '../osim/models/gait9dof18musc.osim')

def test(model_path, visualize):
    model = opensim.Model(model_path)
    brain = opensim.PrescribedController()
    model.addController(brain)
    state = model.initSystem()

    muscleSet = model.getMuscles()
    for j in range(muscleSet.getSize()):
        brain.addActuator(muscleSet.get(j))
        func = opensim.Constant(1.0)
        brain.prescribeControlForActuator(j, func)

    block = opensim.Body('block', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
    model.addComponent(block)
    pj = opensim.PlanarJoint('pin',
                             model.getGround(), # PhysicalFrame
                             opensim.Vec3(0, 0, 0),
                             opensim.Vec3(0, 0, 0),
                             block, # PhysicalFrame
                             opensim.Vec3(0, 0, 0),
                             opensim.Vec3(0, 0, 0))
    model.addComponent(pj)
    model.initSystem()
    pj.getCoordinate(1)


    
test(model_path,False)
test(model_path,False)

from osim.env.run import RunEnv

env = RunEnv(visualize=False)
env1 = RunEnv(visualize=False)
env1.reset()
env1.compute_reward()
env.reset()
env.compute_reward()
