from osim.env.osim import Osim
import os
import opensim

model_path = os.path.join(os.path.dirname(__file__), '../osim/models/gait9dof18musc.osim')

def test(model_path, visualize):
    brain = opensim.PrescribedController()

    model = opensim.Model(model_path)
    model.initSystem()

    muscleSet = model.getMuscles()
        
    for j in range(muscleSet.getSize()):
        brain.addActuator(muscleSet.get(j))
       
    model.addController(brain)
    
test(model_path,False)
test(model_path,False)
