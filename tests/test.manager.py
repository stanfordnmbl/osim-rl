import opensim

model_path = "../osim/models/arm2dof6musc.osim"
model = opensim.Model(model_path)
model.setUseVisualizer(True)
state = model.initSystem()
manager = opensim.Manager(model)
muscleSet = model.getMuscles()
stepsize = 0.01

for i in range(10):
    for j in range(muscleSet.getSize()):
        muscleSet.get(j).setActivation(state, 1.0)
#        muscleSet.get(j).setExcitation(state, 1.0)
    t = state.getTime()
    manager.setInitialTime(stepsize * i)
    manager.setFinalTime(stepsize * (i + 1))
    manager.integrate(state)
    model.realizeDynamics(state)
    print("%f %f" % (t,muscleSet.get(0).getActivation(state)))
