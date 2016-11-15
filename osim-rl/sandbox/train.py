from environments.arm import ArmEnv

env = ArmEnv()
env.reset()

for i in xrange(100):
    env.step()
