from environments.arm import ArmEnv

env = ArmEnv(visualize=True)
env.reset()

for i in range(10000):
    env.step([i] * 6)
