from osim.env.run import RunEnv
#from osim.env.arm import ArmEnv

env = RunEnv(visualize=True, max_obstacles = 0, report = "test")
#env = ArmEnv(visualize=True)

observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        print("RESET")
        env.reset()
