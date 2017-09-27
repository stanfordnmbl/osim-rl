from osim.env.run import RunEnv

env = RunEnv(visualize=True, max_obstacles = 10)

observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
#        break
    

