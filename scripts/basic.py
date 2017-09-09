from osim.env.run import RunEnv

env = RunEnv(visualize=True, max_obstacles=3)

observation = env.reset(difficulty = 3)
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
#        break
    

