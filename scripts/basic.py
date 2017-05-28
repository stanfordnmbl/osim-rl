from osim.env.run import RunEnv

env = RunEnv(visualize=True)

observation = env.reset(difficulty = 0)
for i in range(500):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        observation = env.reset(difficulty = 0)
