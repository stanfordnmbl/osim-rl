from osim.env.generic import OsimEnv

env = OsimEnv(visualize=True)

observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
