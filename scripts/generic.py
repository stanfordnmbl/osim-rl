from osim.env.osim import RunEnv
import pprint

env = RunEnv(visualize=True)

observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    pprint.pprint(observation)
    if done:
        env.reset()
