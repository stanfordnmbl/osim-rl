from osim.env.osim import Run3DEnv
import pprint

env = Run3DEnv(visualize=True)

observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample(), project = False)
#    print(len(observation))
    pprint.pprint(observation)
    if done:
        env.reset()
