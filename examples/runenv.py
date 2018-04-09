from osim.env.osim import L2RunEnv
import pprint

env = L2RunEnv(visualize=True)

observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    print(len(observation))
#    pprint.pprint(env.get_state_desc())
    if done:
        env.reset()
