from osim.env.osim import L2RunEnv
import pprint

env = L2RunEnv(visualize=True)

observation = env.reset()
for i in range(200):
    res = env.step(env.action_space.sample())
    observation, reward, done, info = res
    print(res)

    # pprint.pprint(env.get_state_desc())
    if done:
        env.reset()
