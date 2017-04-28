from osim.env import GaitEnv
from guppy import hpy; h=hpy()

env = GaitEnv(visualize=False)
observation = env.reset()

s = 0
while s<50000:   
    print(h.heap().size)
    o = env.reset()   
    d = False   
    while not d:   
        o, r, d, i = env.step(env.action_space.sample())   
        s += 1
