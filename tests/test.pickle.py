import pickle
from env.human import GaitEnv

env = GaitEnv(visualize=False)
f = open('test.p', 'wb')
pickle.dump(env,f)
f.close()
