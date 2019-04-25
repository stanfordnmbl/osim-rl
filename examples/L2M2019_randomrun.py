from osim.env import L2M2019Env

env = L2M2019Env(seed=5, difficulty=2)
env.change_model(model='3D', difficulty=2, seed=11)
observation = env.reset()
for i in range(300):
    observation, reward, done, info = env.step(env.action_space.sample(), project = True)
    obs_dict = env.get_observation_dict()
    import pdb; pdb.set_trace()
    if done:
        env.reset(seed=10)
        break
