from osim.env import ProstheticsEnv

env = ProstheticsEnv(seed=5)
env.change_model(model='3D', prosthetic=True, difficulty=2, seed=11)
observation = env.reset()
for i in range(300):
    observation, reward, done, info = env.step(env.action_space.sample(), project = False)
    if done:
        env.reset(seed=10)
