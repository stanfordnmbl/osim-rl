from osim.env.run import RunEnv, generate_env

env = RunEnv(visualize=False)
env.setup(10)

observation = env.reset()
for i in range(500):
    observation, reward, done, info = env.step(env.action_space.sample())
    if i % 100 == 99:
        env.setup(10)
        observation = env.reset()
    print(reward)
