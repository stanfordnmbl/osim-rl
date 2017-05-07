from osim.env.run import RunEnv, generate_env

env = RunEnv(visualize=True)

observation = env.reset(5,22)
for i in range(500):
    observation, reward, done, info = env.step(env.action_space.sample())
    if i % 100 == 99 or done:
        observation = env.reset(5,22)
