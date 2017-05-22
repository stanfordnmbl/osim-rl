from osim.env.run import RunEnv, generate_env

env = RunEnv(visualize=True)

observation = env.reset(difficulty = 10)
for i in range(500):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        observation = env.reset(difficulty = 10)
