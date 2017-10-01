from osim.env import RunEnv
import opensim

env = RunEnv(visualize=True)
observation = env.reset(seed=0)

s = 0
for s in range(50000):
    d = False

    if s == 30:
        state_old = opensim.State(env.osim_model.state)
        print("State stored")
        print(state_old)
    if s % 50 == 49:
        env.osim_model.revert(state_old)
        state_old = opensim.State(state_old)
        print("Rollback")
        print(state_old)

    o, r, d, i = env.step(env.action_space.sample())
