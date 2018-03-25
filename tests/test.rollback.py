from osim.env import L2RunEnv
import opensim

env = L2RunEnv(visualize=True)
observation = env.reset()

s = 0
for s in range(300):
    d = False

    if s == 30:
        state_old = opensim.State(env.osim_model.state)
        print("State stored")
        print(state_old)
    if s % 50 == 49:
        env.osim_model.set_state(state_old)
        state_old = opensim.State(state_old)
        print("Rollback")
        print(state_old)

    o, r, d, i = env.step(env.action_space.sample())
