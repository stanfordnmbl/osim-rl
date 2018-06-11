from osim.env import L2RunEnv
import opensim

env = L2RunEnv(visualize=True)
observation = env.reset()

s = 0
for s in range(80):
    d = False

    if s == 30:
        state_old = env.osim_model.get_state()
        print("State stored")
        print(state_old)
    if s % 50 == 49:
        env.osim_model.set_state(state_old)
        print("Rollback")
        print(state_old)

    o, r, d, i = env.step(env.action_space.sample())
