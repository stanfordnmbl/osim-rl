from osim.env import ProstheticsEnv
import numpy as np
import unittest

class SetStateTest(unittest.TestCase):
    def test_activations(self):
        env = ProstheticsEnv(visualize=False, integrator_accuracy=1e-1)  # we quickly want to see what happens
        env.reset()
        state_checkpoint = env.osim_model.get_state()  # store state

        for i in range(5):
            env.step(env.action_space.high)  # execute step with static action
        obs1 = env.get_observation()

        env.osim_model.set_state(state_checkpoint)  # restore state
        for i in range(5):
            env.step(env.action_space.high)
        obs2 = env.get_observation()

        dist = np.sum((np.array(obs1) - np.array(obs2))**2)
        self.assertTrue(dist < 0.05)
        
if __name__ == '__main__':
    unittest.main()
