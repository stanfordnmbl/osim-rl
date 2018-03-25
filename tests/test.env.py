from osim.env import L2RunEnv
import numpy as np
import unittest

class SimulationTest(unittest.TestCase):
    def test_reset(self):
        env = L2RunEnv(visualize=False)
        for i in range(10):
            observation = env.reset()

        action = env.action_space.sample()
        action[5] = np.NaN
        self.assertRaises(ValueError, env.step, action)

    def test_actions(self):
        env = L2RunEnv(visualize=False)
        env.reset()
        v = env.action_space.sample()
        v[0] = 1.5
        v[1] = -0.5
        observation, reward, done, info = env.step(v)

if __name__ == '__main__':
    unittest.main()
