from osim.env import RunEnv
import numpy as np
import unittest

class SimulationTest(unittest.TestCase):
    def test1(self):
        env = RunEnv(visualize=False)
        observation = env.reset()

        action = env.action_space.sample()
        action[5] = np.NaN
        self.assertRaises(ValueError, env.step, action)

    def test2(self):
        env = RunEnv(visualize=False)
        env.reset()
        desc = env.env_desc
        env.reset()
        env.reset_from_desc(desc)

if __name__ == '__main__':
    unittest.main()
