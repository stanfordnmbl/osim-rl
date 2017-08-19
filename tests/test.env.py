from osim.env import RunEnv
import numpy as np
import unittest

class SimulationTest(unittest.TestCase):
    def test1(self):
        
        env = RunEnv(visualize=False)
        observation = env.reset(difficulty=2, seed=123)
        env1 = env.env_desc
        observation = env.reset(difficulty=2, seed=3)
        observation = env.reset(difficulty=2, seed=3)
        observation = env.reset(difficulty=2, seed=3)
        observation = env.reset(difficulty=2, seed=3)
        observation = env.reset(difficulty=2, seed=123)
        env2 = env.env_desc

        s = map(lambda x: x[0] - x[1], list(zip(env1["obstacles"][1],env2["obstacles"][1])))
        self.assertAlmostEqual(sum([k**2 for k in s]), 0.0)

        action = env.action_space.sample()
        action[5] = np.NaN
        self.assertRaises(ValueError, env.step, action)

if __name__ == '__main__':
    unittest.main()
