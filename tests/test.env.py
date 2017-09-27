from osim.env import RunEnv
import numpy as np
import unittest

class SimulationTest(unittest.TestCase):
    def test_reset(self):
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

    def test_actions(self):
        env = RunEnv(visualize=False)
        env.reset()
        v = env.action_space.sample()
        v[0] = 1.5
        v[1] = -0.5
        observation, reward, done, info = env.step(v)
        self.assertLessEqual(env.last_action[0],1.0)
        self.assertGreaterEqual(env.last_action[1],0.0)

    def test_first_obs(self):
        env = RunEnv(visualize=False)
        observation_start = env.reset()
        observation, reward, done, info = env.step(env.action_space.sample())
        self.assertAlmostEqual(observation_start[-1], observation[-1])
        self.assertAlmostEqual(observation_start[-2], observation[-2])

if __name__ == '__main__':
    unittest.main()
