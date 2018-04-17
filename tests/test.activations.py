from osim.env import L2RunEnv
import numpy as np
import unittest

class ActivationsTest(unittest.TestCase):
    def test_activations(self):
        env = L2RunEnv(visualize=False)
        observation = env.reset()

        newact = np.array([0.0] * 18)
        env.osim_model.set_activations(newact)
        
        current = np.array(env.osim_model.get_activations())
        dist = np.linalg.norm(newact - current)
        self.assertTrue(dist < 0.05)
        
        newact = np.array([1.0] * 18)
        env.osim_model.set_activations(newact)

        current = np.array(env.osim_model.get_activations())
        dist = np.linalg.norm(newact - current)
        self.assertTrue(dist < 0.05)

if __name__ == '__main__':
    unittest.main()
