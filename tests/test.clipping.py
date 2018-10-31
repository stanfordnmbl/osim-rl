from osim.env import L2RunEnv
import numpy as np
import unittest

class ActivationsTest(unittest.TestCase):
    def test_clipping(self):
        env = L2RunEnv(visualize=False)
        observation = env.reset()

        env.step(np.array([5.0] * 18))
        self.assertLessEqual( np.sum(env.osim_model.last_action), 18.1 ) 
        env.step(np.array([-1.0] * 18))
        self.assertGreaterEqual( np.sum(env.osim_model.last_action), -0.1 ) 
        
if __name__ == '__main__':
    unittest.main()
