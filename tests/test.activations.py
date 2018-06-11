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

    def test_activations_changes(self):
        env = L2RunEnv(visualize=False)

        # Do not set new activations
        newAct = [0.9] * 18
        observation = env.reset()
        env.osim_model.set_activations(newAct)
        for i in range(5): 
            withoutAct = env.osim_model.get_activations()
            observation, reward, done, info = env.step([0.5]*18)

        # Set new activations
        newAct = [0.1] * 18
        observation = env.reset()
        env.osim_model.set_activations(newAct)
        for i in range(5): 
            withAct = env.osim_model.get_activations()
            observation, reward, done, info = env.step([0.5]*18)

        dist = np.linalg.norm(np.array(withAct) - np.array(withoutAct))

        self.assertFalse(dist < 1e-2,"Activations after 5 steps haven't changed (despite different initial conditions)")

        
if __name__ == '__main__':
    unittest.main()
