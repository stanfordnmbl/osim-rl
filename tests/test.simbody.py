from osim.env import ProstheticsEnv
import numpy as np
import unittest

class SimulationTest(unittest.TestCase):
    def test1(self):
        
        env = ProstheticsEnv(visualize=True)
        observation = env.reset()

        simbody_state = env.osim_model.get_state()
        print(simbody_state.getNumSubsystems())
        print(simbody_state.getY())
        oldy = simbody_state.updY()
        for i in range(len(oldy)):
            oldy[i] += 0.2
        print(simbody_state.getY())
        env.osim_model.set_state(simbody_state)

        action = env.action_space.sample()

        for i in range(50):
            env.step(action)


if __name__ == '__main__':
    unittest.main()
