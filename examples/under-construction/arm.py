import os
from osim.env import OsimEnv
import pprint
import numpy as np

class Arm3dEnv(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../osim/models/MoBL_ARMS_J_Simple_032118.osim')
    time_limit = 200
    current_objective = np.array([0,0,0])
    
    def is_done(self):
        # End the simulation if the pelvis is too low
        state_desc = self.get_state_desc()
        return False

    def get_observation(self):
        state_desc = self.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []

        # Map some of the state variables to the observation vector
        for body_part in state_desc["body_pos_rot"].keys():
            res = res + state_desc["body_pos_rot"][body_part][2:]
            res = res + state_desc["body_pos"][body_part][0:2]
            res = res + state_desc["body_vel_rot"][body_part][2:]
            res = res + state_desc["body_vel"][body_part][0:2]
            res = res + state_desc["body_acc_rot"][body_part][2:]
            res = res + state_desc["body_acc"][body_part][0:2]

        for joint in state_desc["joint_pos"].keys():
            res = res + state_desc["joint_pos"][joint]
            res = res + state_desc["joint_vel"][joint]
            res = res + state_desc["joint_acc"][joint]

        res = res + state_desc["misc"]["mass_center_pos"] + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]
        res += self.current_objective.tolist()

        res = np.array(res)
        res[np.isnan(res)] = 0

        return res

    def get_observation_space_size(self):
        return 168

    def reset_objective(self):
        self.current_objective = np.random.uniform(-0.5,0.5,3)

    def reset(self):
        print(self.reward())
        self.reset_objective()
        return super(Arm3dEnv, self).reset()

    def reward(self):
        # Get the current state and the last state
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        state_desc = self.get_state_desc()

        res = 0

        # # Penalize movement of the pelvis
        # res = -(prev_state_desc["misc"]["mass_center_pos"][0] - state_desc["misc"]["mass_center_pos"][0])**2\
        #       -(prev_state_desc["misc"]["mass_center_pos"][1] - state_desc["misc"]["mass_center_pos"][1])**2

        # # Penalize very low position of the pelvis
        # res += -(state_desc["joint_pos"]["ground_pelvis"][2] < 0.8)
        
        return -np.linalg.norm(np.array(state_desc["markers"]["Handle"]["pos"]) - self.current_objective)

env = Arm3dEnv(visualize=True)

if __name__ == '__main__':
    observation = env.reset()
    for i in range(200):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
