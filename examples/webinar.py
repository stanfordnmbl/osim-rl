import os
from osim.env import OsimEnv
import operator
import functools

## Define an environment where we teach an agent to stand still
# We use a walker model and define the objective to keep the center of mass still
class StandingEnv(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../osim/models/gait9dof18musc.osim')    
    time_limit = 300

    def reward(self):
        # Get the current state and the last state
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0

        # Penalize movement of the pelvis
        res = -(prev_state_desc["misc"]["mass_center_pos"][0] - state_desc["misc"]["mass_center_pos"][0])**2\
              -(prev_state_desc["misc"]["mass_center_pos"][1] - state_desc["misc"]["mass_center_pos"][1])**2

        # Penalize very low position of the pelvis
        res += -(state_desc["joint_pos"]["ground_pelvis"][2] < 0.8)
        
        return res

    ## WARNING: THIS IS SUBOPTIMAL WAY TO CREATE OBSERVATION VECTOR
    ## IT SHOULD BE DESIGNED CAREFULLY FOR EACH MODEL
    def all_values(self, d):
        l = [self.all_values(v) if type(v) == dict else v for v in d.values()]
        l = sum([v if type(v) == list else [v,] for v in l], [])
        return l

    def get_observation(self):
        state_desc = self.get_state_desc()
        return self.all_values(state_desc)

    def get_observation_space_size(self):
        return 408


