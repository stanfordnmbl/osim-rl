import os
from osim.env import OsimEnv

## Define an environment where we teach an agent to stand still
# We use a walker model and define the objective to keep the center of mass still
class StandingEnv(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../osim/models/SoccerKickingModel.osim')    
    time_limit = 300
    ninput = 99
    
    def is_done(self):
        # End the simulation if the pelvis is too low
        state_desc = self.get_state_desc()
        return False

    def get_observation(self):
        state_desc = self.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []

        # # Map some of the state variables to the observation vector
        # for body_part in ["pelvis","head","torso","toes_l","toes_r","talus_l","talus_r"]:
        #     res = res + state_desc["body_pos_rot"][body_part][2:]
        #     res = res + state_desc["body_pos"][body_part][0:2]
        #     res = res + state_desc["body_vel_rot"][body_part][2:]
        #     res = res + state_desc["body_vel"][body_part][0:2]
        #     res = res + state_desc["body_acc_rot"][body_part][2:]
        #     res = res + state_desc["body_acc"][body_part][0:2]

        # for joint in ["ankle_l","ankle_r","back","ground_pelvis","hip_l","hip_r","knee_l","knee_r"]:
        #     res = res + state_desc["joint_pos"][joint]
        #     res = res + state_desc["joint_vel"][joint]
        #     res = res + state_desc["joint_acc"][joint]

        # res = res + state_desc["misc"]["mass_center_pos"] + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        return res

    def get_observation_space_size(self):
        return 99

    def reward(self):
        # Get the current state and the last state
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0

        res = 0
        # # Penalize movement of the pelvis
        # res = -(prev_state_desc["misc"]["mass_center_pos"][0] - state_desc["misc"]["mass_center_pos"][0])**2\
        #       -(prev_state_desc["misc"]["mass_center_pos"][1] - state_desc["misc"]["mass_center_pos"][1])**2

        # # Penalize very low position of the pelvis
        # res += -(state_desc["joint_pos"]["ground_pelvis"][2] < 0.8)
        
        return res

env = StandingEnv(visualize=True)

observation = env.reset()
for i in range(2000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(action)
    print("Reward %f" % reward)
    if done:
        env.reset()
