from osim.env import ArmEnv, HopEnv, GaitEnv, StandEnv, CrouchEnv
import joblib
import argparse

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

parser = argparse.ArgumentParser(description='Test a policy')
parser.add_argument('-p', action="store", dest="params_file")
parsed = parser.parse_args()

params = joblib.load(parsed.params_file)
env = params['env']
env.test = True

obs = env.reset()
total_rew = 0.0
for i in range(200):
    action = params['policy'].get_action(obs)
    obs,reward,_,_ = env.step(action[0])
    total_rew += reward

    if env._wrapped_env.__class__.__name__ == "ArmEnv" and i % 200 == 0:
        env._wrapped_env.new_target()
        print(env._wrapped_env.shoulder, env._wrapped_env.elbow)

print(total_rew)
