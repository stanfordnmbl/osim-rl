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
for i in range(500):
    action = params['policy'].get_action(obs)
    step = env.step(action[0])

    if env._wrapped_env.__class__.__name__ == "ArmEnv" and i % 100 == 0:
        env._wrapped_env.new_target()
        print(env._wrapped_env.shoulder, env._wrapped_env.elbow)
