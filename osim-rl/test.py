from env.arm import ArmEnv
from env.human import HopEnv, GaitEnv, StandEnv
import joblib

env = HopEnv(visualize=True)

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
# from rllab.exploration_strategies.ou_strategy import OUStrategy
# from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
# from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

fname = "/home/lukasz/workspace/rllab/data/local/experiment/experiment_2016_11_18_15_36_08_0001/params.pkl"
#fname = "/home/lukasz/workspace/rllab/data/local/experiment/experiment_2016_11_16_21_55_28_0001/params.pkl"
params = joblib.load(fname)
#env = params['env']

obs = env.reset()
for i in range(500):
    action = params['policy'].get_action(obs)
    step = env.step(action[0])
