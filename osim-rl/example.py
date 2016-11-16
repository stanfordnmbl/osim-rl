from env.arm import ArmEnv
from env.human import GaitEnv

env = GaitEnv(visualize=True)
env.reset()

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
# from rllab.exploration_strategies.ou_strategy import OUStrategy
# from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
# from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

for i in range(10000):
    env.step([i] * 24)
