from osim.env import ArmEnv, GaitEnv, StandEnv, CrouchEnv, HopEnv
import argparse

from rllab.algos.ddpg import DDPG
from rllab.algos.trpo import TRPO
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite, StubClass
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

parser = argparse.ArgumentParser(description='Train a policy')
parser.add_argument('-a', action="store", dest="alg")
parser.add_argument('-e', action="store", dest="env")
parsed = parser.parse_args()

stub(globals())

alg = "DDPG"

envs = {"Arm": ArmEnv,
        "Stand": StandEnv,
        "Gait": GaitEnv,
        "Crouch": CrouchEnv,
        "Hop": HopEnv}

env = normalize(envs[parsed.env](visualize=False))

# env = normalize(CartpoleEnv())
# env = normalize(GymEnv("Pendulum-v0", record_video=False, record_log=False))

if alg == "DDPG":
    qf = ContinuousMLPQFunction(
        env_spec=env.spec,
        hidden_sizes=(64, 64, 64)
    )

    es = OUStrategy(env_spec=env.spec, theta = 0.5)

    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32, 32)
    )

    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=32,
        max_path_length=100,
        epoch_length=100,
        min_pool_size=1000,
        n_epochs=1000,
        discount=0.99,
        scale_reward=0.01,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

else:
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(128, 128)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=500,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )


run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    use_gpu=True,
    # plot=True,
)

