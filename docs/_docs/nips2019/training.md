---
title: Training an arm
---

Try out our toy environment `Arm2DEnv` and an example code for training a controller for the environment.
The arm model has a weak shoulder muscle that it cannot keep its arm forward.

Test `Arm2DEnv`:

    python -m examples.arm2d

You can train a neural network model using existing frameworks. For example, to train a DDPG using `keras-rl` simply install `keras-rl` with following lines

    pip install keras-rl

Now, you can train a control model (DDPG) for the environment:

    python -m examples.train_arm --train --model sample

To test a control model run

    python -m examples.train_arm --test --visualize --model examples/model_exp

where `model_exp_actor` and `model_exp_critic` are networks trained with `python examples.train_arm --train --steps 100000 --model examples/model_exp`
