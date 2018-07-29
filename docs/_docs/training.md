---
title: Training your first model
---

Your goal is to construct a controller, i.e. a function from the state space (current positions, velocities and accelerations of joints) to action space (muscle excitations), that will enable the model to perform a certain task like walking, reaching, throwing a ball, etc. Suppose you trained a neural network mapping observations (the current state of the model) to actions (muscle excitations), i.e. you have a function `action = my_controller(observation)`, then
```python
# ...
total_reward = 0.0
for i in range(200):
    # make a step given by the controller and record the state and the reward
    observation, reward, done, info = env.step(my_controller(observation))
    total_reward += reward
    if done:
        break

# Your reward is
print("Total reward %f" % total_reward)
```
There are many ways to construct the function `my_controller(observation)`. We will show how to do it with a DDPG (Deep Deterministic Policy Gradients) algorithm, using `keras-rl`. 

## Your first controller

Below we present how to train a basic controller using [keras-rl](https://github.com/matthiasplappert/keras-rl). First you need to install extra packages:

    conda install keras -c conda-forge
    pip install tensorflow git+https://github.com/matthiasplappert/keras-rl.git
    git clone https://github.com/stanfordnmbl/osim-rl.git

`keras-rl` is an excellent package compatible with [OpenAI Gym](http://gym.openai.com/), which allows you to quickly build your first models!


    cd osim-rl/examples

To train the model using DDPG algorithm you can simply run the scirpt
`ddpg.keras-rl.py` as follows:

### Training

    python ddpg.keras-rl.py --visualize --train --model sample

### Testing

and for the gait example (walk as far as possible):

    python ddpg.keras-rl.py --visualize --test --model sample
