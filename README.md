# osim-rl

## What?

[OpenSim](https://github.com/opensim-org/opensim-core) is a biomechanical physics environment for musculoskeletal simulations. Biomechanical community designed a range of musculoskeletal models compatible with this environment. These models can be, for example, fit to clinical data to understand underlying causes of injuries using inverse kinematics and inverse dynamics.

For many of these models there are controllers designed for forward simulations of movement, however they are often finely tuned for the model and data. Advancements in reinforcement learning may allow building more robust controllers which can in turn provide another tool for validating the models. Moreover they could help visualize, for example, kinematics of patients after surgeries.

We include two musculoskeletal models: ARM with 6 muscles and 2 degrees of freedom and HUMAN with 18 muscles and 9 degrees of freedom. For the ARM model we designed an environment where the arm is supposed to reach certain points. For HUMAN we have four environmnets: standing still, crouch, jump and gait. Environments are compatible with [rllab](https://github.com/openai/rllab), [keras-rl](https://github.com/matthiasplappert/keras-rl) and [OpenAI gym](https://gym.openai.com/).

![ARM environment](https://github.com/kidzik/osim-rl/blob/master/demo/arm.gif)
![HUMAN environment](https://github.com/kidzik/osim-rl/blob/master/demo/stand.gif)

Note that from reinforcement learning perspective, due to high dimensionality of muscles space, the problem is significantly harder than 'textbook' reinforcement learning problems.

## Installation

Requires OpenSim 4.0 - https://github.com/opensim-org/opensim-core . You can either install it from source or use my builds https://github.com/kidzik/opensim-core/releases/tag/v4.0-1kidzik (since there are no official builds available.

For Debian/Ubuntu 64bit architecture run:

    wget https://github.com/kidzik/opensim-core/releases/download/v4.0-1kidzik/python3-opensim-4.0-1kidzik-amd64.deb
    sudo dpkg -i python3-opensim-4.0-1kidzik-amd64.deb
    sudo apt-get install -f
    sudo apt-get install python3-pip git
    pip3 install git+https://github.com/kidzik/osim-rl.git

## Basic usage

To run 200 steps of environment:

    from osim.env import ArmEnv

    env = ArmEnv(visualize=True)
    for i in range(200):
        env.step(env.action_space.sample())

## Objective

The goal is to construct a controler, i.e. a function from the state space to 

    from osim.env import ArmEnv

    env = ArmEnv(visualize=True)
    observation = env.reset() # restart the environment and get the current state
    
    def my_controler(observation):
        # your controler
        return env.action_space.sample() # for now just random action
    
    total_reward = 0
    for i in range(200):
        # make a step given by the controler and record the state and the reward
        observation, reward, _, _ = env.step(my_controler(observation)) 
        total_reward += reward
    
    # Your reward is
    print("Total reward %f" % total_reward)
    
The goal is to maximize total reward.

## Training in rllab

Go to
    
    scripts/rllab/
    
### Training

For training the Arm example with DDPG:

    python experiment.py -e Arm -a DDPG
    
For training the Arm example with TRPO:

    python experiment.py -e Arm -a TRPO

### Test

Show the result

    python visualize.py -p /path/to/params.pkl

## Training in keras-rl

Go to
    
    scripts/keras-rl/

### Training

For training the Arm example (move the arm to certain randomly chosen angles and keep it there):

    python train.ddpg.py --visualize --test --env Arm --output models/Arm
    
For the 'gait' example (walk as far as possible):

    python train.ddpg.py --visualize --test --env Gait --output models/Gait
    
For the 'stand still' example:

    python train.ddpg.py --visualize --test --env Stand --output models/Stand

### Test

For training the Arm example (move the arm to certain randomly chosen angles and keep it there):

    python train.ddpg.py --visualize --train --env Arm
    
and for the gait example (walk as far as possible):

    python train.ddpg.py --visualize --train --env Gait
    
After every 10000 iterations the model is dumped to model_[NUM_ITERATIONS].h5f In ordere to test it run

    python train.ddpg.py --visualize --test --env Gait --output model_[NUM_ITERATIONS]

## Credits

Thanks to @carmichaelong for simplified human model and to OpenSim team for making OpenSim!
