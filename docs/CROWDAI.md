# Learning how to walk

Our movement originates in the brain. Many neurological disorders, such as Cerebral Palsy, Multiple Sclerosis or strokes can lead to problems with walking. Treatments are often symptomatic and its often hard to predict outcomes of surgeries. Understanding underlying mechanisms is key to improvement of treatments.

In this challenge your task is to model the motor control unit in human brain. You are given a musculoskeletal model with 16 muscles to control. At every 10ms you send signals to these muscles to activate or deactivate them. The objective is to walk as far as possible in 5 seconds.

For modelling physics we use [OpenSim](https://github.com/opensim-org/opensim-core) - a biomechanical physics environment for musculoskeletal simulations. 

![HUMAN environment](https://github.com/kidzik/osim-rl/blob/master/demo/training.gif)

## Evaluation

Your task is to build a function `f` which takes current state `observation` (25 dimensional vector) and returns mouscle activations `action` (16 dimensional vector) in a way that maximizes the reward.

The trial ends either if the pelvis of the model goes below `0.7` meter or if you reach `500` iterations (corresponding to `5` seconds in the virtual environment). Let `N` be the length of the trial. Your total reward is simply the position of the pelvis on the `x` axis after `N` steps. The value is given in centimeters.

After each iteration you get a reward equal to the change of the `x` axis of pelvis during this iteration.

You can test your model on your local machine. For submission, you will need to interact with the remote environment: crowdAI sends you the current `observation` and you need to send back the action you take in the given state.

### Rules

You are allowed to:
* Modify objective function for training (eg. extra penalty for falling or moving to fast, reward keeping head at the same level, etc.), by 
* Modify the musculoskeletal model for training (eg. constrain the Y axis of pelvis)
* Submit a maximum of one submissions each 6 hours.

Note, that the model trained in your modified environment must still be compatible with the challenge environment. 

You are not allowed to:
* Use external datasets (ex. kinematics of people walking)
* Engineer the trajectories/muscle activations by hand

Other:
* crowdAI reserves the right to modify challenge rules as required.

## Installation

Requires OpenSim 4.0 - https://github.com/opensim-org/opensim-core . You can either install it from source (https://github.com/opensim-org/opensim-core/releases/tag/v4.0.0_alpha) or use conda builds as presented below.

**Requires Anaconda2**, you can get it from here https://www.continuum.io/downloads choosing version 2.7.
Below we assume that Anaconda is installed.

For the moment we only support 64-bit architecture (32-bit coming soon) on either Windows, Linux or Mac OSX. On Windows open a command prompt and type:
    
    conda create -n opensim-rl -c kidzik opensim
    activate opensim-rl

on Linux/OSX run:

    conda create -n opensim-rl -c kidzik opensim
    source activate opensim-rl

Then on any system you can install the RL environment with

    conda install -c conda-forge lapack git
    pip install git+https://github.com/kidzik/osim-rl.git

If the command `python -c "import opensim"` runs smoothly you are done! If you encounter this error

    ImportError: /home/deepart/anaconda2/envs/opensim-rl/lib/python2.7/site-packages/opensim/../../../libSimTKcommon.so.3.6: symbol _ZTVNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE, version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference
    
Try `conda install libgcc`

## Basic usage

To run 200 steps of environment enter `python` interpreter and run:

    from osim.env import GaitEnv

    env = GaitEnv(visualize=True)
    observation = env.reset()
    for i in range(500):
        observation, reward, done, info = env.step(env.action_space.sample())

![Random walk](https://github.com/stanfordnmbl/osim-rl/blob/master/demo/random.gif)

In this example muscles are activated randomly (red color indicates an active muscle and blue an inactive muscle). Clearly with this technique we won't go to far.

Your goal is to construct a controler, i.e. a function from the state space (current positions, velocities and accelerations of joints) to action space (muscles activations), to go as far as possible in limited time. Suppose you trained a neural network mapping observations (the current state of the model) to actions (muscles activations), i.e. you have a function `action = my_controler(observation)`, then 

    # ...
    total_reward = 0.0
    for i in range(500):
        # make a step given by the controler and record the state and the reward
        observation, reward, done, info = env.step(my_controler(observation)) 
        total_reward += reward
    
    # Your reward is
    print("Total reward %f" % total_reward)

There are many ways to construct the function `my_controler(observation)`. We will show how to do it with a DDPG algorithm, using keras-rl.

## Training in keras-rl

Below we present how to train a basic controller using [keras-rl](https://github.com/matthiasplappert/keras-rl). First you need to install extra packages

    conda install keras
    pip install git+https://github.com/matthiasplappert/keras-rl.git
    git clone http://github.com/stanfordnmbl/osim-rl.git
    
`keras-rl` is an excelent package compatible with OpenAi, which allows you to quickly build your first models!

Go to `scripts` subdirectory from this repository
    
    cd osim-rl/scripts

There are two scripts:
* `example.py` for training (and testing) an agent using DDPG algorithm. 
* `submit.py` for submitting the result to crowdAI.org

### Training

    python example.py --visualize --train --model sample
    
### Test

and for the gait example (walk as far as possible):

    python example.py --visualize --test --model sample

### Submission

After having trained your model you can submit it using the following script

    python submit.py --model sample

This script will interact with an environment on the crowdAI.org server.

## Questions

**Can I use different languages than python?**

Yes, you just need to set up your own python grader and interact with it
https://github.com/kidzik/osim-rl-grader. Find more details here [OpenAI http client](https://github.com/openai/gym-http-api)

## Credits

This challenge wouldn't be possible without:
* [OpenSim](https://github.com/opensim-org/opensim-core)
* Stanford NMBL group & Stanford Mobilize Center
* [OpenAI gym](https://gym.openai.com/)
* [OpenAI http client](https://github.com/openai/gym-http-api)
* [keras-rl](https://github.com/matthiasplappert/keras-rl)
* and many other teams, individuals and projects

For details please contact [Łukasz Kidziński](http://kidzinski.com/)
