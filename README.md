# NIPS2017: Learning to run

In this competition, you are tasked with developing a controller to enable a physiologically-based human model to navigate a complex obstacle course as quickly as possible. You are provided with a human musculoskeletal model and a physics-based simulation environment where you can synthesize physically and physiologically accurate motion. Potential obstacles include external obstacles like steps, or a slippery floor, along with internal obstacles like muscle weakness or motor noise. You are scored based on the distance you travel through the obstacle course in a set amount of time.

![HUMAN environment](https://github.com/kidzik/osim-rl/blob/master/demo/training.gif)

For modelling physics we use [OpenSim](https://github.com/opensim-org/opensim-core) - a biomechanical physics environment for musculoskeletal simulations. 

## Getting started

**Anaconda2** is required to run our simulation environment - you can get it from here https://www.continuum.io/downloads choosing version 2.7. In the following part we assume that Anaconda is successfully installed.

We support Windows, Linux or Mac OSX in 64-bit version. To install our simulator, you first need to create a conda environment with OpenSim package. On Windows open a command prompt and type:
    
    conda create -n opensim-rl -c kidzik opensim git
    activate opensim-rl

on Linux/OSX run:

    conda create -n opensim-rl -c kidzik opensim git
    source activate opensim-rl

These command will create a virtual environment on your computer with simulation libraries installed. Next, you need to install our python reinforcement learning environment. Type (independently on the system)

    conda install -c conda-forge lapack git
    pip install git+https://github.com/stanfordnmbl/osim-rl.git

If the command `python -c "import opensim"` runs smoothly you are done! Otherwise, please refer to our FAQ section.

## Basic usage

To run 200 steps of environment enter `python` interpreter and run:

    from osim.env import RunEnv

    env = RunEnv(visualize=True)
    observation = env.reset()
    for i in range(200):
        observation, reward, done, info = env.step(env.action_space.sample())

![Random walk](https://github.com/stanfordnmbl/osim-rl/blob/master/demo/random.gif)

In this example muscles are activated randomly (red color indicates an active muscle and blue an inactive muscle). Clearly with this technique we won't go to far.

Your goal is to construct a controler, i.e. a function from the state space (current positions, velocities and accelerations of joints) to action space (muscles activations), to go as far as possible in limited time. Suppose you trained a neural network mapping observations (the current state of the model) to actions (muscles activations), i.e. you have a function `action = my_controler(observation)`, then 

    # ...
    total_reward = 0.0
    for i in range(200):
        # make a step given by the controler and record the state and the reward
        observation, reward, done, info = env.step(my_controler(observation)) 
        total_reward += reward
        if done:
            break
    
    # Your reward is
    print("Total reward %f" % total_reward)

There are many ways to construct the function `my_controler(observation)`. We will show how to do it with a DDPG algorithm, using keras-rl.

## Evaluation

Your task is to build a function `f` which takes current state `observation` (25 dimensional vector) and returns mouscle activations `action` (16 dimensional vector) in a way that maximizes the reward.

The trial ends either if the pelvis of the model goes below `0.7` meter or if you reach `500` iterations (corresponding to `5` seconds in the virtual environment). Let `N` be the length of the trial. Your total reward is simply the position of the pelvis on the `x` axis after `N` steps. The value is given in centimeters.

After each iteration you get a reward equal to the change of the `x` axis of pelvis during this iteration.

You can test your model on your local machine. For submission, you will need to interact with the remote environment: crowdAI sends you the current `observation` and you need to send back the action you take in the given state.

### Submission

After having trained your model you can submit it by modifying the `/scripts/submit.py` (see the comments in the file for details) and executing

    python submit.py

This script will interact with an environment on the crowdAI.org server.

### Rules

You are not allowed to:
* Use external datasets (ex. kinematics of people walking)

Other:
* Organizers reserve the right to modify challenge rules as required.

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
    
## Datails of the environment

### Functions

### Physics of the model

## Questions

**I'm getting 'version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference' error**

If you are getting this error

    ImportError: /home/deepart/anaconda2/envs/opensim-rl/lib/python2.7/site-packages/opensim/../../../libSimTKcommon.so.3.6: symbol _ZTVNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE, version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference
    
Try `conda install libgcc`

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
