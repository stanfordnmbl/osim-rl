# NIPS2018: AI for prosthetics

This repository contains software required for participation in the NIPS 2018 Challenge: AI for prosthetics. See more details about the challenge [here](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge). See full documentation of our reinforcement learning environment [here](https://osim-rl.stanford.edu). In this document we will give very basic steps to get you set up for the challenge!

In this competition, you are tasked with developing a controller to enable a physiologically-based human model with a prosthetic leg to walk and run. You are provided with a human musculoskeletal model, a physics-based simulation environment where you can synthesize physically and physiologically accurate motion, and datasets of normal gait kinematics. You are scored based on how well your agent adapts to requested velocity vector changing in real time.

[![AI for prosthetics](https://s3-eu-west-1.amazonaws.com/kidzinski/nips-challenge/images/ai-prosthetics.jpg)](https://github.com/stanfordnmbl/osim-rl)

To model physics and biomechanics we use [OpenSim](https://github.com/opensim-org/opensim-core) - a biomechanical physics environment for musculoskeletal simulations.

## What's new compared to NIPS 2017: Learning to run?

We took into account comments from the last challenge and there are several changes:

* You can use experimental data (to greatly speed up learning process)
* We released the 3rd dimensions (the model can fall sideways)
* We added a prosthetic leg -- the goal is to solve a medical challenge on modeling how walking will change after getting a prosthesis. Your work can speed up design, prototying, or tuning prosthetics!

You haven't heard of NIPS 2017: Learning to run? [Watch this video!](https://www.youtube.com/watch?v=rhNxt0VccsE)

![HUMAN environment](https://s3.amazonaws.com/osim-rl/videos/running.gif)

## Getting started

**Anaconda** is required to run our simulations. Anaconda will create a virtual environment with all the necessary libraries, to avoid conflicts with libraries in your operating system. You can get anaconda from here https://www.continuum.io/downloads. In the following instructions we assume that Anaconda is successfully installed.

We support Windows, Linux, and Mac OSX (all in 64-bit). To install our simulator, you first need to create a conda environment with the OpenSim package.

On **Windows**, open a command prompt and type:

    conda create -n opensim-rl -c kidzik opensim python=3.6.1
    activate opensim-rl

On **Linux/OSX**, run:

    conda create -n opensim-rl -c kidzik opensim python=3.6.1
    source activate opensim-rl

These commands will create a virtual environment on your computer with the necessary simulation libraries installed. Next, you need to install our python reinforcement learning environment. Type (on all platforms):

    conda install -c conda-forge lapack git
    pip install git+https://github.com/stanfordnmbl/osim-rl.git

If the command `python -c "import opensim"` runs smoothly, you are done! Otherwise, please refer to our [FAQ](#frequently-asked-questions) section.

Note that `source activate opensim-rl` activates the anaconda virtual environment. You need to type it every time you open a new terminal.

## Basic usage

To execute 200 iterations of the simulation enter the `python` interpreter and run the following:
```python
from osim.env import ProstheticsEnv

env = ProstheticsEnv(visualize=True)
observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
```
![Random walk](https://raw.githubusercontent.com/stanfordnmbl/osim-rl/1679344e509e29bdcc2ee368ddf83e868d93bf61/demo/random.gif)

The function `env.action_space.sample()` returns a random vector for muscle activations, so, in this example, muscles are activated randomly (red indicates an active muscle and blue an inactive muscle).  Clearly with this technique we won't go too far.

Your goal is to construct a controller, i.e. a function from the state space (current positions, velocities and accelerations of joints) to action space (muscle excitations), that will enable to model to travel as far as possible in a fixed amount of time. Suppose you trained a neural network mapping observations (the current state of the model) to actions (muscle excitations), i.e. you have a function `action = my_controller(observation)`, then
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

You can find details about the [observation object here](http://osim-rl.stanford.edu/docs/nips2018/observation/).

## Evaluation

Your task is to build a function `f` which takes the current state `observation` (a dictionary describing the current state) and returns the muscle excitations `action` (19-dimensional vector) maximizing the total reward. The trial ends either if the pelvis of the model falls below `0.6` meters or if you reach `1000` iterations (corresponding to `10` seconds in the virtual environment). 

### Round 1
The objective is to run at a constant speed of 3 meters per second. The total reward is `9 * s - p * p` where `s` is the number of steps before reaching one of the stop criteria and `p` is the absolute difference between horizonal velocity and `3`.
 
### Round 2
In the second round the task is also to follow a requested velocity vector. However, in this round the vector will change in time and it will be a random process. We will provide the distribution of this process in mid-July.

You can test your model on your local machine. For submission, you will need to interact with the remote environment: [crowdAI](https://www.crowdai.org/challenges/nips-2017-learning-to-run) sends you the current `observation` and you need to send back the action you take in the given state. 

Read more about [evaluation here](http://osim-rl.stanford.edu/docs/nips2018/evaluation/).

## Submission

Assuming your controller is trained and is represented as a function `my_controller(observation)` returning an `action` you can submit it to [crowdAI](https://www.crowdai.org/challenges/nips-2017-learning-to-run) through interaction with an environment there:

```python
import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "[YOUR_CROWD_AI_TOKEN_HERE]"

client = Client(remote_base)

# Create environment
observation = client.env_create(crowdai_token)

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)

while True:
    [observation, reward, done, info] = client.env_step(my_controller(observation), True)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
```

In the place of `[YOUR_CROWD_AI_TOKEN_HERE]` put your token from the profile page from [crowdai.org](http://crowdai.org/) website.

### Rules

In order to avoid overfitting to the training environment, the top participants will be asked to resubmit their solutions in the second round of the challenge. The final ranking will be based on the results from the second round.

Additional rules:

* Organizers reserve the right to modify challenge rules as required.

## Training your models

We suggest you start from [reviewing solutions from the last year](http://osim-rl.stanford.edu/docs/nips2017/solutions/).

You can find many tutorials, frameworks and lessons on-line. We particularly recommend:

Tutorials & Courses on Reinforcement Learning:
* [Berkeley Deep RL course by Sergey Levine](http://rll.berkeley.edu/deeprlcourse/)
* [Intro to RL on Karpathy's blog](http://karpathy.github.io/2016/05/31/rl/)
* [Intro to RL by Tambet Matiisen](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
* [Deep RL course of David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLHOg3HfW_teiYiq8yndRVwQ95LLPVUDJe)
* [A comprehensive list of deep RL resources](https://github.com/dennybritz/reinforcement-learning)

Frameworks and implementations of algorithms:
* [OpenAI baselines](https://github.com/openai/baselines)
* [RLLAB](https://github.com/openai/rllab)
* [modular_rl](https://github.com/joschu/modular_rl)
* [keras-rl](https://github.com/matthiasplappert/keras-rl)

OpenSim and Biomechanics:
* [OpenSim Documentation](http://simtk-confluence.stanford.edu:8080/display/OpenSim/OpenSim+Documentation)
* [Muscle models](http://simtk-confluence.stanford.edu:8080/display/OpenSim/First-Order+Activation+Dynamics)
* [Publication describing OpenSim](http://nmbl.stanford.edu/publications/pdf/Delp2007.pdf)
* [Publication describing Simbody (multibody dynamics engine)](http://ac.els-cdn.com/S2210983811000241/1-s2.0-S2210983811000241-main.pdf?_tid=c22ea7d2-50ba-11e7-9f69-00000aacb361&acdnat=1497415051_124f3094c7fec3c60165f5d544a184f4)

This list is *by no means* exhaustive. If you find some resources particularly well-fit for this tutorial, please let us know!

## Read more in [the official documentation](http://osim-rl.stanford.edu/)

* [Osim-rl interface](http://osim-rl.stanford.edu/docs/nips2018/interface/)
* [Experimental data](http://osim-rl.stanford.edu/docs/nips2018/experimental/)
* [Physics underlying the model](http://osim-rl.stanford.edu/docs/nips2017/physics/)
* [Frequently Asked Questions](http://osim-rl.stanford.edu/docs/faq/)
* [Citing and credits](http://osim-rl.stanford.edu/docs/credits/)
