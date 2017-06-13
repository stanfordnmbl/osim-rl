# NIPS2017: Learning to run

This repository contains software required for participation in the NIPS 2017 Challenge: Learning to Run. See more details about the challenge [here](https://www.crowdai.org/challenges/nips-2017-learning-to-run).

In this competition, you are tasked with developing a controller to enable a physiologically-based human model to navigate a complex obstacle course as quickly as possible. You are provided with a human musculoskeletal model and a physics-based simulation environment where you can synthesize physically and physiologically accurate motion. Potential obstacles include external obstacles like steps, or a slippery floor, along with internal obstacles like muscle weakness or motor noise. You are scored based on the distance you travel through the obstacle course in a set amount of time.

![HUMAN environment](https://github.com/kidzik/osim-rl/blob/master/demo/training.gif)

For modelling physics we use [OpenSim](https://github.com/opensim-org/opensim-core) - a biomechanical physics environment for musculoskeletal simulations. 

## Getting started

**Anaconda2** is required to run our simulation environment - you can get it from here https://www.continuum.io/downloads; you must choose version 2.7. In the following part we assume that Anaconda is successfully installed.

We support Windows, Linux, and Mac OSX (all in 64-bit). To install our simulator, you first need to create a conda environment with the OpenSim package. On Windows, open a command prompt and type:
    
    conda create -n opensim-rl -c kidzik opensim git
    activate opensim-rl

On Linux/OSX, run:

    conda create -n opensim-rl -c kidzik opensim git
    source activate opensim-rl

These commands will create a virtual environment on your computer with the necessary simulation libraries installed. Next, you need to install our python reinforcement learning environment. Type (on all platforms)

    conda install -c conda-forge lapack git
    pip install git+https://github.com/stanfordnmbl/osim-rl.git

If the command `python -c "import opensim"` runs smoothly, you are done! Otherwise, please refer to our FAQ section.

## Basic usage

To run 200 steps of environment enter the `python` interpreter and run the following:
```python
from osim.env import RunEnv

env = RunEnv(visualize=True)
observation = env.reset(difficulty = 0)
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
```
![Random walk](https://github.com/stanfordnmbl/osim-rl/blob/master/demo/random.gif)

In this example, muscles are activated randomly (red indicates an active muscle and blue an inactive muscle). Clearly with this technique we won't go too far.

Your goal is to construct a controller, i.e. a function from the state space (current positions, velocities and accelerations of joints) to action space (muscle excitations), to go as far as possible in limited time. Suppose you trained a neural network mapping observations (the current state of the model) to actions (muscle excitations), i.e. you have a function `action = my_controller(observation)`, then 
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
There are many ways to construct the function `my_controller(observation)`. We will show how to do it with a DDPG (Deep Deterministic Policy Gradients) algorithm, using `keras-rl`. If you already have experience with training reinforcement learning models, you can skip the next chapter and go to [evaluation](#evaluation)

## Training in keras-rl

Below we present how to train a basic controller using [keras-rl](https://github.com/matthiasplappert/keras-rl). First you need to install extra packages:

    conda install keras -c conda-forge
    pip install git+https://github.com/matthiasplappert/keras-rl.git
    git clone http://github.com/stanfordnmbl/osim-rl.git
    
`keras-rl` is an excelent package compatible with OpenAI, which allows you to quickly build your first models!

Go to the `scripts` subdirectory from this repository
    
    cd osim-rl/scripts

There are two scripts:
* `example.py` for training (and testing) an agent using the DDPG algorithm. 
* `submit.py` for submitting the result to crowdAI.org

### Training

    python example.py --visualize --train --model sample
    
### Test

and for the gait example (walk as far as possible):

    python example.py --visualize --test --model sample

## Evaluation

Your task is to build a function `f` which takes current state `observation` (41 dimensional vector) and returns muscle excitations `action` (18 dimensional vector) in a way that maximizes the reward. See the Section "Details of the environment" for a more precise description.

The trial ends either if the pelvis of the model goes below `0.65` meters or if you reach `500` iterations (corresponding to `5` seconds in the virtual environment). Your total reward is the position of the pelvis on the `x` axis after the last iteration minus a penalty for using ligament forces. Ligaments are tissues which prevent your joints from bending too much - overusing these tissues leads to injuries, so we want to avoid it. The penalty in the total reward is equal to the sum of forces generated by ligaments over the trial, divided by `1000`.

After each iteration you get a reward equal to the change of the `x` axis of pelvis during this iteration minus the magnitude of the ligament forces used in that iteration.

You can test your model on your local machine. For submission, you will need to interact with the remote environment: crowdAI sends you the current `observation` and you need to send back the action you take in the given state. You will be evaluated at three different levels of difficulty. For details, please refer to "Detalis of the environment".

### Submission

Assuming your controller is trained and is represented as a function `my_controller(observation)` returning an `action` you can submit it to crowdAI through interaction with an environment there:

```python
import opensim as osim
from osim.http.client import Client
from osim.env import *

# Settings
remote_base = "http://grader.crowdai.org"
crowdai_token = "[YOUR_CROWD_AI_TOKEN_HERE]"

client = Client(remote_base)

# Create environment
observation = client.env_create(crowdai_token)

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)

# Run a single step
for i in range(1500):
    [observation, reward, done, info] = client.env_step(my_controller(observation), True)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
```

Not that during the submission, the environment will get restarted. Since the environment is stochastic, you will need to submit three trials -- this way we make sure that your model is robust.

### Rules

In order to avoid overfitting to the training environment, top 10 participants will be asked to resubmit their solutions in the private challenge. The final ranking will be based on results from that private phaze.

Additional rules:
* You are not allowed to use external datasets (e.g., kinematics of people walking),
* Organizers reserve the right to modify challenge rules as required.

## Datails of the environment

In order to create an environment, use:
```python
    env = RunEnv(visualize = True)
```
Parameters:

* `visualize` - turn the visualizer on and off

### Methods of `RunEnv`

#### `reset(difficulty = 0, seed = None)`

* `difficulty` - `0` - no obstacles, `1` - 2 randomly positioned obstacles (balls fixed in the ground), `2` - same as `1` but also strength of psoas muscles varies. It is set to z * 100%, where z is a normal variable with the mean 1 and the standard deviation 0.1
* `seed` - starting seed for the random number generator. If the seed is `None`, generation from the previous seed is continued. 

Restart the enivironment with a given `difficulty` level and a `seed`.

#### `step(action)`

* `action` - a list of length `18` of continuous values in `[0,1]` corresponding to excitation of muscles. 

The function returns:

* `observation` - a list of length `41` of real values corresponding to the current state of the model. Variables are explained in the section "Physics of the model".

* `reward` - reward gained in the last iteration. It is computed as a change in position of the pelvis along the x axis minus the penalty for the use of ligaments. See the "Physics of the model" section for details. 

* `done` - indicates if the move was the last step of the environment. This happens if either `500` iterations were reached or the pelvis height is below `0.65` meters.

* `info` - for compatibility with OpenAI, currently not used.

### Physics of the model

The model is implemented in [OpenSim](https://github.com/opensim-org/opensim-core)[1], which relies on the [Simbody](https://github.com/simbody/simbody) physics engine. Note that, given recent successes in model-free reinforcement learning, biomechanical details are not required to successfully compete in this challenge.

In a very brief summary, given the musculoskeletal structure of bones, joint, and muscles, at each step of the simulation (corresponding to 0.01 seconds), the engine:
* computes activations of muscles from the excitations vector provided to the `step()` function,
* actuates muscles according to these activations,
* computes torques generated due to mucsle activations,
* computes forces caused by contacting the ground,
* computes velocities and positions of joints and bodies,
* generates a new state based on forces, velcities, and positions of joints.

In each action, the following 18 muscles are actuated (9 per leg):
* hamstrings,
* biceps femoris,
* gluteus maximus,
* iliopsoas,
* rectus femoris,
* vastus,
* gastrocnemius,
* soleus,
* tibialis anterior.
The action vector corresponds to these muscles in the same order (9 muscles of the right leg first, then 9 muscles of the left leg).

The observation contains 41 values:
* position of the pelvis (rotation, x, y)
* velocity of the pelvis (rotation, x, y)
* rotation of each ankle, knee and hip (6 values)
* angular velocity of each ankle, knee and hip (6 values)
* position of the center of mass (2 values)
* velocity of the center of mass (2 values)
* positions (x, y) of head, pelvis, torso, left and right toes, left and right talus (14 values)
* strength of left and right psoas: 1 for `difficulty < 2`, otherwise a random normal variable with mean 1 and standard deviation 0.1 fixed for the entire simulation
* next obstacle: x distance from the pelvis, y position of the center relative to the the ground, radius.

For more details on the actual simulation, please refer to [1].

[1] Delp, Scott L., et al. *"OpenSim: open-source software to create and analyze dynamic simulations of movement."* IEEE transactions on biomedical engineering 54.11 (2007): 1940-1950.

## Frequently Asked Questions (FAQ)

**I'm getting 'version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference' error**

If you are getting this error:

    ImportError: /opensim-rl/lib/python2.7/site-packages/opensim/libSimTKcommon.so.3.6:
      symbol _ZTVNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE, version
      GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference
    
Try `conda install libgcc`.

**Can I use languages other than python?**

Yes, you just need to set up your own python grader and interact with it
https://github.com/kidzik/osim-rl-grader. Find more details here [OpenAI http client](https://github.com/openai/gym-http-api)

**Do you have a docker container?**

Yes, you can use https://hub.docker.com/r/stanfordnmbl/opensim-rl/
Note, that connecting a display to a docker can be tricky and it's system dependent. Nevertheless, for training your models the display is not necessary -- the docker container can be handy for using multiple machines.

**Some libraries are missing. What is required to run the environment?**

Most of the libraries by dafoult exist in major distributions of operating systems or are automatically downloaded by the conda environment. Yet, sometimes things are still missing. The minimal set of dependencies under Linux can be installed with

    sudo apt install libquadmath0 libglu1-mesa libglu1-mesa-dev libsm6 libxi-dev libxmu-dev liblapack-dev

Please, try to find equivalent libraries for your OS and let us know -- we will put them here.

## Credits

This challenge would not be possible without:
* [OpenSim](https://github.com/opensim-org/opensim-core)
* Stanford NMBL group & Stanford Mobilize Center
* [OpenAI gym](https://gym.openai.com/)
* [OpenAI http client](https://github.com/openai/gym-http-api)
* [keras-rl](https://github.com/matthiasplappert/keras-rl)
* and many other teams, individuals and projects

For details please contact [Łukasz Kidziński](http://kidzinski.com/)
