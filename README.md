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

## Evaluation

Your task is to build a function `f` which takes the current state `observation` (a dictionary describing the current state) and returns the muscle excitations `action` (19-dimensional vector) maximizing the total reward. The trial ends either if the pelvis of the model falls below `0.6` meters or if you reach `1000` iterations (corresponding to `10` seconds in the virtual environment). 

### Round 1
The objective is to run at a constant speed of 3 meters per second. The total reward is `9 * s - p * p` where `s` is the number of steps before reaching one of the stop criteria and `p` is the absolute difference between horizonal velocity and `3`.
 
### Round 2
In the second round the task is also to follow a requested velocity vector. However, in this round the vector will change in time and it will be a random process. We will provide the distribution of this process in mid-July.

You can test your model on your local machine. For submission, you will need to interact with the remote environment: [crowdAI](https://www.crowdai.org/challenges/nips-2017-learning-to-run) sends you the current `observation` and you need to send back the action you take in the given state. 

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

### Submission

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

## Details of the environment

In order to create an environment, use:
```python
    from osim.env import ProstheticsEnv

    env = ProstheticsEnv(visualize = True)
```
Parameters:

* `visualize` - turn the visualizer on and off

#### `reset(project = True)`

Restart the enivironment.

The function returns:

* `observation` - a vector (if `project = True`) or a dictionary describing the state of muscles, joints, and bodies in the biomechanical system.

#### `step(action, project = True)`

Make one iteration of the simulation.

* `action` - a list of continuous values in `[0,1]` corresponding to excitation of muscles. The length of the vector is expected to be: `22` and `18` for `3D` and `2D` models without the prosthesis; `19` and `15` with a prosthesis.

The function returns:

* `observation` - a vector (if `project = True`) or a dictionary describing the state of muscles, joints, and bodies in the biomechanical system.

* `reward` - reward gained in the last iteration.

* `done` - indicates if the move was the last step of the environment. This happens if either `10000` iterations were reached or the pelvis height is below `0.6` meters.

* `info` - for compatibility with OpenAI, currently not used.

#### `change_model(model='3D', prosthetic=True, difficulty=0, seed=None)`

Change model parameters. Your solution will be graded in the environment with `difficulty = 2, prosthetic = True` and `model = 3D`, yet it might be easier to train a simplified model first (where `model = 2D`, difficulty = 0, prosthetic = False` is the simplest).

* `model` - `3D` model can move in all directions, `2D` one dimension is fixed, i.e. the model cannot fall to the left or right.

* `prosthetic` - if `True` the right leg of the model is a prosthesis.

* `difficulty` - For the 3D model: `0` - go forward at 3 meters per second, (other values not used for now) 

* `seed` - starting seed for the random number generator. If the seed is `None`, generation from the previous seed is continued.

This function does not return any value. `reset()` must be run after changing the model.

### Experimental data

Unlike in our NIPS 2017, now you can use publicly available experimental data to bootstrap the algorithm. Example running and walking kinematics can be found in many publicly available papers. For example we refer to *Schwartz, M. et al. (2018)* (citation details below).

You can find there [joint angles](https://s3.amazonaws.com/osim-rl/data/schwartz2008data/joint_angles.txt) and [EMG signals](https://s3.amazonaws.com/osim-rl/data/schwartz2008data/emg.txt). Data is represented as a function of time in a gait cycle (one step) at different speeds.

You can use it for supervised learning for bootstrapping your models (where for given kinematics you predict muscle activity).

    @article{schwartz2008effect,
      title={The effect of walking speed on the gait of typically developing children},
      author={Schwartz, Michael H and Rozumalski, Adam and Trost, Joyce P},
      journal={Journal of biomechanics},
      volume={41},
      number={8},
      pages={1639--1650},
      year={2008},
      publisher={Elsevier}
    }

### Physics and biomechanics of the model

The model is implemented in [OpenSim](https://github.com/opensim-org/opensim-core)[1], which relies on the [Simbody](https://github.com/simbody/simbody) physics engine. Note that, given recent successes in model-free reinforcement learning, expertise in biomechanics is not required to successfully compete in this challenge.

To summarize briefly, the agent is a musculoskeletal model that include body segments for each leg, a pelvis segment, and a single segment to represent the upper half of the body (trunk, head, arms). The segments are connected with joints (e.g., knee and hip) and the motion of these joints is controlled by the excitation of muscles. The muscles in the model have complex paths (e.g., muscles can cross more than one joint and there are redundant muscles). The muscle actuators themselves are also highly nonlinear. For example, there is a first order differential equation that relates electrical signal the nervous system sends to a muscle (the excitation) to the activation of a muscle (which describes how much force a muscle will actually generate given the muscle's current force-generating capacity). Given the musculoskeletal structure of bones, joint, and muscles, at each step of the simulation (corresponding to 0.01 seconds), the engine:
* computes activations of muscles from the excitations vector provided to the `step()` function,
* actuates muscles according to these activations,
* computes torques generated due to muscle activations,
* computes forces caused by contacting the ground,
* computes velocities and positions of joints and bodies,
* generates a new state based on forces, velocities, and positions of joints.

In each action, the following 22 muscles are actuated (11 per leg):
* hamstrings,
* biceps femoris,
* gluteus maximus,
* iliopsoas,
* rectus femoris,
* vastus,
* gastrocnemius,
* soleus,
* tibialis anterior.
The action vector corresponds to these muscles in the same order (11 muscles of the right leg first, then 11 muscles of the left leg).

The observation contains a dictionary describing the state.

For more details on the simulation framework, please refer to [1]. For more specific information about the muscles model we use, please refer to [2] or to [OpenSim documentation](ysimtk-confluence.stanford.edu:8080/display/OpenSim/Muscle+Model+Theory+and+Publications).

[1] Delp, Scott L., et al. *"OpenSim: open-source software to create and analyze dynamic simulations of movement."* IEEE transactions on biomedical engineering 54.11 (2007): 1940-1950.

[2] Thelen, D.G. *"Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older adults."* ASME Journal of Biomechanical Engineering 125 (2003): 70–77.

## Frequently Asked Questions

**I'm getting 'version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference' error**

If you are getting this error:

    ImportError: /opensim-rl/lib/python2.7/site-packages/opensim/libSimTKcommon.so.3.6:
      symbol _ZTVNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE, version
      GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference

Try `conda install libgcc`.

**Can I use languages other than python?**

Yes, you just need to set up your own python grader and interact with it
https://github.com/kidzik/osim-rl-grader. Find more details here [OpenAI http client](https://github.com/openai/gym-http-api)

**Some libraries are missing. What is required to run the environment?**

Most of the libraries by default exist in major distributions of operating systems or are automatically downloaded by the conda environment. Yet, sometimes things are still missing. The minimal set of dependencies under Linux can be installed with

    sudo apt install libquadmath0 libglu1-mesa libglu1-mesa-dev libsm6 libxi-dev libxmu-dev liblapack-dev

Please, try to find equivalent libraries for your OS and let us know -- we will put them here.

**Why there are no energy constraints?**

Please refer to the issue https://github.com/stanfordnmbl/osim-rl/issues/34.

**I have some memory leaks, what can I do?**

Please refer to
https://github.com/stanfordnmbl/osim-rl/issues/10
and to
https://github.com/stanfordnmbl/osim-rl/issues/58

**How to visualize observations when running simulations on the server?**

Please refer to
https://github.com/stanfordnmbl/osim-rl/issues/59

**I still have more questions, how can I contact you?**

For questions related to the challenge please use [the challenge forum](https://www.crowdai.org/challenges/nips-2017-learning-to-run/topics).
For issues and problems related to installation process or to the implementation of the simulation environment feel free to create an [issue on GitHub](https://github.com/stanfordnmbl/osim-rl/issues).

## Citing

If you use `osim-rl` in your research, you can cite it as follows:

    @incollection{kidzinski2018learningtorun,
      author      = "Kidzi\'nski, {\L}ukasz and Mohanty, Sharada P and Ong, Carmichael and Hicks, Jennifer and Francis, Sean and Levine, Sergey and Salath\'e, Marcel and Delp, Scott",
      title       = "Learning to Run challenge: Synthesizing physiologically accurate motion using deep reinforcement learning",
      editor      = "Escalera, Sergio and Weimer, Markus",
      booktitle   = "NIPS 2017 Competition Book",
      publisher   = "Springer",
      address     = "Springer",
      year        = 2018
    }

If you use the top solutions or other solution reports from the Learning to Run challenge, you can cite them as:

    @incollection{jaskowski2018rltorunfast,
      author      = "Ja\'skowski, Wojciech and Lykkeb{\o}, Odd Rune and Toklu, Nihat Engin and Trifterer, Florian and Buk, Zden\v{e}k and Koutn\'{i}k, Jan and Gomez, Faustino",
      title       = "{Reinforcement Learning to Run... Fast}",
      editor      = "Escalera, Sergio and Weimer, Markus",
      booktitle   = "NIPS 2017 Competition Book",
      publisher   = "Springer",
      address     = "Springer",
      year        = 2018
    }

    @incollection{kidzinski2018l2rsolutions,
      author      = "Kidzi\'nski, {\L}ukasz and Mohanty, Sharada P and Ong, Carmichael and Huang, Zhewei and Zhou, Shuchang and Pechenko, Anton and Stelmaszczyk, Adam and Jarosik, Piotr and Pavlov, Mikhail and Kolesnikov, Sergey and Plis, Sergey and Chen, Zhibo and Zhang, Zhizheng and Chen, Jiale and Shi, Jun and Zheng, Zhuobin and Yuan, Chun and Lin, Zhihui and Michalewski, Henryk and Miłoś, Piotr and Osiński, Błażej and Melnik andrew and Schilling, Malte and Ritter, Helge and Carroll, Sean and Hicks, Jennifer and Levine, Sergey and Salathé, Marcel and Delp, Scott",
      title       = "Learning to run challenge solutions: Adapting reinforcement learning methods for neuromusculoskeletal environments",
      editor      = "Escalera, Sergio and Weimer, Markus",
      booktitle   = "NIPS 2017 Competition Book",
      publisher   = "Springer",
      address     = "Springer",
      year        = 2018
    }

## Credits

This challenge would not be possible without:
* [OpenSim](https://github.com/opensim-org/opensim-core)
* [National Center for Simulation in Rehabilitation Research](http://opensim.stanford.edu/)
* [Mobilize Center](http://mobilize.stanford.edu/)
* [CrowdAI](http://crowdai.org/)
* [OpenAI gym](https://gym.openai.com/)
* [OpenAI http client](https://github.com/openai/gym-http-api)
* [keras-rl](https://github.com/matthiasplappert/keras-rl)
* and many other teams, individuals and projects
