# NeurIPS 2019: Learn to Move - Walk Around

This repository contains software required for participation in the NeurIPS 2019 Challenge: Learn to Move - Walk Around. See more details about the challenge [here](https://www.aicrowd.com/challenges/neurips-2019-learn-to-move-walk-around). See full documentation of our reinforcement learning environment [here](https://osim-rl.stanford.edu). In this document we will give very basic steps to get you set up for the challenge!

Your task is to develop a controller for a physiologically plausible 3D human model to walk or run following velocity commands with minimum effort. You are provided with a human musculoskeletal model and a physics-based simulation environment, OpenSim. There will be three tracks:

1) **Best performance**
2) **Novel ML solution**
3) **Novel biomechanical solution**, where all the winners of each track will be awarded.

To model physics and biomechanics we use [OpenSim](https://github.com/opensim-org/opensim-core) - a biomechanical physics environment for musculoskeletal simulations.

## What's new compared to NIPS 2017: Learning to run?

We took into account comments from the last challenge and there are several changes:

* You can use experimental data (to greatly speed up learning process)
* We released the 3rd dimensions (the model can fall sideways)
* We added a prosthetic leg -- the goal is to solve a medical challenge on modeling how walking will change after getting a prosthesis. Your work can speed up design, prototying, or tuning prosthetics!

You haven't heard of NIPS 2017: Learning to run? [Watch this video!](https://www.youtube.com/watch?v=rhNxt0VccsE)

![HUMAN environment](https://s3.amazonaws.com/osim-rl/videos/running.gif)

## Getting started

**Anaconda** is required to run our simulations. Anaconda will create a virtual environment with all the necessary libraries, to avoid conflicts with libraries in your operating system. You can get anaconda from here https://docs.anaconda.com/anaconda/install/. In the following instructions we assume that Anaconda is successfully installed.

For the challenge we prepared [OpenSim](http://opensim.stanford.edu/) binaries as a conda environment to make the installation straightforward

We support Windows, Linux, and Mac OSX (all in 64-bit). To install our simulator, you first need to create a conda environment with the OpenSim package.

On **Windows**, open a command prompt and type:

    conda create -n opensim-rl -c kidzik -c conda-forge opensim python=3.6.1
    activate opensim-rl
    pip install osim-rl

On **Linux/OSX**, run:

    conda create -n opensim-rl -c kidzik -c conda-forge opensim python=3.6.1
    source activate opensim-rl
    pip install osim-rl

These commands will create a virtual environment on your computer with the necessary simulation libraries installed. If the command `python -c "import opensim"` runs smoothly, you are done! Otherwise, please refer to our [FAQ](http://osim-rl.stanford.edu/docs/faq/) section.

Note that `source activate opensim-rl` activates the anaconda virtual environment. You need to type it every time you open a new terminal.

## Basic usage

To execute 200 iterations of the simulation enter the `python` interpreter and run the following:
```python
from osim.env import L2M2019Env

env = L2M2019Env(visualize=True)
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

## Submission

* Option 1: [**submit solution in docker container**](https://github.com/stanfordnmbl/neurips2019-learning-to-move-starter-kit)
* Option 2: run controller on server environment: [**./examples/submission.py**](https://github.com/stanfordnmbl/osim-rl/blob/master/examples/submission.py)

In order to make a submission to AIcrowd, please refer to [this page](https://github.com/AIcrowd/neurips2019-learning-to-move-starter-kit)

## Rules

Organizers reserve the right to modify challenge rules as required.

## Read more in [the official documentation](http://osim-rl.stanford.edu/)

* [Osim-rl interface](http://osim-rl.stanford.edu/docs/nips2018/interface/)
* [How to train a model?](http://osim-rl.stanford.edu/docs/training/)
* [More on training models](http://osim-rl.stanford.edu/docs/resources/)
* [Experimental data](http://osim-rl.stanford.edu/docs/nips2018/experimental/)
* [Physics underlying the model](http://osim-rl.stanford.edu/docs/nips2017/physics/)
* [Frequently Asked Questions](http://osim-rl.stanford.edu/docs/faq/)
* [Citing and credits](http://osim-rl.stanford.edu/docs/credits/)
* [OpenSim documentation](http://opensim.stanford.edu/)

## Contributions of participants

* [Understanding the Challenge](https://www.endtoend.ai/blog/ai-for-prosthetics-1) - Great materials from [@seungjaeryanlee](https://github.com/seungjaeryanlee/) on how to start

## Partners

<div class="markdown-wrap">
            <a target="_blank" href="https://cloud.google.com/">
              <img class="img-logo" height="50" src="https://dnczkxd1gcfu5.cloudfront.net/images/challenge_partners/image_file/27/google-cloud-logo.png">
</a>            <a target="_blank" href="http://deepmind.com/">
              <img class="img-logo" height="50" src="https://dnczkxd1gcfu5.cloudfront.net/images/challenge_partners/image_file/28/Deep-Mind-Health-WTT-10.05.15.jpg">
</a>            <a target="_blank" href="http://nvidia.com/">
              <img class="img-logo" height="50" src="https://dnczkxd1gcfu5.cloudfront.net/images/challenge_partners/image_file/29/nvidia.png">
</a>            <a target="_blank" href="http://opensim.stanford.edu/about/">
              <img class="img-logo" height="50" src="https://dnczkxd1gcfu5.cloudfront.net/images/challenge_partners/image_file/36/ncsrr.png">
</a>            <a target="_blank" href="https://www.tri.global/">
              <img class="img-logo" height="50" src="https://dnczkxd1gcfu5.cloudfront.net/images/challenge_partners/image_file/37/tri1.png">
</a>        </div>
