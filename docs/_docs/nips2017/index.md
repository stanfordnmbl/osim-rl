---
title: About Learning to run
permalink: /docs/nips2017/
redirect_from: /docs/nips2017/index.html
---

The `osim-rl` package was used as a simulator for an offcial NIPS 2017 Challenge: Learning to run. See the actual challenge website [here](https://www.crowdai.org/challenges/nips-2017-learning-to-run). This documentation contains details about the environment and biomechanics, tips on how to train the model, and solutions of other users.

In the competition, you are tasked with developing a controller to enable a physiologically-based human model to navigate a complex obstacle course as quickly as possible. You are provided with a human musculoskeletal model and a physics-based simulation environment where you can synthesize physically and physiologically accurate motion. Potential obstacles include external obstacles like steps, or a slippery floor, along with internal obstacles like muscle weakness or motor noise. You are scored based on the distance you travel through the obstacle course in a set amount of time.

![HUMAN environment](https://raw.githubusercontent.com/stanfordnmbl/osim-rl/1679344e509e29bdcc2ee368ddf83e868d93bf61/demo/training.gif)

To model physics and biomechanics we use [OpenSim](https://github.com/opensim-org/opensim-core) - a biomechanical physics environment for musculoskeletal simulations.

## Why?

Human movement results from the intricate coordination of muscles, tendons, joints, and other physiological elements. While children learn to walk, run, climb, and jump in their first years of life and most of us can navigate complex environments--like a crowded street or moving subway--without considerable active attention, developing controllers that can efficiently and robustly synthesize realistic human motions in a variety of environments remains a grand challenge for biomechanists, neuroscientists, and computer scientists. Current controllers are confined to a small set of pre-specified movements or driven by torques, rather than the complex muscle actuators found in humans.

Competitiors tackle three major challenges: 1) a high-dimensional action space, 2) complexity of a biological system, including delayed actuation and complex muscle-tendon interactions, and 3) developing a flexible controller for an unseen environment. We advance and popularize an important class of reinforcement learning problems with a large set of output parameters (human muscles) and comparatively small dimensionality of the input (state of a dynamic system). Algorithms developed in the complex biomechanical environment will also generalize to other reinforcement learning settings with highly-dimensional decisions, such as robotics, multivariate decision making (corporate decisions, drug quantities), stock exchange, etc.

