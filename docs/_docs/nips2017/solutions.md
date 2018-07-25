---
title: Solutions
---

<table>
<caption style="font-size: 0.8em">A footage from the winning solution by NNAISENSE</caption>
<div style="width: 100%; padding-bottom: 75%; position: relative;">
<video style="position: absolute; top: 0;	left: 0;	width: 100%;	height: 100%;" controls="controls">
  <source src="https://s3.amazonaws.com/osim-rl/videos/01-nnaisense.mp4">
</video>
</div>
</table>

All participants whose models traveled at least 15 meters in 10 seconds of the simulator time were invited to share their solutions in this manuscript. Nine teams agreed to contribute papers. The winning algorithm is published separately (to appear), while the remaining eight are collected [here](https://arxiv.org/pdf/1804.00361.pdf). On this page, we present solutions of teams who released videos describing their solutions.

### Tips and tricks

We identified multiple strategies shared across teams.

#### Speeding up OpenSim
* *Parallelization:* run multiple simulations with different parameters on multiple CPUs.
* *Accuracy:* in OpenSim, the accuracy of the integrator is parametrized and can be manually set before the simulation. Users reduced accuracy to have faster simulations.

#### Speeding up exploration
* *Frameskip:* instead of sending signals every 1/100 of a second, keep the same control for, for example, 5 frames.
* *Symmetry:* assume that given a mirrored environment you should take a mirrored action.
* *Binary actions:* excitations 0 or 1 instead of values in the interval [0,1].
* *Sample efficiency:* most teams used learning algorithms leveraging history, such as [Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1509.02971.pdf).

#### Leveraging the model
* *Ground reaction forces:* these were not given in the environment so users tried to estimate them.
* *Current muscle activity:* this was also not given but can be estimated following the OpenSim muscles dynamics model.
* *Reward shaping:* modifying the reward for training in such a way that it still makes the model train faster for the actual initial reward. E.g. reward in the challenge is to run as quickly as possible; one can add an extra penalty term for falling (it seems it’s easier to first learn not to fall and then to run, rather than just learn to run).

## NNAISENSE (1st place)

[Github repository](https://github.com/nnaisense/2017-learning-to-run)

*Authors:* <a href="https://github.com/wjaskowski" class="post-author">{% avatar wjaskowski size=30 %}</a>
<a href="https://github.com/nnaisense" class="post-author">{% avatar nnaisense size=30 %}</a>



## PKU (2nd place)

PKU team used an Actor-Critic Ensemble (ACE) method for improving the performance of Deep Deterministic Policy Gradient (DDPG) algorithm. At inference time, their method uses a critic ensemble to select the best action from proposals of multiple actors running in parallel. By having a larger candidate set, their method can avoid actions that have fatal consequences, while staying deterministic. 

<div style="width: 100%; padding-bottom: 58%; position: relative;">
<video style="position: absolute; top: 0;	left: 0;	width: 100%;	height: 100%;" controls="controls">
  <source src="https://s3.amazonaws.com/osim-rl/videos/02-pku.mp4">
</video>
</div>

[Github repository](https://github.com/hzwer/NIPS2017-LearningToRun)

*Authors:* <a href="https://github.com/hzwer" class="post-author">{% avatar hzwer size=30 %}</a>
<a href="https://github.com/NewGod" class="post-author">{% avatar NewGod size=30 %}</a>
<a href="https://github.com/liu-jc" class="post-author">{% avatar liu-jc size=30 %}</a>

## reason8.ai (3rd place)

Reason8 taem benchmarked state of the art policy-gradient methods and concluded that Deep Deterministic Policy Gradient (DDPG) method is the most efficient method for this environment. They also applied several improvements to DDPG method, such as layer normalization, parameter noise, action and state reflecting. All this improvements helped to stabilize training and improve its sample-efficiency.

<div style="width: 100%; padding-bottom: 58%; position: relative;">
<iframe style="position: absolute; top: 0;	left: 0;	width: 100%;	height: 100%;"  src="https://www.youtube.com/embed/y-fmlotLL3o" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
</div>

Reason8.ai team provides to implementations on GitHub: [Theano](https://github.com/fgvbrt/nips_rl) and [PyTorch](https://github.com/Scitator/Run-Skeleton-Run).

*Authors:* <a href="https://github.com/fgvbrt" class="post-author">{% avatar fgvbrt size=30 %}</a>
<a href="https://github.com/Scitator" class="post-author">{% avatar Scitator size=30 %}</a>
<a href="https://github.com/wassname" class="post-author">{% avatar wassname size=30 %}</a>

## IMCL (4th place)

For improving the training effectiveness of DDPG on this physics-based simulation environment which has high computational complexity, ICML team designed a parallel architecture with deep residual network for the asynchronous training of DDPG.

<div style="width: 100%; padding-bottom: 58%; position: relative;">
<video style="position: absolute; top: 0;	left: 0;	width: 100%;	height: 100%;" controls="controls">
  <source src="https://s3.amazonaws.com/osim-rl/videos/04-imcl-new.mp4" type="video/mp4">
</video>
</div>

## deepsense.ai (6th place)

Deepsense.ai solution was based on the distributed Proximal Policy Optimization algorithm combined with a few efficiency-improving techniques. They used *frameskip* to increase exploration. They changed rewards to encourage the agent to \textit{bend its knees}, which significantly stabilized the gait and accelerated the training. In the final stage, they found it beneficial to transfer skills from  small networks (easier to train) to bigger ones (with more expressive power). They developed *policy blending*, a general cloning/transferring technique.

<div style="width: 100%; padding-bottom: 58%; position: relative;">
<video style="position: absolute; top: 0;	left: 0;	width: 100%;	height: 100%;" controls="controls">
  <source src="https://s3.amazonaws.com/osim-rl/videos/06-deepsense.mp4">
</video>
</div>

## Adam Melnik (22nd place)

Team of Adam Melnik trained the final model with PPO on 80 cores in 5 days using *reward shaping* with a normalized observation vector.

<div style="width: 100%; padding-bottom: 75%; position: relative;">
<video style="position: absolute; top: 0;	left: 0;	width: 100%;	height: 100%;" controls="controls">
  <source src="https://s3.amazonaws.com/osim-rl/videos/MELNIK.mp4">
</video>
</div>

## Other materials

A Medium article from [{% avatar AdamStelmaszczyk size=30 %} Adam Stelmaszczyk](https://medium.com/mlreview/our-nips-2017-learning-to-run-approach-b80a295d3bb5).
