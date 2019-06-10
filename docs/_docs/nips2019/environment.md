---
title: Environment
---

<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

Your task is to develop a controller for a physiologically plausible 3D human model to move (walk or run) following velocity commands with minimum effort.
Formally, you build a policy \\(\pi:V \times S \rightarrow A\\) from the body state \\(S\\) and target velocity map \\(V\\) to action \\(A\\). The performance of your policy will be evaluated by the cumulative reward \\(J(\pi)\\) it receives from simulation.

## Human Model
...todo

## Reward

A simulation runs until either the pelvis of the human model falls below \\(0.6\\) meters or when it reaches \\(10\\) seconds (\\(i=1000\\)).
During the simulation, you receive a survival reward every timestep \\(i\\) and a footstep reward whenever the human model makes a new footstep \\(step_i\\).
The reward is designed so that the total reward \\(J(\pi)\\) is high when the human model locomotes at desired velocities with minimum effort.

$$ J(\pi) = R_{alive} + R_{step} \\
= \sum_{i}^{} r_{alive} +\sum_{step_i}^{} (w_{step}r_{step} - w_{vel}c_{vel}  - w_{eff}c_{eff} ). $$

The footstep reward \\(R_{step}\\) is designed to evaluate step behaviors rather than instantaneous behaviors, for example, to allow the human model's walking speed to vary within a footstep as real humans do. Specifically, the rewards and costs are defined as

$$ r_{alive} = 0.1 \\
r_{step} = \sum_{i \text{ in } step_i}^{} \Delta t_i = \Delta t_{step_i} \\
c_{vel} = \Vert \sum_{i \text{ in } step_i}^{} (v_{pel} - v_{tgt})\Delta t_i \Vert \\
c_{eff} = \sum_{i \text{ in } step_i}^{} \sum_{m}^{muscles} A_m^2 \Delta t_i. $$

\\(\Delta t_{i}=0.01\\) sec is the simulation timestep, \\(v_{pel}\\) is the velocity of the pelvis, \\(v_{tgt}\\) is the target velocity, \\(A_{m}\\)s are the muscle activations, and \\(w_{step}\\), \\(w_{vel}\\) and \\(w_{eff}\\) are the weights for the stepping reward and velocity and effort costs.

## Observation (input to your controller)

The observation or the input to your controller consists of a local target velociy map \\(V\\) and the body state \\(S\\).

\\(V\\) is a \\( 2 \times 11 \times 11 \times \\) matrix, representing a \\(2D\\) vector field on an \\(11 \times 11 \\) grid. The \\(2D\\) vectors are target velocities, and the \\(11 \times 11 \\) grid is for every \\(0.5 \\) meter within \\(\pm 5\\) meters back-to-front and left-to-side. For example, in the figure below, the global target velocity map (top-left) shows that the velocity field converges to \\((2.9, 5.7)\\) and the human model is at \\((3.5, 0.0)\\) (end of the black line).
Thus, the local target velocity map (bottom-left) shows that the human model should locomote to the right as the vector points towards the right (i.e. close to \\([0, 1]\\)).

\\(S\\) is a \\(97D\\) vector representing the body state.
It consists of pelvis state, ground reaction forces joint angles and rates and muscle states.
The keys of the observation dictionary should be self evident what the values represent.

## Action (output of your controller)

The action space \\([0,1]^{22}\\) represents muscle activations of the (\\22\\) muscles, 11 per leg.
