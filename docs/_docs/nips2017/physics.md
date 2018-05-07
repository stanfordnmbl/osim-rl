---
title: Biomechanics
---

The model is implemented in [OpenSim](https://github.com/opensim-org/opensim-core)[1], which relies on the [Simbody](https://github.com/simbody/simbody) physics engine. Note that, given recent successes in model-free reinforcement learning, expertise in biomechanics is not required to successfully compete in this challenge.

To summarize briefly, the agent is a musculoskeletal model that include body segments for each leg, a pelvis segment, and a single segment to represent the upper half of the body (trunk, head, arms). The segments are connected with joints (e.g., knee and hip) and the motion of these joints is controlled by the excitation of muscles. The muscles in the model have complex paths (e.g., muscles can cross more than one joint and there are redundant muscles). The muscle actuators themselves are also highly nonlinear. For example, there is a first order differential equation that relates electrical signal the nervous system sends to a muscle (the excitation) to the activation of a muscle (which describes how much force a muscle will actually generate given the muscle's current force-generating capacity). Given the musculoskeletal structure of bones, joint, and muscles, at each step of the simulation (corresponding to 0.01 seconds), the engine:
* computes activations of muscles from the excitations vector provided to the `step()` function,
* actuates muscles according to these activations,
* computes torques generated due to muscle activations,
* computes forces caused by contacting the ground,
* computes velocities and positions of joints and bodies,
* generates a new state based on forces, velocities, and positions of joints.

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

For more details on the simulation framework, please refer to [1]. For more specific information about the muscles model we use, please refer to [2] or to [OpenSim documentation](ysimtk-confluence.stanford.edu:8080/display/OpenSim/Muscle+Model+Theory+and+Publications).

[1] Delp, Scott L., et al. *"OpenSim: open-source software to create and analyze dynamic simulations of movement."* IEEE transactions on biomedical engineering 54.11 (2007): 1940-1950.

[2] Thelen, D.G. *"Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older adults."* ASME Journal of Biomechanical Engineering 125 (2003): 70–77.

