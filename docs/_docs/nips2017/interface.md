---
title: Interface
---

In order to create an environment, use:
```python
from osim.env import L2RunEnv

env = L2RunEnv(visualize = True)
```
Parameters:

* `visualize` - turn the visualizer on and off

### Methods of `L2RunEnv`

#### `reset(difficulty = 2, seed = None, project = True)`

Restart the enivironment with a given `difficulty` level and a `seed`.

* `difficulty` - `0` - no obstacles, `1` - 3 randomly positioned obstacles (balls fixed in the ground), `2` - same as `1` but also strength of the psoas muscles (the muscles that help bend the hip joint in the model) varies. The muscle strength is set to z * 100%, where z is a normal variable with the mean 1 and the standard deviation 0.1
* `seed` - starting seed for the random number generator. If the seed is `None`, generation from the previous seed is continued.

Your solution will be graded in the environment with `difficulty = 2`, yet it might be easier to train your model with `difficulty = 0` first and then retrain with a higher difficulty

Returns

* `observation` - a vector (if `project = True`) or a dictionary describing the state of muscles, joints, and bodies in the biomechanical system.

#### `step(action, project = True)`

Make one iteration of the simulation.

* `action` - a list of length `18` of continuous values in `[0,1]` corresponding to excitation of muscles.

The function returns:

* `observation` - a vector (if `project = True`) or a dictionary describing the state of muscles, joints, and bodies in the biomechanical system. Note that only `project = True` is consistent with the actual NIPS 2017 challenge.

* `reward` - reward gained in the last iteration. The reward is computed as a change in position of the pelvis along the x axis minus the penalty for the use of ligaments. See the "Physics of the model" section for details.

* `done` - indicates if the move was the last step of the environment. This happens if either `1000` iterations were reached or the pelvis height is below `0.65` meters.

* `info` - for compatibility with OpenAI, currently not used.
