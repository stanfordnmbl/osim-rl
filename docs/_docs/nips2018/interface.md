---
title: Interface
---

In order to create an environment, use:
```python
from osim.env import ProstheticsEnv

env = ProstheticsEnv(visualize=True)
```
Parameters:

* `visualize` - turn the visualizer on and off

### Methods of `ProstheticsEnv`

#### `reset(model='3D', prosthetic=True, difficulty=2, seed=None, project = True)`

Restart the enivironment with a given `difficulty` level and a `seed`.

* `difficulty` - For the 3D model: `0` - go forward with sinusoidal change of speed, `1` - sinusoidal change of speed and direction `2` - as in `1` but with stochasticity. For the 2D model the generated vector is projected on the plane in which the model travels (i.e. the Z coordinate is ignored).
* `seed` - starting seed for the random number generator. If the seed is `None`, generation from the previous seed is continued.

Your solution will be graded in the environment with `difficulty = 2`, yet it might be easier to train your model with `difficulty = 0` first and then retrain with a higher difficulty

Returns:

* `observation` - a vector (if `project = True`) or a dictionary describing the state of muscles, joints, and bodies in the biomechanical system.

#### `step(action, project = True)`

Make one iteration of the simulation.

* `action` - a list of length `22` of continuous values in `[0,1]` corresponding to excitation of muscles.

The function returns:

* `observation` - a vector (if `project = True`) or a dictionary describing the state of muscles, joints, and bodies in the biomechanical system.

* `reward` - reward gained in the last iteration.

* `done` - indicates if the move was the last step of the environment. This happens if either `10000` iterations were reached or the pelvis height is below `0.5` meters.

* `info` - for compatibility with OpenAI, currently not used.
