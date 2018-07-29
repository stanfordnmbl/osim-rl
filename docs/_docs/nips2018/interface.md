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

* `done` - indicates if the move was the last step of the environment. This happens if either `300` iterations were reached or the pelvis height is below `0.6` meters.

* `info` - for compatibility with OpenAI, currently not used.

#### `change_model(model='3D', prosthetic=True, difficulty=0,seed=None)`

Change model parameters. Your solution will be graded in the environment with `difficulty = 2, prosthetic = True` and `model = 3D`, yet it might be easier to train a simplified model first (where `model = 2D`, difficulty = 0, prosthetic = False` is the simplest).

* `model` - `3D` model can move in all directions, `2D` one dimension is fixed, i.e. the model cannot fall to the left or right.

* `prosthetic` - if `True` the right leg of the model is a prosthesis.

* `difficulty` - For the 3D model: `0` - go forward at 3 meters per second, (other values not used for now) 

* `seed` - starting seed for the random number generator. If the seed is `None`, generation from the previous seed is continued.

This function does not return any value. `reset()` must be run after changing the model.
