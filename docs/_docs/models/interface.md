---
title: Basic interface
---

[All environments](/docs/models/) share two fundamental functions `reset()` and `step(action)`. `reset` restarts the environment to the initial state. `step` sends muscle excitations and runs the simulation for one step. These two functions are the minimal requirement for most of the reinforcement learning algorithms.

### Initialization

You can see all available environments in [this document](/docs/models/). Here we describe how to initialize and run the most basic one `Arm2DEnv`

In order to create an environment, use:
```python
from osim.env import Arm2DEnv
env = Arm2DEnv(visualize = True)
```

Parameters:

* `visualize` - turn the visualizer on and off

### Methods

#### `reset(project = True)`

Restart the environment to the initial state. Note that extra parameters can be available depending on the [environment](/docs/models/).

The function returns:

* `observation` - a vector (if `project = True`) or a dictionary describing the state of muscles, joints, and bodies in the biomechanical system.

#### `step(action, project = True)`

Make one iteration of the simulation.

* `action` - a list of numbers in the `[0,1]` interval, corresponding to excitations of muscles exposed in the environment.

The function returns:

* `observation` - a vector (if `project = True`) or a dictionary describing the state of muscles, joints, and bodies in the biomechanical system.

* `reward` - reward gained in the last iteration.

* `done` - indicates if the move was the last step of the environment.

* `info` - for compatibility with [OpenAI gym](https://github.com/openai/gym), currently not used.
