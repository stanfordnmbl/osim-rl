---
title: Interface
---

In order to create an environment, use:
```python
    from osim.env import RunEnv

    env = RunEnv(visualize = True)
```
Parameters:

* `visualize` - turn the visualizer on and off

### Methods of `RunEnv`

#### `reset(difficulty = 2, seed = None)`

Restart the enivironment with a given `difficulty` level and a `seed`.

* `difficulty` - `0` - no obstacles, `1` - 3 randomly positioned obstacles (balls fixed in the ground), `2` - same as `1` but also strength of the psoas muscles (the muscles that help bend the hip joint in the model) varies. The muscle strength is set to z * 100%, where z is a normal variable with the mean 1 and the standard deviation 0.1
* `seed` - starting seed for the random number generator. If the seed is `None`, generation from the previous seed is continued.

Your solution will be graded in the environment with `difficulty = 2`, yet it might be easier to train your model with `difficulty = 0` first and then retrain with a higher difficulty

#### `step(action)`

Make one iteration of the simulation.

* `action` - a list of length `18` of continuous values in `[0,1]` corresponding to excitation of muscles.

The function returns:

* `observation` - a list of length `41` of real values corresponding to the current state of the model. Variables are explained in the section "Physics of the model".

* `reward` - reward gained in the last iteration. The reward is computed as a change in position of the pelvis along the x axis minus the penalty for the use of ligaments. See the "Physics of the model" section for details.

* `done` - indicates if the move was the last step of the environment. This happens if either `1000` iterations were reached or the pelvis height is below `0.65` meters.

* `info` - for compatibility with OpenAI, currently not used.
