---
title: Advanced features
---

There are two main components of the `osim-rl`:

* `OsimModel` class responsible for low level communication with OpenSim simulator.
* `OsimEnv` class responsible for management of the user interface for running reinforcement learning algorithms.

<div class="note unreleased">
  <h5>Unreleased</h5>
  <p>This page is under construction.</p>
</div>

### Methods of `OsimModel`

#### `list_elements()`

List names of model elements including: joints, bodies, muscles, forces, and markers.

#### `get_body(name)`, `get_joint(name)`, `get_muscle(name)`, `get_marker(name)`, `get_force(name)`

Get an element by name. To better understand what can you do with these entities, please refer to [OpenSim documentation](http://myosin.sourceforge.net/2189/). You can find list of all function under the given class, for example, for muscles please refer to [this file](http://myosin.sourceforge.net/2189/classOpenSim_1_1Muscle.html).

#### `actuate(action)`

Set muscle excitations.

#### `set_activations(action)`

Set muscle activations directly in the state.

#### `get_activations(action)`

Get muscle activations in the current state.

#### `get_state_desc(action)`

Get a dictionary describing the state of joints, bodies, muscles, forces, and markers.

#### `get_state()`

Get the current state of the environment.

#### `set_state(state)`

Set the state to `state`.

#### `integrate()`

Run one step of the simulation.

### Methods of `OsimEnv`

You can access the `OsimModel` associated with `OsimEnv` you can access `env.model`.