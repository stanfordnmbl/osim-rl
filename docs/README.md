# Version 1.4.1

After discussing the way the reward function is computed ( https://github.com/stanfordnmbl/osim-rl/issues/43 ), we decided to further update the environment. Uptill version 1.3, the reward received at every step was the total distance travelled from the starting point minus the ligament forces. As a result, the total reward was the cummulative sum of total distances over all steps (or discreet integral of position in time) minus the total sum of ligament forces.

Since, this reward is unconventional in reinforcement learning, we updated the reward function at each step to the distance increment between the two steps minus the ligament forces. As a result, the total reward is the total distance travelled minus the ligament forces. 

Note that this will change the order of magnitude of the total reward from ~1000 to ~10 (now measured in meters travelled). The change does not affect the API of observations and actions. Moreover the measures are strongly correlated and a good model in the old version should perform well in the current version.

# Version 1.3

Due to the errors described in https://github.com/stanfordnmbl/osim-rl/issues/31 we updated the environment
introducing the following changes:

* added velocities of joints (previously we had a duplicate of positions of joints)
* added the left psoas in `muscles` (previously there was only right psoas twice)
* moved obstacles closer to the starting point (so that it's easier to train on them)
* extended the trials to 1000 iterations (if they don't trip/fall earlier as it was before)

In order to switch to the new environment you need to update the `osim-rl` scripts with the following command:

    pip install git+https://github.com/stanfordnmbl/osim-rl.git -U


Since the observation vector changed, you may need to retrain your existing model to account for these new changes.
However, the old observation is in fact a subset of the new observation so if you want to submit the old model
```python
for j in range(6,12):
    observation[j+6] = observation[j]
observation[36] = observation[37]
observation, reward, done, info = env.step(my_controller(observation))
```
Yet, with new information, your controller should be able to perform betters, so we definitely advise to retrain model.
