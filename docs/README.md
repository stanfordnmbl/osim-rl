# Round 2 of the NIPS challenge

In order to minimize overfitting, we will check your solution in the second round on an environment with the **10 obstacles** across a total of **N simulations**. There will be a limit of total **3 submissions** per participant throughout the entire second round.

Only participants with the score above 15 points on the leaderboard will be invited to submit their solutions in the second round.
More details about the actual submission format are available [here](https://hub.docker.com/r/spmohanty/crowdai-nips-learning-to-run-challenge/).

You can recreate the settings of the second round using:

    env = RunEnv(visualize=True, max_obstacles = 10)
    observation = env.reset(difficulty = 2)

We are excited about these last few weeks and we are looking forward to see your final models.

Winners will be selected based on leaderboard position at the end of the second round, for these prizes:

* 1st - NVIDIA DGX Station
* 2nd - NVIDIA Titan Xp
* 3rd - NVIDIA Titan Xp

Moreover, the winner will receive travel grants to:

* NIPS 2017 (including registration fee), December 8th 2017
* Stanford, December 12th 2017 (can be changed),
* Applied ML Days, Switzerland, January 27th-30th 2018.

Please let us know about your availability at NIPS, if you are interested in giving a talk about your solution and if you are already registered, by filling out this [form](https://plantvillage.us12.list-manage.com/track/click?u=f912313fcb27b4deb61905df6&id=cb5cbf2ad2&e=d92672213d).

Presence at NIPS is not mandatory for eligibility, but it’s highly encouraged.
If you are not competing for the prizes, but you will be present at NIPS, please also fill out the form.

# Version 1.5

Grader now accepts only this version. In order to switch to the new environment you need to update the `osim-rl` scripts with the following command:

    pip install git+https://github.com/stanfordnmbl/osim-rl.git -U

This release includes following bugfixes

* Fixed first observation (previously it wasn't showing the first obstacle correctly). ( https://github.com/stanfordnmbl/osim-rl/issues/53 )
* Fixed geometries for the right leg. ( https://github.com/stanfordnmbl/osim-rl/issues/75 )
* Activations from outside [0,1] are clipped to [0,1] ( https://github.com/stanfordnmbl/osim-rl/issues/64 )

# Version 1.4.1

After discussing the way the reward function is computed ( https://github.com/stanfordnmbl/osim-rl/issues/43 ), we decided to further update the environment. Uptill version 1.3, the reward received at every step was the total distance travelled from the starting point minus the ligament forces. As a result, the total reward was the cummulative sum of total distances over all steps (or discreet integral of position in time) minus the total sum of ligament forces.

Since, this reward is unconventional in reinforcement learning, we updated the reward function at each step to the distance increment between the two steps minus the ligament forces. As a result, the total reward is the total distance travelled minus the ligament forces.

In order to switch to the new environment you need to update the `osim-rl` scripts with the following command:

    pip install git+https://github.com/stanfordnmbl/osim-rl.git -U

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
