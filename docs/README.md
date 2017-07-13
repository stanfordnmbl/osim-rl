# Version 1.3

Due to the errors described in https://github.com/stanfordnmbl/osim-rl/issues/31 we updated the environment
introducing the following changes:

* added velocities of joints (previously we had a duplicate of positions of joints)
* added the left psoas in `muscles` (previously there was only right psoas twice)
* moved obstacles closer to the starting point (so that it's easier to train on them)
* extended the trials to 1000 iterations (if they don't trip/fall earlier as it was before)

In order to switch to the new environment you need to update the `osim-rl` scripts with the following command:

    pip install git+https://github.com/stanfordnmbl/osim-rl.git -U

