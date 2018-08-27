# osim-rl

Install the most recent version with
```bash
pip install git+https://github.com/stanfordnmbl/osim-rl.git -U
```
in the conda environment with OpenSim.

## [2.1.0] - 2018-08-27
- `equilibrateMuscles` called in the first step (https://github.com/stanfordnmbl/osim-rl/issues/133)
- Added the new reward function (round 2)
- `timestap_limit` for the second round is 1000
- Fixed the `set_state` (https://github.com/stanfordnmbl/osim-rl/issues/125)
- `difficulty` added to the environment (https://github.com/stanfordnmbl/osim-rl/issues/127). `1` for the second round and `0` for the first round
- `seed` added to the environment (https://github.com/stanfordnmbl/osim-rl/issues/158)
