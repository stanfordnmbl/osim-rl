# osim-rl

## Requirements

OpenSim 4.0 - https://github.com/opensim-org/opensim-core

Make sure you have python bindings installed, i.e. after building OpenSim do

    make install
    cd [opensim_install]/lib/python2.7/site-packages/
    python setup.py install
    
Install requirements for the python package

    cd python
    sudo pip install -r requirements.txt

## Recommended

https://github.com/kidzik/keras-rl
https://github.com/openai/rllab

## Training in rllab

Go to
    
    scripts/rllab/
    
### Training

For training the Arm example with DDPG:

    python experiment.py -e Arm -a DDPG
    
For training the Arm example with TRPO:

    python experiment.py -e Arm -a TRPO

### Test

Show the result

    python visualize.py -p /path/to/params.pkl

## Training in heras-rl

Go to
    
    scripts/keras-rl/

### Training

For training the Arm example (move the arm to certain randomly chosen angles and keep it there):

    python train.ddpg.py --visualize --test --env Arm --output models/Arm
    
For the 'gait' example (walk as far as possible):

    python train.ddpg.py --visualize --test --env Gait --output models/Gait
    
For the 'stand still' example:

    python train.ddpg.py --visualize --test --env Stand --output models/Stand

### Test

For training the Arm example (move the arm to certain randomly chosen angles and keep it there):

    python train.ddpg.py --visualize --train --env Arm
    
and for the gait example (walk as far as possible):

    python train.ddpg.py --visualize --train --env Gait
    
After every 10000 iterations the model is dumped to model_[NUM_ITERATIONS].h5f In ordere to test it run

    python train.ddpg.py --visualize --test --env Gait --output model_[NUM_ITERATIONS]
