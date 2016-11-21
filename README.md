# osim-rl

## What?

[OpenSim](https://github.com/opensim-org/opensim-core) is a biomechanical physics environment for musculoskeletal simulations. Biomechanical community designed a range of musculoskeletal models compatible with this environment. These models can be, for example, fit to clinical data to understand underlying causes of injuries using inverse kinematics and inverse dynamics.

For many of these models there are controllers designed for forward simulations of movement, however they are often finely tuned for the model and data. Advancements in reinforcement learning may allow building more robust controllers which can in turn provide another tool for validating the models. Moreover they could help visualize, for example, kinematics of patients after surgeries.

We include two musculoskeletal models: ARM with 6 muscles and 2 degrees of freedom and HUMAN with 18 muscles and 9 degrees of freedom. For the ARM model we designed an environment where the arm is supposed to reach certain points. For HUMAN we have four environmnets: standing still, crouch, jump and gait. Environments are compatible with [rllab](https://github.com/openai/rllab), [keras-rl](https://github.com/matthiasplappert/keras-rl) and [OpenAI gym](https://gym.openai.com/).

![ARM environment](https://github.com/kidzik/osim-rl/blob/master/demo/arm.gif)
![HUMAN environment](https://github.com/kidzik/osim-rl/blob/master/demo/stand.gif)

Note that from reinforcement learning perspective, due to high dimensionality of muscles space, the problem is significantly harder than 'textbook' reinforcement learning problems.

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

https://github.com/matthiasplappert/keras-rl

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

## Training in keras-rl

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

## Credits

Thanks to @carmichaelong for simplified human model and to OpenSim team for making OpenSim!
