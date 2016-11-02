# deep-control

## Requirements

OpenSim 4.0 - https://github.com/opensim-org/opensim-core

Make sure you have python bindings installed, i.e. after building OpenSim do

    make install
    cd [opensim_install]/lib/python2.7/site-packages/
    python setup.py install

## Running the python example

    cd python

Now, for training the Arm example (move the arm to certain randomly chosen angles and keep it there):

    python train.ddpg.py --visualize --train --env Arm
    
and for the gait example (walk as far as possible):

    python train.ddpg.py --visualize --train --env Gait
    
After every 10000 iterations the model is dumped to model_[NUM_ITERATIONS].h5f In ordere to test it run

    python train.ddpg.py --visualize --test --env Gait --output model_[NUM_ITERATIONS].h5f

## Running the C++ example (not useful)

Compile

    mkdir build
    cd build
    cmake ../src/ -DCMAKE_PREFIX_PATH=/path/to/OpenSim/lib/cmake/OpenSim/
    make
    cd ..

Run

    ./build/deepControl
