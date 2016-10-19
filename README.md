# deep-control

## Requirements

OpenSim 4.0 - https://github.com/opensim-org/opensim-core
Make sure you have python bindings installed

## Running the python example

    cd python
    python deepcontroller.py

## Running the C++ example

Compile

    mkdir build
    cd build
    cmake ../src/ -DCMAKE_PREFIX_PATH=/path/to/OpenSim/lib/cmake/OpenSim/
    make
    cd ..

Run

    ./build/deepControl
