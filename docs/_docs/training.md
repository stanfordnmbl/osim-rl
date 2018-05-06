---
title: Training your first model
---

Below we present how to train a basic controller using [keras-rl](https://github.com/matthiasplappert/keras-rl). First you need to install extra packages:

    conda install keras -c conda-forge
    pip install git+https://github.com/matthiasplappert/keras-rl.git
    git clone http://github.com/stanfordnmbl/osim-rl.git

`keras-rl` is an excellent package compatible with [OpenAI](http://openai.com/), which allows you to quickly build your first models!

Go to the `scripts` subdirectory from this repository

    cd osim-rl/scripts

There are two scripts:
* `example.py` for training (and testing) an agent using the DDPG algorithm.
* `submit.py` for submitting the result to [crowdAI.org](https://www.crowdai.org/challenges/nips-2017-learning-to-run)

## Training

    python example.py --visualize --train --model sample

## Testing

and for the gait example (walk as far as possible):

    python example.py --visualize --test --model sample
