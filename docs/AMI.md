# Using Amazon AMI

You can easily set up OpenSim Reinforcement learning environmen on AWS Cloud using our AMI.
Just create a machine using our AMI `ami-971be8ee` and log in using `ssh ubuntu@[IP_OF_THE_MACHINE]`.

Next activate conda opensim environment with

    source activate opensim-rl

and you are ready to use our `osim-rl` package. For the example script, navigate to

    cd ~/osim-rl/scripts

and run

    python example.py --train --model sample

once your model is trained you can run 

    python example.py --test --model sample --token [YOUR_CROWD_AI_TOKEN]

to submit the solution.
