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

## Screencast

![1](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/1.png)
![2](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/2.png)
![3](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/3.png)
![4](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/4.png)
![5](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/5.png)
![6](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/6.png)
![7](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/7.png)
![8](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/8.png)
![9](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/9.png)
