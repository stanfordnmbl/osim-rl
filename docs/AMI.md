# Using Amazon AMI

You can easily set up OpenSim Reinforcement learning environmen on AWS Cloud using our AMI.
Just create a machine with `ami-971be8ee` and log in using `ssh ubuntu@[IP_OF_THE_MACHINE]`.
Remember to turn off your machine if you are not using it. Amazon will charge you for every hour of usage.

## Screencast

### Log in and choose EC2
![1](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/1.png)

### Click "Launch instance"
![2](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/2.png)

### Choose "Community AMI" and seacrh "ami-971be8ee" 
![3](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/3.png)

### Choose instance type according to your needs
**Attention**: They greatly vary in price, check Amazon Pricelist and Terms!
![4](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/4.png)

### Confirm and launch your instance
![5](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/5.png)

### Create a keypair and download it
![6](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/6.png)

**You need to add this key to your keychain or add it to your SSH client.** On Linux/MacOS type `ssh-add [path-to-key.pem]` on windows add it to your ssh client (e.g. putty)

### See the confirmation, go to your instances
![7](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/7.png)

### Click on your instance and check IPv4 Public IP
![8](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/8.png)

### Login with ssh client to ubuntu@[IP]
![9](https://s3-eu-west-1.amazonaws.com/kidzinski/opensim-ami/9.png)

Activate conda opensim environment with

    source activate opensim-rl

and you are ready to use our `osim-rl` package. For the example script, navigate to

    cd ~/osim-rl/scripts

and run

    python example.py --train --model sample

once your model is trained you can run 

    python example.py --test --model sample --token [YOUR_CROWD_AI_TOKEN]

to submit the solution.
