from osim.env import L2M2019Env
import numpy as np
import json

#=== this is the official setting for Learn to Move 2019 ===#
model = '3D'
round_n = 1 # Round 1
#round_n = 2 # Round 2

if round_n == 1:
    difficulty = 2 # 2: Round 1; 3: Round 2
    seed = None
    project = True
    obs_as_dict = True
elif round_n == 2:
    difficulty = 3 # 2: Round 1; 3: Round 2
    seed = None
    project = True
    obs_as_dict = True
else:
    difficulty = 1 # 0: constant forward velocities; 1: consecutive sinks forward for walking
    seed = None
    project = True
    obs_as_dict = True
#=== this is the official setting for Learn to Move 2019 ===#

env = L2M2019Env(seed=seed, difficulty=difficulty)
env.change_model(model=model, difficulty=difficulty, seed=seed)
obs_dict = env.reset(project=project, seed=seed, obs_as_dict=obs_as_dict)


with open('actions/actions_list.json') as json_file:
    data = json.load(json_file)

text_file = open("17018.json", "w")
text_file.write(json.dumps(data['17018']))
text_file.close()

rewsum = 0
for action in data['17018'][2:10000]:
    if action == "reset":
        break
    action = eval(action)
    obs_dict, reward, done, info=env.step(action, project=project, obs_as_dict=obs_as_dict)
    rewsum += reward
print(rewsum)

