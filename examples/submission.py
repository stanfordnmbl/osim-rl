import opensim as osim
from osim.http.client import Client
import numpy as np

# Settings
remote_base = "http://osim-rl-grader.aicrowd.com/"
aicrowd_token = "a66245c8324e2d37b92f098a57ef3f99" # use your aicrowd token
# your aicrowd token (API KEY) can be found at your prorfile page at https://www.aicrowd.com

client = Client(remote_base)

# Create environment
observation = client.env_create(aicrowd_token, env_id='L2M2019Env')

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)

while True:
    #action = my_controller.update(observation)
    action = [.5]*22
    [observation, reward, done, info] = client.env_step(action)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()