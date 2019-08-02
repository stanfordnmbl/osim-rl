import opensim as osim
from osim.http.client import Client

# Settings
remote_base = "http://osim-rl-grader.aicrowd.com "
#crowdai_token = "a66245c8324e2d37b92f098a57ef3f99"
crowdai_token = "f97727c493474bf00e23d62235341b31"

client = Client(remote_base)

# Create environment
observation = client.env_create(crowdai_token, env_id='L2M2019Env')

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)

while True:
    [observation, reward, done, info] = client.env_step([0.27365685, 0.3674228, 0.97836083, 0.15261972, 0.3319228, 0.03692374, 0.09905472, 0.1971763 , 0.8908676 , 0.5744208, 0.9313108 , 0.26675472, 0.54930794, 0.91221607, 0.7701997, 0.95412385, 0.43612957, 0.2880115 , 0.26009786, 0.27759373, 0.9234055 , 0.63657844], True)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()