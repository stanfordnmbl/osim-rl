---
title: Submission
---

Assuming your controller is trained and is represented as a function `my_controller(observation)` returning an `action` you can submit it to [crowdAI](https://www.crowdai.org/challenges/nips-2017-learning-to-run) through interaction with an environment there:

```python
import opensim as osim
from osim.http.client import Client

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "[YOUR_CROWD_AI_TOKEN_HERE]"

client = Client(remote_base)

# Create environment
observation = client.env_create(crowdai_token)

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)

while True:
    [observation, reward, done, info] = client.env_step(my_controller(observation), True)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
```

In the place of `[YOUR_CROWD_AI_TOKEN_HERE]` put your token from the profile page from [crowdai.org](http://crowdai.org/) website. You can also use [this script](https://github.com/stanfordnmbl/osim-rl/blob/master/examples/submit.py).

Note that during the submission, the environment will get restarted. Since the environment is stochastic, you will need to submit three trials -- this way we make sure that your model is robust.

