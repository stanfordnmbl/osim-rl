---
title: Evaluation
---

<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

Your task is to build a controller, i.e. a function from the state of the biomechanical model to action, such that the velocity of the controller matches the requested velocity as precisely as possible. 

Formally, you build a function \\(a:S \rightarrow A\\) from the state space \\(S\\) to the action space \\(A\\). Each element \\(s\\) of the state space is represented as a dictionary structure that includes current positions, velocities, accelerations of joints and body parts, muscles activity, etc. The action space \\([0,1]^{22}\\) represents muscle activations. Your objective is to find such a function \\(a\\) that reward throughout the episode is maximized

The trial ends either if the pelvis of the model goes below \\(5\\) meters or if you reach \\(1000\\) iterations (corresponding to \\(10\\) seconds in the virtual environment). Your total reward is the sum of 

$$ -\sum_{t=1}^{1000} \|v(s_t) - R_t\|, $$

where \\(s_t\\) is the state of the model at time \\(t\\), \\(v(s)\\) is the velocity vector of the pelvis in the state \\(s\\), and \\(s_t = M(s_{t-1}, a(s_{t_1}))\\), i.e. states follow the simulation given by model \\(M\\). The process \\(R_t\\) for the final environment is unknown but we will provide a distribution from which it is drawn.

<div class="note unreleased">
  <h5>Unreleased</h5>
  <p>This is not the final reward function. The final objective function for the NIPS 2018 challenge will be released when the competition starts.</p>
</div>

You can test your model on your local machine. For submission, you will need to interact with the remote environment: [crowdAI](https://www.crowdai.org/challenges/nips-2017-learning-to-run) sends you the current state \\(s\\) and you need to send back the action you take in the given state. You will be evaluated at three different levels of difficulty. 

## Rules

In order to avoid overfitting to the training environment, the top participants (those who obtained 15.0 points or more) will be asked to resubmit their solutions in the second round of the challenge. Each participant will have a limit of **3 submissions**. The final ranking will be based on the results from the second round.

Additional rules:
* Organizers reserve the right to modify challenge rules as required.
