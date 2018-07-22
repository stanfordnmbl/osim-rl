---
title: Evaluation
---

<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

Your task is to build a controller, i.e. a function from the state of the biomechanical model to action, such that the velocity of the controller matches the requested velocity as precisely as possible. 

Formally, you build a function \\(a:S \rightarrow A\\) from the state space \\(S\\) to the action space \\(A\\). Each element \\(s\\) of the state space is represented as a dictionary structure that includes current positions, velocities, accelerations of joints and body parts, muscles activity, etc. The action space \\([0,1]^{19}\\) represents muscle activations. Your objective is to find such a function \\(a\\) that reward throughout the episode is maximized. The challenge has two rounds, for now we provide the objective function of the first round.

## Round 1

The trial ends in the step \\(T\\) when the pelvis of the model goes below \\(0.6\\) meters or when it reaches \\(300\\) iterations (corresponding to \\(3\\) seconds in the virtual environment). Your total reward is the sum of 

$$ \sum_{t=1}^{T} 9 - |v_x(s_t) - 3|^2, $$

where \\(s_t\\) is the state of the model at time \\(t\\), \\(v_x(s)\\) is the horizontal velocity vector of the pelvis in the state \\(s\\), and \\(s_t = M(s_{t-1}, a(s_{t_1}))\\), i.e. states follow the simulation given by model \\(M\\).

## Submission

You can test your model on your local machine. For submission, you will need to interact with the remote environment: [crowdAI](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge) sends you the current state \\(s\\) and you need to send back the action you take in the given state. You will be evaluated at three different levels of difficulty. 

## Rules

In order to avoid overfitting to the training environment, the top participants will be asked to resubmit their solutions in the second round of the challenge. The final ranking will be based on the results from the second round.

Additional rules:
* Organizers reserve the right to modify challenge rules as required.
