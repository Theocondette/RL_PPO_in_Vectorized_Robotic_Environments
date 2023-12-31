# RL_PPO_in_Vectorized_Robotic_Environments

## Objective
This project delves into implementing and analyzing the Proximal Policy Optimization (PPO) algorithm in vectorized robotic environments, such as those provided by Mujoco and BRAX.

## Methodology

1 - Understand Reinforcement Learning (RL) and PPO

Check REPORT.MD for my explanations about Reinforcement Learning basis and PPO algorithm.
Also, I recommend many websites to [learn about RL]( https://spinningup.openai.com/en/latest/spinningup/rl_intro.html), to [learn about Policy Gradient](https://towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c), and to [learn about PPO](https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8).

2 - Code a PPO

Based on the coding references provided, I developed an implementation of PPO tailored for environments characterized by continuous observation and action spaces.

The code is organized as follow :

* parameters.py : contains the parameter of the training.
* model.py : contains Actor (Forward propagation of actor neural network) and Critic class(Forward propagation of critic neural network).
* PPO.py : contains PPO class (training process, Kullback-Leibler divergence computation, generalized advantage estimation process) and Normalize class.
* main.py : contains the main code to run.

3 - Implement the algorithm in various environment

I implement and evaluate the PPO algorithm in two Mujoco environments : InvertedPendulum-v4 and Humanoid-v4.

## Results

My implementation concerns two Mujoco environments : [InvertedPendulum-v4](https://www.gymlibrary.dev/environments/mujoco/inverted_pendulum/) and [Humanoid-v4](https://www.gymlibrary.dev/environments/mujoco/humanoid/). Check my results below !


**InvertedPendulum-v4**


  
Before training

<p align="center">
<img src="https://github.com/Theocondette/RL_PPO_in_Vectorized_Robotic_Environments/blob/main/Graph/InvertedPendulum_before.gif" width="300" height="300">
</p>

After training

<p align="center">
<img src="https://github.com/Theocondette/RL_PPO_in_Vectorized_Robotic_Environments/blob/main/Graph/Inverted_pendulum_after.gif" width="300" height="300">
</p>

<p align="center">
<img src="https://github.com/Theocondette/RL_PPO_in_Vectorized_Robotic_Environments/blob/main/Graph/Inverted_pendulum_learning.png" width="500" height="300">
</p>

**Humanoid-v4**

Before training

<p align="center">
<img src="https://github.com/Theocondette/RL_PPO_in_Vectorized_Robotic_Environments/blob/main/Graph/Humanoid_before.gif" width="300" height="300">
</p>

After training

<p align="center">
<img src="https://github.com/Theocondette/RL_PPO_in_Vectorized_Robotic_Environments/blob/main/Graph/Humanoid_after.gif" width="300" height="300">
</p>

<p align="center">
<img src="https://github.com/Theocondette/RL_PPO_in_Vectorized_Robotic_Environments/blob/main/Graph/Humanoid_learning.png" width="500" height="300">
</p>

## Citations

### Paper

*  Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

### Code

* Eric Yang Yu. (2020). PPO-for-Beginners. https://github.com/ericyangyu/PPO-for-Beginners
* Han. (2020). PPO-pytorch-Mujoco. https://github.com/qingshi9974/PPO-pytorch-Mujoco.git
* Phil Tabor. (2020). Youtube-Code-Repository. https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch



