# Proximal policy optimisation

## Introduction 

The proximal policy optimisation (PPO) is an algorithm of the deep reinforcement learning (RL) field. It belongs to the family of policy gradient methods where the agent learns (through deep learning and optimisation process) a policy that dictates the probability of taking different actions in different states. This report gives some theoretical insights of how RL and PPO works.
The first part is dedicated to the basis to understand RL. The second part delves into PPO algorithm.

## Basic of Reinforcement Learning

### Principle 
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The process can be summarized with the following scheme:

<p align="center">

<img src="https://github.com/Theocondette/RL_PPO_in_Vectorized_Robotic_Environments/blob/main/Docs/RL_schemes.jpg" width="300" height="200">

</p>

Where $s_{t}$ and $r_{t}$ are respectively the state and reward at state t and $a_{t}$ is the action taken by the agent at time $t$.


$t \in {1,2,...,T} $,

$a_{t} \in A(s)$,

$s_{t} \in S $,

$r_{t} \in 	\mathbb{R} $

For instance, when the agent is in a particular state s_{t}, he chooses an action a_{t}. Subsequently, the environment respond to a_{t} by providing a reward r_{t+1} and transitioning to a new state s_{t+1}, which is then presented to the agent anew.

The idea of RL is to optimize the action decisions of the agent to get rewards as much as possible.

### Terminology

In reinforcement learning, there are some objects and terminology we have to deal with, in our context, it concerns : 

- **The Policy** : This is the probability of taking an action available at state $s_{t}$.

$$\pi(a|s) = P(A(s)=a_{t} |s = s_{t}) $$


- **The return** : This is the sum of the rewards over the futur discounted rewards. The idea is to value more the present over the futur.

$$ G_{t}=\sum_{k=t+1}^{T}\delta^{k-t-1}r_{k} $$

- **The state value function** : This is the expected return we get given that we are at state t and that we follow the policy pi.

$$ V_{\pi}(s) =\mathrm{E}[G_{t} |s = s_{t}] =\mathrm{E}[\sum_{k=t+1}^{T}\delta^{k-t-1}r_{k} |s = s_{t}]  $$

- **The action value function** : This is the expected return we get given that at t we are at state t, we take action a and we follow the policy pi.

$$ Q_{\pi}(a,s) =\mathrm{E}[G_{t} |s = s_{t},a = a_{t}] =\mathrm{E}[\sum_{k=t+1}^{T}\delta^{k-t-1}r_{k} |s = s_{t}, a = a_{t}]  $$

Our objective in reinforcement learning is to solve the following optimisation problem : 


$$\pi = \argmax\mathrm{E}[G_{t}] $$

In simple terms, we wants the policy that bring us the best rewards path in expectation.

Our goal is to ensure that our policy is optimal, meaning the agent consistently makes favorable decisions a_{t}. With a foundational understanding of reinforcement learning in place, we can now explore the PPO algorithm.

## Proximal policy optimisation 

Proximal Policy Optimization (PPO) is an algorithm for training reinforcement learning agents, particularly in environments with continuous action spaces. PPO has gained popularity due to its stability, sample efficiency, and performance across a wide range of tasks. This is a specific method that we will describe among policy gradient methods.

### Policy Gradient method 

Policy Gradient methods are a class of algorithms that directly optimize the policy to maximize expected rewards. It aims to find the policy directly by adjusting its parameters in the direction that increases expected rewards.

Initial objective to maximize is an estimator of the gradient estimator : 

$$ \max_{\theta}\hat{\mathrm{E_{t}}}[\dfrac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{Old}}(a_{t}|s_{t})} \hat{A_{t}}] $$ 

This formula gives some intuitions : 


- The ratio : 

If the policy does not update, $\dfrac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{Old}}(a_{t}|s_{t})} = 1$. If we increase the probability of taking and action $\dfrac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{Old}}(a_{t}|s_{t})} > 1$.


- The advantage part : 
	Formula : $\hat{Q}_{\pi}(a,s)- \hat{V}_{\pi}(s) = \hat{A_{t}}$ 
	
    The advantage function describes how better we are if we take a specific action ($\hat{Q}$) over taking completely random action ($\hat{V}$) in a states s. 
    Thus, if we take an action which is in average quite better than doing random thing, we will have $\hat{A_{t}}>0$. In the other sense, if we take an action which is in average quite armfull in terms of returns, we would get $\hat{A_{t}}<0$.

The final intuition is satisfying : If we change the policy in a direction such that $\dfrac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{Old}}(a_{t}|s_{t})} > 1$, and $\hat{A_{t}}>0$ we increase the probability of doing good action. If the action is wrong, A<0, we penalize the action and thus the policy in this situation.
The general idea is that we increase the occurrence of right actions and decrease the occurrence of bad ones with this objective.
Now that we have an understanding of the problem, we can delve into the PPO.

Some details are important in this objective :

- Where $\theta$ comes from ?

In the PPO algorithm, the policy is estimated through the help of a Neural Network(NN) that we call actor neural network.


- How do we estimate $A_{t}$?

In the PPO algorithm, the advantage is estimated through the help of a Neural Network(NN) that we call a critic neural network and through the generalized advantage estimation function.

## PPO with clipped surrogate objective :

The PPO algorithm works as follows:


<p align="center">
<img src="https://github.com/Theocondette/RL_PPO_in_Vectorized_Robotic_Environments/blob/main/Docs/PPO_Algo.jpg" width="300" height="200">
</p>

The PPO described here (and the one implemented) is called PPO with clipped surrogate objective because of the objective formula : 

$$ \max_{\theta}\hat{\mathrm{E_{t}}}[\min(\dfrac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{Old}}(a_{t}|s_{t})} \hat{A_{t}} , clip(\dfrac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{Old}}(a_{t}|s_{t})},1 - \epsilon , 1 + \epsilon)\hat{A_{t}}] $$ 


The idea is to take either the initial objective or the clipped one. The intuition is that we want to update the policy, but we do not want to go to far from the actual policy. This is what the clipped part does. The ratio is bounded by $[ 1 - \epsilon , 1 + \epsilon ]$, thus, if the policy update is under the bound, basic objective is used, otherwise the clipped one is used.

This way of penalizing large policy update is not unique, there exist other ways : the trust region methods, the fixed or adaptative Kullback - Leibler penalty. However, these methods are less performant than the clipped surrogate objective according to the Schulman (2017) empirical study. Thus, this method is the one we implement here.



