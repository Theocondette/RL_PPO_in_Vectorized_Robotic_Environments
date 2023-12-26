from model import Actor,Critic
import torch.optim as optim
from parameters import *
import torch
import numpy as np

class Ppo:
    def __init__(self,N_S,N_A):
      # Initialize all the object we need for PPO.
        self.actor_net =Actor(N_S,N_A)
        self.critic_net = Critic(N_S)
        self.actor_optim = optim.Adam(self.actor_net.parameters(),lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(),lr=lr_critic,weight_decay=l2_rate)
        self.critic_loss_func = torch.nn.MSELoss()

    def train(self,memory):

        #Easier to hande as np array and to separate everything for each episodes.
        memory = np.array(memory)
        states = torch.tensor(np.vstack(memory[:,0]),dtype=torch.float32)
        actions = torch.tensor(list(memory[:,1]),dtype=torch.float32)
        rewards = torch.tensor(list(memory[:,2]),dtype=torch.float32)
        masks = torch.tensor(list(memory[:,3]),dtype=torch.float32)

        # Use critic network defined in Model.py
        # This function enables to get the current state value V(S).
        values = self.critic_net(states)
        # Get advantage.
        returns,advants = self.get_gae(rewards,masks,values)
        #Get old mu and std.
        old_mu,old_std = self.actor_net(states)
        #Get the old distribution.
        pi = self.actor_net.distribution(old_mu,old_std)
        #Compute old policy.
        old_log_prob = pi.log_prob(actions).sum(1,keepdim=True)

        # Everything happens here
        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n//batch_size):
                b_index = arr[batch_size*i:batch_size*(i+1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                #New parameter of the policy distribution by action.
                mu,std = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu,std)
                new_prob = pi.log_prob(b_actions).sum(1,keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                #Regularisation fixed KL : does not work as good as following clipping strategy
                # empirically.
                # KL_penalty = self.kl_divergence(old_mu[b_index],old_std[b_index],mu,std)
                ratio = torch.exp(new_prob-old_prob)

                surrogate_loss = ratio*b_advants
                values = self.critic_net(b_states)
                # MSE Loss : (State action value - State value)^2
                critic_loss = self.critic_loss_func(values,b_returns)
                # critic_loss = critic_loss - beta*KL_penalty

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                #Clipping strategy
                ratio = torch.clamp(ratio,1.0-epsilon,1.0+epsilon)
                clipped_loss =ratio*b_advants
                # Actual loss
                actor_loss = -torch.min(surrogate_loss,clipped_loss).mean()
                #Now that we have the loss, we can do the backward propagation to learn : everything is here.
                self.actor_optim.zero_grad()
                actor_loss.backward()

                self.actor_optim.step()
    # Get the Kullback - Leibler divergence: Measure of the diff btwn new and old policy:
    # Could be used for the objective function depending on the strategy that needs to be
    # teste.
    def kl_divergence(self,old_mu,old_sigma,mu,sigma):

        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()

        kl = torch.log(old_sigma) - torch.log(sigma) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / \
             (2.0 * sigma.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    # Advantage estimation:
    def get_gae(self,rewards, masks, values):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        #Create an equivalent fullfilled of 0.
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        #Init
        running_returns = 0
        previous_value = 0
        running_advants = 0
        #Here we compute A_t the advantage.
        for t in reversed(range(0, len(rewards))):
            # Here we compute the discounted returns. Gamma is the discount factor.
            running_returns = rewards[t] + gamma * running_returns * masks[t]
            #computes the difference between the estimated value at time step t (values.data[t]) and the discounted next value.
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            # Compute advantage
            running_advants = running_tderror + gamma * lambd * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        #Normalization to stabilize final advantage of the history to now.
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

# Creation of a class to normalize the states
class Normalize:
    def __init__(self, N_S):
        self.mean = np.zeros((N_S,))
        self.std = np.zeros((N_S, ))
        self.stdd = np.zeros((N_S, ))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean

        x = x - self.mean
        x = x / (self.std + 1e-8)
        x = np.clip(x, -5, +5)
        return x
