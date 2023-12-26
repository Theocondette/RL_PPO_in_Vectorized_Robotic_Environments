import torch
import torch.nn as nn
from parameters import *

# Actor class: Used to choose actions of a continuous action space.

class Actor(nn.Module):
    def __init__(self,N_S,N_A):
      # Initialize NN structure.
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.sigma = nn.Linear(64,N_A)
        self.mu = nn.Linear(64,N_A)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        # This approach use gaussian distribution to decide actions. Could be
        # something else.
        self.distribution = torch.distributions.Normal

    def set_init(self,layers):
      # Initialize weight and bias according to a normal distrib mean 0 and sd 0.1.
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
      # Use of tanh activation function is recommanded : bounded [-1,1],
      # gives some non-linearity, and tends to give some stability.
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        # mu action output of the NN.
        mu = self.mu(x)
        #log_sigma action output of the NN
        log_sigma = self.sigma(x)
        sigma = torch.exp(log_sigma)
        return mu,sigma

    def choose_action(self,s):
      # Choose action in the continuous action space using normal distribution
      # defined by mu and sigma of each actions returned by the NN.
        mu,sigma = self.forward(s)
        Pi = self.distribution(mu,sigma)
        return Pi.sample().numpy()


# Critic class : Used to estimate V(state) the state value function through a NN.
class Critic(nn.Module):
    def __init__(self,N_S):
      # Initialize NN structure.
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def set_init(self,layers):
      # Initialize weight and bias according to a normal distrib mean 0 and sd 0.1.
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
      # Use of tanh activation function is recommanded.
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        values = self.fc3(x)
        return values