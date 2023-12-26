import gym
import torch
import numpy as np
import argparse
from parameters import *
from PPO import Ppo,Normalize
from collections import deque
parser = argparse.ArgumentParser()
#Here we specify whether we want to train InvertedPendulum or Humanoid.
#This PPO implementation works only for continuous action and environement space.

## TO CHOOSE :
#'InvertedPendulum-v4'
#'Humanoid-v4'
parser.add_argument('--env_name', type=str, default='InvertedPendulum-v4',
                    help='name of Mujoco environement')
parser.add_argument("-f", required=False)
args = parser.parse_args()

#We set the environement using gym library completed by Mujoco environnement.
#We set the environement to Humanoid-v4 (link: https://www.gymlibrary.dev/environments/mujoco/humanoid/)
env = gym.make(args.env_name)
#Number of state and action
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

# Random seed initialization
env.seed(500)
torch.manual_seed(500)
np.random.seed(500)

# Run the Ppo class
frames = []
ppo = Ppo(N_S,N_A)
# Normalisation for stability, fast convergence... always good to do.
normalize = Normalize(N_S)
episodes = 0
eva_episodes = 0
for iter in range(Iter):
    memory = deque()
    scores = []
    steps = 0
    while steps <2048: #Horizon
        episodes += 1
        s = normalize(env.reset())
        score = 0
        for _ in range(MAX_STEP):
            steps += 1
            #Choose an action: detailed in PPO.py
            # The action is a numpy array of 17 elements. It means that in the 17 possible directions of action we have a specific value in the continuous space.
            # Exemple : the first coordinate correspond to the Torque applied on the hinge in the y-coordinate of the abdomen: this is continuous space.
            a=ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]

            #Environnement reaction to the action : There is a reaction in the 376 elements that characterize the space :
            # Exemple : the first coordinate of the states is the z-coordinate of the torso (centre) and using env.step(a), we get the reaction of this state and
            # of all the other ones after the action has been made.
            s_ , r ,done,info = env.step(a)
            s_ = normalize(s_)

            # Do we continue or do we terminate an episode?
            mask = (1-done)*1
            memory.append([s,a,r,mask])

            score += r
            s = s_
            if done:
                break
        with open('log_' + args.env_name  + '.txt', 'a') as outfile:
            outfile.write('\t' + str(episodes)  + '\t' + str(score) + '\n')
        scores.append(score)
    score_avg = np.mean(scores)
    print('{} episode score is {:.2f}'.format(episodes, score_avg))
    ppo.train(memory)



