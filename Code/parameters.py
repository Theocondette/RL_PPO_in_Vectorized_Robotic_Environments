#learning rate backward propagation NN action
lr_actor = 0.0003
#learning rate backward propagation NN state value estimation
lr_critic = 0.0003
#Number of Learning Iteration we want to perform
Iter = 40
#Number max of step to realise in one episode. 
MAX_STEP = 1000
#How rewards are discounted.
gamma =0.98
#How do we stabilize variance in the return computation.
lambd = 0.95
#batch to train on
batch_size = 64
# Do we want high change to be taken into account.
epsilon = 0.2
#weight decay coefficient in ADAM for state value optim.
l2_rate = 0.001

