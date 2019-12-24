import time
import gym
import gym_minigrid
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions.categorical import Categorical
from itertools import count

# A simple, memoryless MLP (Multy Layer Perceptron) agent.
# Last layer are logits (scores for which higher values
# represent preferred actions.
class Policy(nn.Module):
    def __init__(self, obs_size, act_size, inner_size, **kwargs):
        super(Policy, self).__init__(**kwargs)
        self.affine1 = nn.Linear(obs_size, inner_size)
        self.affine2 = nn.Linear(inner_size, act_size)

    def forward(self, x):
        x = x.view(-1, 7*7)
        x = F.relu(self.affine1(x))
        act_probs = self.affine2(x).clamp(-1000.0, +1000.0) 
        return act_probs

# Simple function to calculate the loss
def Loss(policy, optimizer, states, actions, discounted_rewards):
    optimizer.zero_grad()
    losses = []
    for (step, a) in enumerate(actions):
        logits = policy(state_filter(states[step]))
        dist = Categorical(logits=logits)
        loss = -dist.log_prob(torch.tensor(a).cuda()) * discounted_rewards[step]
        losses.append(loss.item())
        loss.backward()
    optimizer.step()
    return losses

# Function that, given a policy network and a state selects a random
# action according to the probabilities output by final layer.
def select_action(policy, state):
    probs = policy.forward(state)
    dist = Categorical(logits=probs) # categorical actions' distribution
    action = dist.sample()
    return action

# Utility function. The MiniGrid gym environment uses 3 channels as
# state, but for this we only use the first channel: represents all
# objects (including goal) with integers. This function just strips
# out the first channel and returns it.
def state_filter(state):
    return torch.from_numpy(state['obs'][:,:,0]).float().cuda()

# Function to compute discounted rewards after a complete episode.
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        discounted_rewards.append(running)
    return list(reversed(discounted_rewards))

# The function that runs the simulation for a specified length. The
# nice thing about the MiniGrid environment is that the game never
# ends. After achieving the goal, the game resets. Kind of like
# Sisyphus...

def run_episode(env, policy, length):
    # Restart the MiniGrid environment.
    state = env.reset()

    # We need to keep a record of states, actions, and the
    # instantaneous rewards.
    states = [state]
    actions = []
    rewards = []
    dones = []

    # Run for desired episode length.
    for step in range(length):

        # Get action from policy net based on current state.
        action = select_action(policy, state_filter(state))

        # Simulate one step, get new state and instantaneous reward.
        state, reward, done, _ = env.step(action.item())

        states.append(state)
        rewards.append(reward)
        actions.append(action.item())
        dones.append(done)

        if done:
            break
            
    # Return the sequence of states, actions, and the a list contain boolean variables
    # that is used to understand if in this sequence the agent achieves the goal.
    return (states, actions, dones)

# Simple utility function to save the policy weights
def save_policy_weights(model, save_weights_path):
    torch.save(model.state_dict(), save_weights_path + '/policy_weight.pth')

