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

from ReinforcementLearning.csvRewardModel import csvRewardModel

# A simple, memoryless MLP (Multy Layer Perceptron) agent.
# Last layer are logits (scores for which higher values
# represent preferred actions.
class Policy(nn.Module):
    def __init__(self, obs_size, act_size, inner_size, **kwargs):
        super(Policy, self).__init__(**kwargs)
        self.affine1 = nn.Linear(obs_size, inner_size)
        self.affine2 = nn.Linear(inner_size, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = x.view(-1, 7*7)
        x = F.relu(self.affine1(x))
        act_probs = self.affine2(x).clamp(-1000.0, +1000.0)
        return act_probs

# Function that, given a policy network and a state selects a random
# action according to the probabilities output by final layer.
def select_action(policy, state):
    probs = policy.forward(state)
    dist = Categorical(logits=probs)
    action = dist.sample()
    return action

# Utility function. The MiniGrid gym environment uses 3 channels as
# state, but for this we only use the first channel: represents all
# objects (including goal) with integers. This function just strips
# out the first channel and returns it.
def state_filter(state):
    return torch.from_numpy(state['image'][:,:,0]).float()

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
def run_episode(env, policy, length, gamma=0.99):
    # Restart the MiniGrid environment.
    state = state_filter(env.reset())

    # We need to keep a record of states, actions, and the
    # instantaneous rewards.
    states = [state]
    actions = []
    rewards = []

    # Run for desired episode length.
    for step in range(length):
        env.render('human')
        time.sleep(0.5)
        # Get action from policy net based on current state.
        action = select_action(policy, state)

        # Simulate one step, get new state and instantaneous reward.
        state, reward, done, _ = env.step(action)
        state = state_filter(state)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        if done:
            break

    # Finished with episode, compute loss per step.
    discounted_rewards = compute_discounted_rewards(rewards, gamma)

    # Return the sequence of states, actions, and the corresponding rewards.
    return (states, actions, discounted_rewards)

###### The main loop.
if __name__ == '__main__':
    ###### Some configuration variables.
    episode_len = 50  # Length of each game.
    obs_size = 7*7    # MiniGrid uses a 7x7 window of visibility.
    act_size = 7      # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
    inner_size = 64   # Number of neurons in two hidden layers.
    lr = 0.001        # Adam learning rate
    avg_reward = 0.0  # For tracking average regard per episode.

    env_name = 'MiniGrid-Empty-6x6-v0' # 'MiniGrid-Empty-6x6-v0'
    # Setup OpenAI Gym environment for guessing game.
    env = gym.make(env_name)

    # Instantiate a policy network.
    policy = Policy(obs_size=obs_size, act_size=act_size, inner_size=inner_size)
    policy.load_state_dict(torch.load('zz_saving/weigths/policy_weight_20:55.pth'))

    # Run for a while.
    episodes = 2000
    for step in range(episodes):
        # MiniGrid has a QT5 renderer which is pretty cool.
        env.render('human')
        time.sleep(0.5)

        # Run an episode.
        (states, actions, discounted_rewards) = run_episode(env, policy, episode_len)

	
