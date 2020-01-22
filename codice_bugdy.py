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
def run_episode(env, policy, length, reward_model, gamma=0.99):
    # Restart the MiniGrid environment.
    state = state_filter(env.reset())  

    # We need to keep a record of states, actions, and the
    # instantaneous rewards.
    states = [state]
    actions = []
    rewards = []
    dones = []
    all_r = []

    # Run for desired episode length.
    for step in range(80):
        env.render('human')
        time.sleep(0.1)
        # Get action from policy net based on current state.
        action = select_action(policy, state)

        # Simulate one step, get new state and instantaneous reward.
        s, reward, done, _ = env.step(action)
        state = state_filter(s)
        states.append(state)
        rewards.append(reward_model([s['image']])[0].item())

        if len(all_r) >= 1000:
            all_r.pop(0)
        all_r.append(rewards[-1])
        actions.append(action)
        dones.append(done)

        if done:
            break

    # Finished with episode, compute loss per step.
    #if True in dones:
    for i in range(len(rewards)):
        rewards[i] = ( ( rewards[i] - np.mean(all_r) ) / np.std(all_r) ) * 0.5

    discounted_rewards = compute_discounted_rewards(rewards, gamma)
    #else:
    #    discounted_rewards = np.zeros(len(rewards))
    
    if done:
        print(discounted_rewards)

    # Return the sequence of states, actions, and the corresponding rewards.
    return (states, actions, discounted_rewards)

###### The main loop.
if __name__ == '__main__':
    ###### Some configuration variables.
    episode_len = 50  # Length of each game.
    obs_size = 7*7    # MiniGrid uses a 7x7 window of visibility.
    act_size = 7      # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
    inner_size = 64   # Number of neurons in two hidden layers.
    lr = 5e-5        # Adam learning rate
    avg_reward = 0.0  # For tracking average regard per episode.

    # Setup OpenAI Gym environment for guessing game.
    env = gym.make('MiniGrid-Empty-6x6-v0')

    # Instantiate a policy network.
    policy = Policy(obs_size=obs_size, act_size=act_size, inner_size=inner_size)

    # Instantiate a reward model.
    reward_model = csvRewardModel(obs_size=obs_size, inner_size=inner_size)
    reward_model.load_state_dict(torch.load('SAVE_FOLDER/0.5std_(21-01-2020)/csv_reward_weight_lr0.0001_k1000_14:39.pth'))
    reward_model.cuda()
    # Use the Adam optimizer.
    optimizer = torch.optim.Adam(params=policy.parameters(), lr=lr)

    # Run for a while.
    episodes = 2000
    for step in range(episodes):
        # MiniGrid has a QT5 renderer which is pretty cool.
        env.render('human')
        time.sleep(0.1)

        # Run an episode.
        (states, actions, discounted_rewards) = run_episode(env, policy, episode_len, reward_model)
        avg_reward += np.mean(discounted_rewards) # con i reward veri
        if step % 100 == 0:
            print('Average reward @ episode {}: {}'.format(step, avg_reward / 100))
            avg_reward = 0.0
        
        # Repeat each action, and backpropagate discounted
        # rewards. This can probably be batched for efficiency with a
        # memoryless agent...
	# LOSS...
        l = []
        optimizer.zero_grad()
        for (step, a) in enumerate(actions):
            logits = policy(states[step])
            dist = Categorical(logits=logits)
            loss = -dist.log_prob(actions[step]) * discounted_rewards[step]
            l.append(loss.item())
            loss.backward()
        optimizer.step()

        print('Loss : {:.3f} '.format(sum(l) / len(l)) )

    # Now estimate the diagonal FIM.
    print('Estimating diagonal FIM...')
    episodes = 1000
    log_probs = []
    for step in range(episodes):
        # Run an episode.
        (states, actions, discounted_rewards) = run_episode(env, policy, episode_len)
        avg_reward += np.mean(discounted_rewards)
        if step % 100 == 0:
            print('Average reward @ episode {}: {}'.format(step, avg_reward / 100))
            avg_reward = 0.0
        
        # Repeat each action, and backpropagate discounted
        # rewards. This can probably be batched for efficiency with a
        # memoryless agent...
        for (step, a) in enumerate(actions):
            logits = policy(states[step])
            dist = Categorical(logits=logits)
            log_probs.append(-dist.log_prob(actions[step]) * discounted_rewards[step])

    loglikelihoods = torch.cat(log_probs).mean(0)
    loglikelihood_grads = autograd.grad(loglikelihoods, policy.parameters())
    FIM = {n: g**2 for n, g in zip([n for (n, _) in policy.named_parameters()], loglikelihood_grads)}