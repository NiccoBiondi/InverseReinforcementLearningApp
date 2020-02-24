import os
import time
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utility.utility import save_losses_list

# Utility function. The MiniGrid gym environment uses 3 channels as
# state, but for this we only use the first channel: represents all
# objects (including goal) with integers. This function just strips
# out the first channel and returns it.
def state_filter(state):
    return torch.from_numpy(state[:,:,0]).float()

# RewardModel class. It takes in input a clip, collection of states, 
# and process them giving back a list of reward, of for each state.
class csvRewardModel(nn.Module):
    def __init__(self, obs_size, inner_size, **kwargs):
        super(csvRewardModel, self).__init__(**kwargs)
        self.affine1 = nn.Linear(obs_size, inner_size)
        self.affine2 = nn.Linear(inner_size, 1)

    def forward(self, clip):
        rewards = []

        for obs in clip:
            x_1 = state_filter(obs).cuda().view(-1, 7*7)
            x_1 = F.relu(self.affine1(x_1))
            rewards.append(self.affine2(x_1))
        
        return rewards

    # The loss function. It takes the annotation buffer and process the triples.
    # A triple is a list composed by [first clip, second clip, label]. The label
    # is a list that can be [1, 0] if the user choose the first clip, [0, 1] if the
    # user choose the second clip, or [0.5, 0.5] if the user choose both clips.
    # The labels [0,0], those are discarded clips, will be not included in this process.
    def compute_rewards(self, reward_model, optimizer, train_clips):
        # list for separate loss component, we study why the reward model loss grows 
        # during the system training (for more details please refers to the review)
        loss = []
        loss_05 = []
        loss_01 = []

        for triple in train_clips:    
            preference = triple[2]
            # Compute reward for all single state in each clips 
            reward_clip_1 = reward_model.forward(triple[0])
            reward_clip_2 = reward_model.forward(triple[1])
            
            # Compute the P[signma_1 > sigma_2] probability.
            den = (torch.exp(sum(reward_clip_1)) + torch.exp(sum(reward_clip_2))) + 1e-7 
            sigma_clip_1 = torch.exp(sum(reward_clip_1)) / den
            sigma_clip_2 = torch.exp(sum(reward_clip_2)) / den

            if den.item() == torch.tensor([float('inf')]).item() and sigma_clip_1.item() == torch.tensor([float('inf')]).item():
                sigma_clip_1 = torch.tensor([1]).cuda()
                sigma_clip_2 = torch.tensor([0]).cuda()
            elif den.item() == torch.tensor([float('inf')]).item() and sigma_clip_2.item() == torch.tensor([float('inf')]).item():
                sigma_clip_1 = torch.tensor([0]).cuda()
                sigma_clip_2 = torch.tensor([1]).cuda()                

            if preference == [0.5,0.5]: 
                v = -1 * ( ( preference[0] * torch.log(sigma_clip_1) ) + ( preference[1] * torch.log(sigma_clip_2) ) + 1e-7 )
                loss_05.append( v.item() )

            else:
                v = -1 * ( ( preference[0] * torch.log(sigma_clip_1) ) + ( preference[1] * torch.log(sigma_clip_2) ) + 1e-7 )
                loss_01.append ( v.item() )

            loss.append( v )

        loss_01 = sum(loss_01)
        loss_05 = sum(loss_05)

        # Compute loss and backpropagate.
        optimizer.zero_grad()
        loss = sum(loss)   
        loss.backward() 
        optimizer.step()

        return loss.item(), loss_05, loss_01

# function to save the reward model weights
def save_reward_weights(reward_model, save_weights, default_path, lr, K, reward_losses=None):

    if reward_losses != None and reward_losses:
        save_losses_list(save_weights + '/reward_model_losses.csv', reward_losses)

    current_time = time.strftime("%H:%M", time.localtime())
    if [save_weights + '/' + el for el in os.listdir(save_weights) if 'csv_reward_weight' in el]:
        os.remove([save_weights + '/' + el for el in os.listdir(save_weights) if 'csv_reward_weight' in el][0])

    torch.save(reward_model.state_dict(), save_weights + '/csv_reward_weight_lr' + str(lr) + '_k' + str(K) + '_' + current_time + '.pth')
    if default_path != None:
        torch.save(reward_model.state_dict(), default_path + '/csv_reward_weight_lr' + str(lr) + '_k' + str(K) + '.pth')
