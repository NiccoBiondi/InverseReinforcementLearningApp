import os
import time

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

# Simple class to calculate states rewards. It takes a clip, collection of states, and process them
# giving back a list of reward, of for each state.
class csvRewardModel(nn.Module):
    def __init__(self, obs_size, inner_size, **kwargs):
        super(csvRewardModel, self).__init__(**kwargs)
        self.affine1 = nn.Linear(obs_size, inner_size)
        self.affine2 = nn.Linear(inner_size, 1)

    def forward(self, clip):
        rewards = []

        # (batch, dim_ch, width, height)
        for obs in clip:

            x_1 = state_filter(obs).cuda().view(-1, 7*7)
            x_1 = F.relu(self.affine1(x_1))
            rewards.append( self.affine2(x_1))
        
        return rewards

    # The primary function. It takes the annotation buffer and process the triples.
    # A triple is a list composed by [first clip, second clip, preferency]. Thre preferency
    # is a list that can be [1, 0] if the user choose the firt clip, [0, 1] if the
    # user choose the second clip, [0.5, 0.5] if the user choose both clips or
    # [0, 0] if the user discard the clips in triple. the preferency is used in loss computation.
    def compute_rewards(self, reward_model, optimizer, train_clips):

        probs = []
        optimizer.zero_grad()

        for idx, triple in enumerate(train_clips):
                        
            reward_1 = reward_model.forward(triple[0])
            reward_2 = reward_model.forward(triple[1])
            
            den = (torch.exp(sum(reward_1)) + torch.exp(sum(reward_2))) + 1e-7

            p_sigma_1 = torch.exp(sum(reward_1)) / den
            p_sigma_2 = torch.exp(sum(reward_2)) / den

            probs.append([p_sigma_1, p_sigma_2, triple[2]])

        loss = 0
        for element in probs:
            loss -= ( (element[2][0] * torch.log(element[0])) + (element[2][1] * torch.log(element[1])) )
            
        loss.backward() 

        # nn.utils.clip_grad_norm_(reward_model.parameters(), 5)
        optimizer.step()

        return loss.item()

# Simple utility function to save the reward model weights
def save_reward_weights(reward_model, save_weights, default_path, lr, K, reward_losses=None):

    if reward_losses != None and reward_losses:
        save_losses_list(save_weights + '/reward_model_losses.csv', reward_losses)

    current_time = time.strftime("%H:%M", time.localtime())
    if [save_weights + '/' + el for el in os.listdir(save_weights) if 'csv_reward_weight' in el]:
        os.remove([save_weights + '/' + el for el in os.listdir(save_weights) if 'csv_reward_weight' in el][0])

    torch.save(reward_model.state_dict(), save_weights + '/csv_reward_weight_lr' + str(lr) + '_k' + str(K) + '_' + current_time + '.pth')
    if default_path != None:
        torch.save(reward_model.state_dict(), default_path + '/csv_reward_weight_lr' + str(lr) + '_k' + str(K) + '.pth')

