import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from PyQt5.QtWidgets import QApplication


def state_filter(state):
    return torch.from_numpy(state[:,:,0]).float()

class csvRewardModel(nn.Module):
    def __init__(self, obs_size, inner_size):
        super(csvRewardModel, self).__init__()

        self.affine1 = nn.Linear(obs_size, inner_size)
        self.affine2 = nn.Linear(inner_size, 1)

        self.clips_loss = []

    def forward(self, clip):
        rewads = []
        for obs in clip:
            x_1 = state_filter(obs).cuda().view(-1, 7*7)
            x_1 = F.relu(self.affine1(x_1))
            rewads.append(self.affine2(x_1))
        
        return rewads

    
    def compute_rewards(self, reward_model, optimizer, train_clips, logModel):
        
        probs = []
        reward_model.train()
        optimizer.zero_grad()

        for idx, triple in enumerate(train_clips):

            logModel.log('Processing triple : ...')
            QApplication.processEvents()
            
            reward_1 = reward_model.forward(triple[0])
            reward_2 = reward_model.forward(triple[1])
            
            den = (torch.exp(sum(reward_1)) + torch.exp(sum(reward_2)))
            p_sigma_1 = torch.exp(sum(reward_1)) / den
            p_sigma_2 = torch.exp(sum(reward_2)) / den

            probs.append([p_sigma_1, p_sigma_2, triple[2]])

        loss = 0
        for element in probs:
            loss -= ( (element[2][0] * torch.log(element[0])) + (element[2][1] * torch.log(element[1])) )
            
        loss.backward() 
    
        # nn.utils.clip_grad_norm(reward_model.parameters(), 5)
        optimizer.step()

        reward_model.eval()

        return loss.item()

def save_reward_weights(reward_model, save_weights):
    torch.save(reward_model.state_dict(), save_weights + '/csv_reward_weight.pth')

