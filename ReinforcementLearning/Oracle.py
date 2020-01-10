import os
import sys
import cv2
import time
import numpy as np

from Utility.utility import read_csv_grids

def no_move_count(grid):
    # count for how much frames are equal in the clip

    counts = np.zeros(len(grid))

    for i in range(len(grid)):
        for j in range(i, len(grid)):
            if np.array_equal(grid[i], grid[j]):
                counts[i] += 1
            
    return int(max(counts))

def count_reward(grid, matrix):

    x_old, y_old = 1, 1
    reward = 0

    for i in range(len(grid)):
        x, y = np.where(grid[i] == 10)
        if (x >= x_old or y >= y_old) and (matrix[x, y] != 0):
            reward += matrix[x, y]
            x_old = x
            y_old = y
        else:
            reward = 0
            break
    
    return reward


def give_pref(reward_1, reward_2):
    preference = None

    if reward_1 > 1.5 and reward_2 > 1.5:
        preference = [0.5, 0.5]

    elif reward_1 > 1.5:
        preference = [1, 0]

    elif reward_2 > 1.5:
        preference = [0, 1]

    else:
        preference = [0, 0]
    
    return preference


def createOracleMatrix(wrapper, env):

    # Define the wrapper and initialize an image to create
    # the oracle predefined reward matrix
    img = wrapper.observation(env.reset())    

    # Create a sizexsize matrix of zero
    oracle_rewards = np.zeros((img.shape[0], img.shape[1]))

    # Define the path that the agent must learn to do
    oracle_rewards[1:-1, 1] = 1
    oracle_rewards[-2, 1:-1] = 1

    # Define the reward for each position in the grid
    ind8 = np.where(img == 8) 
    oracle_rewards[ind8[0], ind8[1]] = 10

    for col in range(2, len(oracle_rewards)//2):
        for row in range(1, len(oracle_rewards)-1):
            if oracle_rewards[row][col] == 0:
                m = max(list(oracle_rewards[row-1:row+2, col-1:col+1].flatten()))
                if m == 10: 
                    m = 1 
                elif m <= 0.1:
                    m = 0
                oracle_rewards[row][col] = m / 2

    for row in range(len(oracle_rewards)-2, 1, -1):
        for col in range(1, len(oracle_rewards)-1):
            if oracle_rewards[row][col] == 0:
                m = max(list(oracle_rewards[row-1:row+2, col-1:col+1].flatten()))
                if m == 10:
                    m = 1 
                elif m <= 0.1:
                    m = 0
                oracle_rewards[row][col] = m / 2
    
    return oracle_rewards

class Oracle:
    def __init__(self, wrapper, env):
        
        self._matrix = createOracleMatrix(wrapper, env)


    def takeReward(self, data_path, clip_1, clip_2, env):

        file_1 = [file_ for file_ in os.listdir(data_path + '/' + clip_1['path']) if 'grid_' in file_]
        file_2 = [file_ for file_ in os.listdir(data_path + '/' + clip_2['path']) if 'grid_' in file_]

        states_1 = read_csv_grids(data_path + '/' + clip_1['path'] + '/' + file_1[0], env)
        states_2 = read_csv_grids(data_path + '/' + clip_2['path'] + '/' + file_2[0], env)

        # conta quante volte consegutivamente sono rimasto fermo
        # da 3 in su si scarta altrimenti lo tengo
        count_1 = no_move_count(states_1)
        count_2 = no_move_count(states_2)

        reward_1 = count_reward(states_1, self._matrix)
        reward_2 = count_reward(states_2, self._matrix)
        preference = None

        print(count_1, count_2)

        # In the two clips the agent mooves at least twice
        if count_1 <= 3 and count_2 <= 3:
            preference = give_pref(reward_1, reward_2)
        
        # In one clips make at least 3 moves and in the another clips no
        elif count_1 <= 3:
            preference = give_pref(reward_1, 0)

        # In one clips make at least 3 moves and in the another clips no
        elif count_2 <= 3:
            preference = give_pref(0, reward_2)

        elif (count_1 == 4 and reward_1 >= 5) and (count_2 == 4 and reward_2 >= 5):
            preference = [0.5, 0.5]
        
        elif count_1 == 4 and reward_1 >= 5:
            preference = [1, 0]
        
        elif count_2 == 4 and reward_2 >= 5:
            preference = [0, 1]
        
        else: 
            preference = [0, 0]

        print(preference)

        return preference