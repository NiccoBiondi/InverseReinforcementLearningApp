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
            if np.array_equal(grid[i][:,:,0], grid[j][:,:,0]):
                counts[i] += 1
            
    return int(max(counts))

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
    
    return oracle_rewards.T

class Oracle:
    def __init__(self, wrapper, env):
        
        self._matrix = createOracleMatrix(wrapper, env)


    def takeReward(self, data_path, clip_1, clip_2, env):
        #TODO: creare sta funzione

        file_1 = [file_ for file_ in os.listdir(data_path + '/' + clip_1['path']) if 'grid_' in file_]
        file_2 = [file_ for file_ in os.listdir(data_path + '/' + clip_2['path']) if 'grid_' in file_]

        states_1 = read_csv_grids(data_path + '/' + clip_1['path'] + '/' + file_1[0], env)
        states_2 = read_csv_grids(data_path + '/' + clip_2['path'] + '/' + file_2[0], env)

        #for i in range(len(clip_1)):

        # conta quante volte consegutivamente sono rimasto fermo
        # da 3 in su si scarta altrimenti lo tengo
        count_1 = no_move_count(states_1)
        count_2 = no_move_count(states_2)

        print(count_1, count_2)
        
        # Vedi come fare, io penso che sarebbe bellino passargli le clips e 
        # lui ritorna la preferenza [1,0], [0,1], [0,0] in base al reward poi vedi te
        sys.exit()

        return [1, 0]