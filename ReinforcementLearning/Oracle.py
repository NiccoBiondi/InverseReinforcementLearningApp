import cv2
import numpy as np
from gym_minigrid.wrappers import FullyObsWrapper


def createOracleMatrix(env):

    # Define the wrapper and initialize an image to create
    # the oracle predefined reward matrix
    wrapper = FullyObsWrapper(env)
    dict_ = wrapper.observation(env.reset())    
    img = dict_['image']

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
    
    print(oracle_rewards)
    return oracle_rewards, wrapper

class Oracle:
    def __init__(self, env):
        
        self._matrix, self._wrapper = createOracleMatrix(env)

    def takeReward(self, clip_1, clip_2):
        #TODO: creare sta funzione
        
        # Vedi come fare, io penso che sarebbe bellino passargli le clips e 
        # lui ritorna la preferenza [1,0], [0,1], [0,0] in base al reward poi vedi te


        return [1, 0]