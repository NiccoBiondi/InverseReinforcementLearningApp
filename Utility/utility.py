import csv
import cv2
import sys
import os 
import re
import time
import pandas as pd
import numpy as np 
import shutil
import matplotlib.pyplot as plt 

from ReinforcementLearning.policy import save_policy_weights

# save the loss list. Is used when the application is closed and the user
# don't save the losses graphics. 
def save_losses_list(path, losses):
    
    if os.path.exists(path):
        os.remove(path)
        
    with open(path, 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(losses)

# define a functioon to read and load the losses
def load_losses(path):
    data_df = pd.read_csv(path , error_bad_lines=False, nrows=1)
    return [float(loss) for loss in data_df.columns]

# save the policy model during the policy training
def save_model(path, policy, model_parameters, iteration, policy_losses):

    if not os.path.exists(path):
        os.makedirs(path)
    
    if policy_losses:
        save_losses_list(path + '/policy_losses.csv', policy_losses)
        
    save_policy_weights(policy, path)
    save_model_parameters(path, model_parameters, iteration)


# Simple utility function to read states saved in csv file
def read_csv_clips(dir_path):
    data_df = pd.read_csv(dir_path , error_bad_lines=False, names=["state"])

    states = []
    for element in data_df['state'].values:
        result = convert_string(element)
        result = np.reshape(result, (7, 7, 3))
        states.append(result)
    return states

# Simple utility function to read states saved in csv file
def read_csv_grids(dir_path, env):
    data_df = pd.read_csv(dir_path , error_bad_lines=False, names=["grid"])

    states = []
    for element in data_df['grid'].values:
        result = convert_string(element)
        result = np.reshape(result, (env.width, env.height))
        states.append(result)
    return states

# Create the csv file  containing the clips generated
def save_clips(name, idx, clips, clips_grid):

    for num_clips, clip in enumerate(clips):
        save_path = name + '/clip_' + str(idx)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        with open(save_path + '/grid_' + str(idx) + '.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for i in range(len(clips_grid[num_clips])):
                lines = [clips_grid[num_clips][i][:,:,0].T]
                filewriter.writerow(lines)
        
        with open(save_path + '/clip_' + str(idx) + '.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for i in range(len(clip)):
                lines = [clip[i]['obs']]
                filewriter.writerow(lines)
                plt.imsave(save_path + '/fig_' + str(i) + '.png', clip[i]['image'])
        
        idx += 1

    return idx

# Generate clips from trajectory, make sure to take the actions to achieve the goal
def clips_generator(states, dones, clips_len, grids):

    # define the clips grid partition
    total_clips_grid = []
    clips_grid = []

    # define the trajectory partition
    total_clips = []
    clips = []

    # save the movement to achieve the goal
    clips_goal = []
    clips_grid_goal = []

    # If the trajectory is not divisible for length clips, then 
    # it is sure that the agent achieve the goal
    diff = len(states) % clips_len

    if (diff == 1) and (True in dones):
        clips_goal.append(states[len(states) - 2])
        clips_grid_goal.append(grids[len(grids) - 2])

        goal = states.pop()
        grid_goal = grids.pop()

        for i in range(clips_len - 1):
            clips_goal.append(goal)
            clips_grid_goal.append(grid_goal)

    elif (diff >  1) and (True in dones):

        clips_goal = [states.pop() for i in range(diff)]
        clips_grid_goal = [grids.pop() for i in range(diff)]

        clips_goal = clips_goal[::-1]
        clips_grid_goal = clips_grid_goal[::-1]

        for i in range(clips_len - len(clips_goal)):
            clips_goal.append(clips_goal[-1])
            clips_grid_goal.append(clips_grid_goal[-1])

    for i in range(0, len(states)): 

        if len(clips) != clips_len:
            clips.append(states[i])
            clips_grid.append(grids[i])
            
        elif len(clips) == clips_len:
            total_clips.append(clips)
            total_clips_grid.append(clips_grid)

            clips = [states[i]]
            clips_grid = [grids[i]]
    
    if len(clips_goal) != 0:
        total_clips.insert(0, clips_goal)
        total_clips_grid.insert(0, clips_grid_goal)
    
    return total_clips, total_clips_grid

# Save model parameters define in initialization or in a loaded checkpoint.
# Is usefull to restart from the checkpoint
def save_model_parameters(path, model_parameters, iteration):
    '''
        minigrid_env : gym minigrid environment name

        episode_len : trajectory length 

        lr : policy and reward model learning rate

        clips_len : clips length

        episodes : number of iterations made by the policy
 
        K : mini-batches number for reward model train phase

        n_annotation : number of annotation that the user has to do

        idx : current number of clips set created by policy

        iteration : current episode made by the policy
        
    '''
    current_time = time.strftime("%H:%M", time.localtime())
    if [path + '/' + el for el in os.listdir(path) if 'values' in el]:
        os.remove([path + '/' + el for el in os.listdir(path) if 'values' in el][0])

    with open(path + '/values_' + current_time + '.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow([model_parameters['minigrid_env'], model_parameters['episode_len'], 
                                    model_parameters['lr'], model_parameters['clips_len'], model_parameters['episodes'], 
                                    model_parameters['K'], model_parameters['n_annotation'], model_parameters['idx'], iteration])

# Function to save annotation buffer. It is used to restart annotation
#  and reload what the user do in previous work.
def save_annotation(save_path, annotation_buffer, iteration):
    triple_number = 0
    path = save_path + '/annotation_buffer'

    # If the annotation_buffer folder not exists, it is created
    if not os.path.exists(path):
        os.makedirs(path)
    
    else:

        # If the annotation_buffer folder exists, then is taken the save point to save the new value.
        triple = os.listdir(path)
        triple_number = [int(re.findall('\d+', t)[0]) for t in triple]
        if not triple_number:
            triple_number = 0
        else:
            triple_number = max(triple_number) + 1
    
    # There we discard all the triple with preference [0, 0]
    annotation_buffer = [triple for triple in annotation_buffer if triple[2] != [0, 0]]

    for i, triple in enumerate(annotation_buffer):
        with open(path + '/triple_' + str(triple_number) + '.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)

            # In a triple.csv file in each row is set a first clip stete,
            # a second clip state, the preferency, the folder where the annotation
            # arrived and the clip in the folder that are annoted. 
            for idx, clip in enumerate(triple[0]):
                filewriter.writerow([clip, triple[1][idx], triple[2], iteration])

        triple_number += 1


# Simple function to convert str matrix or list in integer matrix or list
def convert_string(image):
    num = []
    for element in image.split():
        if len(re.findall('\d+', element)) > 0: 
            num.append(int(re.findall('\d+', element)[0]))
    
    return np.asarray(num)

# Function to load previous model parameters saved in previous work             
def load_values(path):
    values = {}
    data_df = pd.read_csv(path , error_bad_lines=False, names=["minigrid_env", "episode_len", "lr", "clips_len", "episodes", "K", "n_annotation", "idx", 'iteration'])
    values['minigrid_env'] = str(data_df["minigrid_env"].values[0])
    values['episode_len'] = int(data_df["episode_len"].values[0])
    values['lr'] = float(data_df["lr"].values[0])
    values['clips_len'] = int(data_df["clips_len"].values[0])
    values['episodes'] = int(data_df["episodes"].values[0])
    values['K'] = int(data_df["K"].values[0])
    values['n_annotation'] = int(data_df["n_annotation"][0])
    values['idx'] = int(data_df["idx"].values[0])

    return values, int(data_df["iteration"].values[0])

# Function to load the previous annotation made in previous work
def load_annotation_buffer(load_path):

    shape = (7, 7, 3)
    annotation_buffer = []
    iteration = []
    
    if len(os.listdir(load_path)) > 0:
        
        for triple in sorted(os.listdir(load_path)):

            data_df = pd.read_csv(load_path + triple , error_bad_lines=False, names=["clip_1", "clip_2", "pref", "iteration"])

            clip_1 = []
            clip_2 = []
            pref = [int(x) for x in re.findall('\d+', data_df['pref'][0])]

            iteration.append(data_df["iteration"].values[0])
            
            for idx, element in enumerate(data_df["clip_1"].values):
                img_1 = convert_string(element)
                img_2 = convert_string(data_df["clip_2"].values[idx])
                clip_1.append(np.reshape(img_1, shape))
                clip_2.append(np.reshape(img_2, shape))

            annotation_buffer.append([clip_1, clip_2, pref])
    
    if iteration == []:
        iteration = 0
    else:
        iteration = max(iteration)
    
    return annotation_buffer, iteration
        


