import csv
import os 
import re
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

import gym
import gym_minigrid

# Create the csv file 
def save_clips(name, clips, num_clips):
    save_path = name + '/clip_' + str(num_clips)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    
    with open(save_path + '/clip_' + str(num_clips) + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile)
        for i in range(len(clips)):
            lines = [clips[i]['obs']]
            filewriter.writerow(lines)
            plt.imsave(save_path + '/fig_' + str(i) + '.png', clips[i]['image'])


def clips_generator(states, dones, clips_len): 
    total_clisp = []
    clips = []
    clip_num = 0
    clips_goal = []
    diff = len(states) % clips_len

    #if not os.path.exists(name):
    #        os.mkdir(name) 

    if (diff == 1) and (True in dones):
        clips_goal.append(states[len(states) - 2])
        goal = states.pop()
        for i in range(clips_len - 1):
            clips_goal.append(goal)

    elif (diff >  1) and (True in dones):
        clips_goal = [states.pop() for i in range(diff)]
        clips_goal = clips_goal[::-1]
        for i in range(clips_len - len(clips_goal)):
            clips_goal.append(clips_goal[-1])

    for i in range(0, len(states)): 

        if len(clips) != clips_len:
            clips.append(states[i])
            
        elif len(clips) == clips_len:
            #save_clips(name, clips, clip_num, obs)
            total_clisp.append(clips)
            clip_num += 1
            clips = [states[i]]
    
    if len(clips_goal) != 0:
        #save_clips(name, clips_goal, clip_num + 1, obs)
        total_clisp.insert(0, clips_goal)
    
    return total_clisp

def convert_string(image):
    num = []
    for element in image.split():
        if len(re.findall('\d+', element)) > 0: 
            num.append(int(re.findall('\d+', element)[0]))
    
    return np.asarray(num)

                      
def read_csv_clips(dir_path):
    data_df = pd.read_csv(dir_path , error_bad_lines=False, names=["state"])

    states = []
    for element in data_df['state'].values:
        result = convert_string(element)
        result = np.reshape(result, (7, 7, 3))
        states.append(result)
    return states

def load_values(path):
    values = {}
    data_df = pd.read_csv(path , error_bad_lines=False, names=["minigrid_env", "episode_len", "lr", "clips_len", "episodes", "K", "idx", 'iteration'])
    values['minigrid_env'] = str(data_df["minigrid_env"].values[0])
    values['episode_len'] = int(data_df["episode_len"].values[0])
    values['lr'] = float(data_df["lr"].values[0])
    values['clips_len'] = int(data_df["clips_len"].values[0])
    values['episodes'] = int(data_df["episodes"].values[0])
    values['K'] = int(data_df["K"].values[0])
    values['idx'] = int(data_df["idx"].values[0])

    return values, int(data_df["iteration"].values[0])

def load_annotation_buffer(load_path):

    shape = (7, 7, 3)
    annotation_buffer = []
    iteration = None
    for triple in os.listdir(load_path):
        data_df = pd.read_csv(load_path + triple , error_bad_lines=False, names=["clip_1", "clip_2", "pref", "iteration"])

        clip_1 = []
        clip_2 = []
        #pref = list(data_df['pref'][0])

        pref = [int(x) for x in re.findall('\d+', data_df['pref'][0])]

        if iteration == None:
            iteration = int(data_df["iteration"][0])
    
        for idx, element in enumerate(data_df["clip_1"].values):
            img_1 = convert_string(element)
            img_2 = convert_string(data_df["clip_1"].values[idx])
            clip_1.append(np.reshape(img_1, shape))
            clip_2.append(np.reshape(img_2, shape))

        annotation_buffer.append([clip_1, clip_2, pref])
    
    return annotation_buffer, iteration
        

def save_annotation_buffer(values, annotator_buffer, iteration, save_path):

    with open(save_path + '/values.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow([values['env_name'], values['episode_len'], values['lr'], values['clips_len'], values['episodes'], values['K'], values['idx']])

    if len(annotator_buffer) > 0:
        # Nota : prima posizione prima clip, seconda posizione seconda clip, terza posizione preferenza, quarta posizione iterazione 
        if not os.path.exists(save_path + '/annotation_buffer'):
            os.makedirs(save_path + '/annotation_buffer')

        for i, triple in enumerate(annotator_buffer):
            with open(save_path + '/annotation_buffer/triple_' + str(i) + '.csv', 'w') as csvfile:
                filewriter = csv.writer(csvfile)
                for idx, clip in enumerate(triple[0]):
        
                    filewriter.writerow([clip, triple[1][idx], triple[2], iteration])


