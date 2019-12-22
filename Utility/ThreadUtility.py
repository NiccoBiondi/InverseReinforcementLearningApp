import os
import sys
import csv
import matplotlib.pyplot as plt

from ReinforcementLearning.policy import save_policy_weights


def save_model(path, policy, model_parameters, iteration):

    if not os.path.exists:
        os.makedirs(path)
    save_policy_weights(policy, path)
    save_model_parameters(path, model_parameters, iteration)

# Create the csv file 
def save_clips(name, clips):

    clips_path = []

    for num_clips, clip in enumerate(clips):
        clips_path.append(name.split('/')[-1] + '/clip_' + str(num_clips))
        save_path = name + '/clip_' + str(num_clips)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        
        with open(save_path + '/clip_' + str(num_clips) + '.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for i in range(len(clip)):
                lines = [clip[i]['obs']]
                filewriter.writerow(lines)
                plt.imsave(save_path + '/fig_' + str(i) + '.png', clip[i]['image'])
                
    return clips_path

def clips_generator(states, dones, clips_len): 
    total_clisp = []
    clips = []
    clip_num = 0
    clips_goal = []
    diff = len(states) % clips_len

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
            total_clisp.append(clips)
            clip_num += 1
            clips = [states[i]]
    
    if len(clips_goal) != 0:
        total_clisp.insert(0, clips_goal)
    
    return total_clisp

def save_model_parameters(path, model_parameters, iteration):
    with open(path + '/values.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow([model_parameters['minigrid_env'], model_parameters['episode_len'], 
                                    model_parameters['lr'], model_parameters['clips_len'], model_parameters['episodes'], 
                                    model_parameters['K'], model_parameters['idx'], iteration])

def save_annotation(self, save_path, annotation_buffer, iteration):

    if not os.path.exists(save_path + '/annotation_buffer'):
            os.makedirs(save_path + '/annotation_buffer')

    for i, triple in enumerate(annotation_buffer):
        with open(save_path + '/annotation_buffer/triple_' + str(i) + '.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for idx, clip in enumerate(triple[0]):
    
                filewriter.writerow([clip, triple[1][idx], triple[2], iteration])