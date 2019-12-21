import os
import sys
import csv
import matplotlib.pyplot as plt

# Create the csv file 
def save_clips(name, clips):

    for num_clips, clip in enumerate(clips):

        save_path = name + '/clip_' + str(num_clips)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        
        with open(save_path + '/clip_' + str(num_clips) + '.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for i in range(len(clip)):
                lines = [clip[i]['obs']]
                filewriter.writerow(lines)
                plt.imsave(save_path + '/fig_' + str(i) + '.png', clip[i]['image'])


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
        total_clisp.insert(0, clips)
    
    return total_clisp

def save_model_parameters(path, model_parameters):
    with open(path + '/values.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow([model_parameters['minigrid_env'], model_parameters['episode_len'], 
                                    model_parameters['lr'], model_parameters['clips_len'], model_parameters['episodes'], 
                                    model_parameters['K'], model_parameters['idx']])