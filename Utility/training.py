import gym, gym_minigrid, torch, copy, tqdm, shutil, os, sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from ReinforcementLearning.policy import Policy, run_episode, save_policy_weights, Loss, compute_discounted_rewards
from ReinforcementLearning.csvRewardModel import csvRewardModel, save_reward_weights
from ReinforcementLearning.Oracle import Oracle
from ReinforcementLearning.wrapper import FullyObsWrapper, RGBImgObsWrapper
from Utility.annotator import Annotator
from Utility.utility import clips_generator, save_clips


def plot_loss(losses, name, save_path):
    
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    if 'reward' in name:
        t = 'Reward Loss'
    else:
        t = 'Policy Loss'
    plt.title(t)
    save_path = save_path + '/' + name
    plt.savefig(save_path)
    plt.show()

# Simple contructor function.
def data_loader(annotation_buffer, batch):
    
    if len(annotation_buffer) < batch:

        return annotation_buffer

    else:
        
        idx = np.random.randint(0, len(annotation_buffer), size = batch)
        train_data = [annotation_buffer[i] for i in idx]

        return train_data

# hyperparameters

DIR_NAME = os.path.dirname(os.path.abspath('__file__'))

parser = argparse.ArgumentParser()
parser.add_argument("--e", dest="epochs", default=50, type=int)
parser.add_argument("--env", dest="env", default='MiniGrid-Empty-6x6-v0', type=str)
args = parser.parse_args()

epochs = args.epochs
env_name = args.env

save_path = DIR_NAME + '/training_saves/' + env_name  +  '_' + str(args.epochs) + 'e'
clips_database = save_path +'/clips_db'
weigth_path = save_path +'/weigths'

checkpoint = 100
obs_size = 7*7    # MiniGrid uses a 7x7 window of visibility.
act_size = 7      # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
inner_size = 64  # Number of neurons in two hidden layers.


reward_batch = 16
K = 1500
percentile = 1

episodes = 150
len_episode = 150
len_clip = 5

lr_reward = 3e-4
lr_policy = 1e-4

if os.path.exists(clips_database):
    shutil.rmtree(clips_database)
os.makedirs(clips_database)

env = gym.make(env_name)
grid_wrapper = FullyObsWrapper(env)
oracle = Oracle(grid_wrapper, env, None)
env = RGBImgObsWrapper(env)
reward_model = csvRewardModel(obs_size = obs_size, inner_size = inner_size).cuda()
policy = Policy(obs_size = obs_size, act_size = act_size, inner_size = inner_size).cuda()
optimizer_p = torch.optim.Adam(params=policy.parameters(), lr = lr_policy, weight_decay=0.01)
optimizer_r = torch.optim.Adam(params=reward_model.parameters(), lr = lr_reward, weight_decay=0.01)

annotator = Annotator()
annotation_buffer = []
rewards = []

# Losses 

policy_loss = []
reward_loss = []
reward_loss_01 = []
reward_loss_05 = []

trainable = False       # during the first iteration the policy will not be trained

# MAIN LOOP 

try:

    for epoch in range(args.epochs):

        if os.path.exists(clips_database):
            shutil.rmtree(clips_database)
            os.makedirs(clips_database)

        # POLICY

        print('starting policy at epoch {}/{}' .format(epoch, args.epochs))    

        idx = 0
        l = []

        for step in tqdm.tqdm(range(episodes)):
                            
            (states, actions, dones, grids) = run_episode(env, policy, len_episode, grid_wrapper)

            states_copy = copy.deepcopy(states)
            clips, clips_grid = clips_generator(states_copy, dones, len_clip, grids)
            idx = save_clips(clips_database, idx, clips, clips_grid)
                
            if trainable:

                reward = reward_model([obs['obs'] for obs in states[1:]]) 
                reward = [r.item() for r in reward]  
                
                for r in reward:
                    if len(rewards) == 300:
                        rewards.pop(0)
                    rewards.append(r)

                # Rewards standardization with std = 0.5 and mean = 0
                for i in range(len(reward)):
                    reward[i] = ( ( reward[i] - np.mean(rewards) ) / ( np.std(rewards) + 1e-7 ) ) * 0.6

                # Compute the discounted rewards 
                discounted_rewards = compute_discounted_rewards(reward)
                
                losses = Loss(policy, optimizer_p, states, actions, discounted_rewards)
                    
                for i in range(len(losses)):
                    l.append(losses[i])

        if not os.path.exists(weigth_path + str(epoch + 1)):
            os.mkdir(weigth_path + str(epoch + 1))

        save_policy_weights(policy, weigth_path + str(epoch + 1))

        if trainable:
            policy_loss.append(sum(l)/len(l))
            print("End train policy at epoch {}, the loss is : {:.3f}".format(epoch, policy_loss[-1]))

        # ORACLE 

        if epoch > 0 and epoch % 5 == 0 and percentile > 0.2 :

           percentile -= 0.2

           if percentile <= 0.2:
               percentile = 0.2

        if (epoch + 1) <= 25:

            #clips_number = int(   ( len(os.listdir(clips_database))  * percentile ) / 2 )
            clips_number = int(len(os.listdir(clips_database)) / 2)
            for i in tqdm.tqdm(range(0, clips_number)):
                
                clips, _ = annotator.load_clips_figure(clips_database)                    
                clip_1 = clips.pop(0)
                clip_2 = clips.pop(0)
                preferences = oracle.takeReward(clips_database, clip_1, clip_2, env)
                annotation_buffer.append([clip_1['clip'], clip_2['clip'], preferences])
                shutil.rmtree(clips_database+'/'+clip_1['path'])
                shutil.rmtree(clips_database+'/'+clip_2['path'])

            # REWARD MODEL

            print('starting reward model training')
            # We create three buffer for storing:
            # - the entaire loss of the iteration training (loss)
            # - the indifferent label loss (losses_05)
            # - the clips with label (0,1) or (1,0)
            # We separated those different components to study
            # the grown of the reward model loss during the training epochs
            loss = []
            losses_05 = []
            losses_01 = []

            print('using the {:}% of the annotation' .format(int(percentile*100)))
            annotation_buffer = [ triple for triple in annotation_buffer if triple[2] != [0,0] ]
        
            # K = len(annotation_buffer) * 4

            for k in tqdm.tqdm(range(K)):
                train_clips = data_loader(annotation_buffer, reward_batch)
                loss_batch, loss_05, loss_01 = reward_model.compute_rewards(reward_model, optimizer_r, train_clips)

                loss.append(loss_batch)
                losses_05.append(loss_05)
                losses_01.append(loss_01)

            save_reward_weights(reward_model, weigth_path + str(epoch + 1), None, lr_reward, K)
            if not trainable:
                trainable = True

            reward_loss_01.append(sum(losses_01)/len(losses_01))
            reward_loss_05.append(sum(losses_05)/len(losses_05))
            reward_loss.append(sum(loss)/len(loss))
            print("End train reward model at epoch {}, the loss is : {:.3f}".format(epoch, reward_loss[-1]))
            annotation_buffer = []
    
    plot_loss(reward_loss, 'reward_loss_' + str(args.epochs) + '.png', save_path)
    plot_loss(policy_loss, 'policy_loss_' + str(args.epochs) + '.png', save_path)
    plot_loss(reward_loss_01, 'reward_loss_01_' + str(args.epochs) + '.png', save_path)
    plot_loss(reward_loss_05, 'reward_loss_05_' + str(args.epochs) + '.png', save_path)

except Exception as e:
    print(e)

    plot_loss(reward_loss, 'reward_loss_' + str(args.epochs) + '.png', save_path)
    plot_loss(policy_loss, 'policy_loss_' + str(args.epochs) + '.png', save_path)
    plot_loss(reward_loss_01, 'reward_loss_01_' + str(args.epochs) + '.png', save_path)
    plot_loss(reward_loss_05, 'reward_loss_05_' + str(args.epochs) + '.png', save_path)
