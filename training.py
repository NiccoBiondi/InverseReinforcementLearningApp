from ReinforcementLearning.policy import Policy, run_episode, save_policy_weights, Loss
from ReinforcementLearning.csvRewardModel import csvRewardModel, save_reward_weights
from ReinforcementLearning.Oracle import Oracle
from ReinforcementLearning.wrapper import FullyObsWrapper, RGBImgObsWrapper
from Utility.annotator import Annotator
from Utility.utility import clips_generator, save_clips

import gym, gym_minigrid, torch, copy, tqdm, shutil, os, argparse
import numpy as np

# Simple contructor function.
def data_loader(annotation_buffer, batch):
    
    if len(annotation_buffer) < batch:
        return annotation_buffer

    else:
        index = np.random.randint(0, len(annotation_buffer), size = batch)
        train_data = [annotation_buffer[i] for i in index]  

        return train_data

# hyperparameters

DIR_NAME = os.path.dirname(os.path.abspath('__file__'))

parser = argparse.ArgumentParser()
parser.add_argument("--e", dest="epochs", default=50, type=int) 
args = parser.parse_args()

epochs = args.epochs
save_path = DIR_NAME + '/zz_saving'
clips_database = save_path +'/clips_db'
weigth_path = save_path +'/weigths'
episodes = 500
env_name = 'MiniGrid-Empty-6x6-v0'
checkpoint = 100
obs_size = 7*7    # MiniGrid uses a 7x7 window of visibility.
act_size = 7      # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
inner_size = 64   # Number of neurons in two hidden layers.
reward_batch = 16 
K = 1000

lr_reward = 1e-4
lr_policy = 5e-4

if os.path.exists(clips_database):
    shutil.rmtree(clips_database)
os.makedirs(clips_database)

if not os.path.exists(weigth_path):
    os.makedirs(weigth_path)

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

trainable = False


# MAIN LOOP 

for epoch in range(epochs):

    if os.path.exists(clips_database):
        shutil.rmtree(clips_database)
        os.makedirs(clips_database)

    # POLICY

    print('starting policy at epoch ', epoch)    

    idx = 0
    l = []

    for step in tqdm.tqdm(range(episodes)):
                        
        (states, actions, dones, grids) = run_episode(env, policy, 80, grid_wrapper)

        states_copy = copy.deepcopy(states)
        clips, clips_grid = clips_generator(states_copy, dones, 5, grids)
        idx = save_clips(clips_database, idx, clips, clips_grid)
        
        if step > 0 and step % checkpoint == 0:
            save_policy_weights(policy, weigth_path)
            
        if trainable:
            s = [obs['obs'] for obs in states]
            reward =reward_model(s)
            rewards += [reward[i].item() for i in range(len(reward))]
            for i in range(len(reward)):
               reward[i] = ( reward[i].item() - np.mean(rewards) ) / 0.05 
            losses = Loss(policy, optimizer_p, states, actions, reward)
            for i in range(len(losses)):
                l.append(losses[i])

        save_policy_weights(policy, weigth_path)

    if trainable:
        print("End train policy at epoch {}, the loss is : {:.3f}".format(epoch, (sum(l)/len(l))))

    # ORACLE 

    print('starting annotation')

    clips_number = int(   len(os.listdir(clips_database))  * 0.5 / 2 )
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

    loss = []
    # annotation_buffer = [ triple for triple in annotation_buffer if triple[2] != [0,0] ]
    # print(len(annotation_buffer))
    for k in tqdm.tqdm(range(K)):
        train_clips = data_loader(annotation_buffer, 16)
        loss.append(reward_model.compute_rewards(reward_model, optimizer_r, train_clips))

    save_reward_weights(reward_model, weigth_path, None, lr_reward, K)
    if not trainable:
        trainable = True

    print("End train reward model at epoch {}, the loss is : {:.3f}".format(epoch, (sum(loss)/len(loss))))
