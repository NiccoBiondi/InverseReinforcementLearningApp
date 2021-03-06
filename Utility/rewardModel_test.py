import numpy as np
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import curses
import time
from optparse import OptionParser

import gym
import gym_minigrid
import torch

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from ReinforcementLearning.csvRewardModel import csvRewardModel
from ReinforcementLearning.wrapper import FullyObsWrapper

KEY_NUMBER = {
    259: 'UP',
    261: 'RIGHT',
    260: 'LEFT',
    339: 'PAGE_UP',
    338: 'PAGE_DOWN',
    32: 'SPACE',
    330: 'ESCAPE',
    10: 'RETURN',
    114: 'BACKSPACE'
}


# Reset the environment
def resetEnv(env, wrapper):
    env.reset()
    wrapper.observation(env.reset())
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)


# Function to compute discounted rewards after a complete episode.
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        discounted_rewards.append(running)
    return list(reversed(discounted_rewards))


# Function that transform the key in command
def keyDownCb(keyValue, env):
    if keyValue == 114:
        resetEnv(env)
        return

    if keyValue == 330:
        sys.exit(0)

    action = 0

    if keyValue == 260:
        action = env.actions.left
    elif keyValue == 261:
        action = env.actions.right
    elif keyValue == 259:
        action = env.actions.forward

    elif keyValue == 32:
        action = env.actions.toggle
    elif keyValue == 339:
        action = env.actions.pickup
    elif keyValue == 338:
        action = env.actions.drop

    elif keyValue == 10:
        action = env.actions.done

    else:
        print("unknown key %s" % keyValue)
        return

    return action


# The user move the agent in a popup window and a reward model predict the reward for the current agent state.
# At the end of the script, it will generate a normalized heat map of the behaviour of the reward model.
# Possible choices for the heat map are max (mode=None) or mean (mode=mean). In the former case in each cell 
# we consider only the max predicted reward. In the latter one we compute the mean between each reward obtained in the same cell. 
def main(mode=None):
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-6x6-v0'
    )
    parser.add_option(
        "-r",
        "--reward",
        dest="reward_model",
        help="path to reard model weigth",
        default=None
    )

    (options, args) = parser.parse_args()

    inner_size = 64
    obs_size = 7 * 7

    reward_model = csvRewardModel(obs_size, inner_size)
    if options.reward_model == None:
        print('define reward model weigth path')
        sys.exit()

    reward_model.load_state_dict(torch.load(options.reward_model))
    reward_model.cuda()

    damn = curses.initscr()
    damn.nodelay(1)
    damn.keypad(True)

    # Load the gym environment
    env = gym.make(options.env_name)
    wrapper = FullyObsWrapper(env)

    obs = resetEnv(env, wrapper)

    renderer = env.render('human')
    number = 1

    rewards = np.zeros((env.width, env.height))
    counts = np.zeros((env.width, env.height))
    states = []

    while True:

        env.render('human')
        time.sleep(0.01)
        try:
            c = damn.getch()
            if c > 1:

                action = keyDownCb(c, env)

                obs, reward, done, info = env.step(action)

                img = wrapper.observation(obs)
                img = img[:, :, 0].T
                x_ag, y_ag = np.where(img == 10)

                states.append([x_ag[0], y_ag[0]])
                reward = reward_model([obs['image']])

                print('step=%s, reward=%.6f' % (env.step_count, reward[0]))

                if mode == 'mean':
                    rewards[x_ag[0], y_ag[0]] += reward[0]
                    counts[x_ag[0], y_ag[0]] += 1

                elif mode == 'max':
                    if rewards[x_ag[0], y_ag[0]] == 0 or reward[0] > rewards[x_ag[0], y_ag[0]]:
                        rewards[x_ag[0], y_ag[0]] = reward[0]

                if done:
                    print('done!')
                    states = []
                    env.reset()

                env.render('human')
                time.sleep(0.01)

        except:
            print()

        # If the window was closed
        if renderer.window == None:
            heatmap_reward(rewards, counts, mode)
            break


def heatmap_reward(rewards, counts, mode=None):
    if mode == 'mean':
        for i in range(1, len(rewards) - 1):
            for j in range(1, len(rewards) - 1):
                if counts[i, j] != 0:
                    rewards[i, j] = rewards[i, j] / counts[i, j]

    # Remove the gray cell of th environment
    rewards = rewards[1:-1, 1:-1]

    # Normalize rewards array to 0 1 for visualization
    rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())

    fig = plt.figure()
    ax = sns.heatmap(rewards, xticklabels=False, yticklabels=False)
    fig.add_subplot(1, 1, 1)
    fig.savefig('Heatmap_images/' + mode + '_heatmap.png')


def plot_loss():
    return 0


if __name__ == '__main__':
    main(mode='max')
